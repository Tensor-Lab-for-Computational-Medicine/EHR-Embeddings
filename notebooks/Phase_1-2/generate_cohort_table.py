import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import logging

# Configuration
HDF_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'all_hourly_data.h5')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'phase_1_outputs')
SEED = 42
STRATIFICATION_TARGET = 'mort_hosp'

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load and filter data by age from HDF5."""
    logging.info(f"Loading data from {HDF_FILE_PATH}...")
    try:
        with pd.HDFStore(HDF_FILE_PATH, 'r') as store:
            df_patients = store['/patients']
            
        # Standardize columns
        df_patients.columns = df_patients.columns.str.strip()
        
        # Age filtering (18-125)
        if 'age' in df_patients.columns:
            # Ensure we filter by subject's first recorded age if multiple (though likely static in this table)
            # This follows the logic in data_preprocessing_LOS.py
            valid_subjects = df_patients.groupby('subject_id')['age'].first()
            valid_subjects = valid_subjects[(valid_subjects >= 18) & (valid_subjects <= 125)].index
            df_patients = df_patients[df_patients.index.get_level_values('subject_id').isin(valid_subjects)]
            logging.info(f"Filtered by age (18-125): {len(df_patients)} ICU stays")
        
        return df_patients
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def create_splits(df_patients):
    """Recreate subject-level train/val/test splits."""
    logging.info("Creating splits...")
    subject_outcomes = df_patients.groupby('subject_id')[STRATIFICATION_TARGET].max()
    subjects, outcomes = subject_outcomes.index.values, subject_outcomes.values
    
    train_val_subjects, test_subjects, _, _ = train_test_split(
        subjects, outcomes, test_size=0.25, random_state=SEED, stratify=outcomes
    )
    train_subjects, val_subjects, _, _ = train_test_split(
        train_val_subjects, subject_outcomes[train_val_subjects], 
        test_size=0.125, random_state=SEED, 
        stratify=subject_outcomes[train_val_subjects]
    )
    
    splits = {
        'Training Set': train_subjects,
        'Validation Set': val_subjects,
        'Test Set': test_subjects
    }
    return splits

def group_ethnicity(eth):
    eth = str(eth).upper()
    if 'WHITE' in eth: return 'White'
    if 'BLACK' in eth: return 'Black'
    if 'ASIAN' in eth: return 'Asian'
    if 'HISPANIC' in eth or 'LATINO' in eth: return 'Hispanic'
    return 'Other'

def group_service(unit):
    unit = str(unit).upper()
    if unit in ['MICU', 'CCU']: return 'Medical'
    if unit in ['SICU', 'CSRU', 'TSICU']: return 'Surgical'
    return 'Other'

def group_admission_type(atype):
    atype = str(atype).upper()
    if 'EMERGENCY' in atype or 'URGENT' in atype: return 'Emergency'
    if 'ELECTIVE' in atype: return 'Elective'
    return 'Other'

def calculate_p_values(df, col_name, is_categorical):
    """Calculate p-value across Train, Val, Test groups."""
    # Filter to only the split groups
    df_subset = df[df['Split'].isin(['Training Set', 'Validation Set', 'Test Set'])].copy()
    
    groups = [df_subset[df_subset['Split'] == 'Training Set'][col_name].dropna(),
              df_subset[df_subset['Split'] == 'Validation Set'][col_name].dropna(),
              df_subset[df_subset['Split'] == 'Test Set'][col_name].dropna()]
    
    if is_categorical:
        # Chi-square test
        contingency = pd.crosstab(df_subset[col_name], df_subset['Split'])
        if contingency.empty or contingency.shape[0] < 2:
            return np.nan
        chi2, p, dof, ex = stats.chi2_contingency(contingency)
        return p
    else:
        # Kruskal-Wallis test
        if any(len(g) == 0 for g in groups):
            return np.nan
        stat, p = stats.kruskal(*groups)
        return p

def format_continuous(series):
    return f"{series.median():.1f} [{series.quantile(0.25):.1f}-{series.quantile(0.75):.1f}]"

def format_categorical(series, total_n):
    return f"{series.sum():,}"

def format_p_value(p):
    if pd.isna(p): return "-"
    if p < 0.001: return "<0.001"
    if p > 0.99: return ">0.99"
    return f"{p:.3f}"

def generate_table_row(df_full, label, col_name, is_categorical=False, category_value=None, indent=False, calc_p=True):
    """Generate a row for the table with stats for each split and p-value."""
    row = {'Characteristic': f"\\hspace{{1em}}{label}" if indent else label}
    
    # Total Cohort
    n_total = len(df_full)
    if is_categorical:
        if category_value is not None:
            subset = (df_full[col_name] == category_value)
        else:
            subset = df_full[col_name].astype(bool)
        count = subset.sum()
        row['Total Cohort'] = f"{count:,}"
    else:
        row['Total Cohort'] = format_continuous(df_full[col_name])

    # Splits
    splits = ['Training Set', 'Validation Set', 'Test Set']
    for split_name in splits:
        df_split = df_full[df_full['Split'] == split_name]
        n_split = len(df_split)
        
        if is_categorical:
            if category_value is not None:
                subset_split = (df_split[col_name] == category_value)
            else:
                subset_split = df_split[col_name].astype(bool)
            
            count_split = subset_split.sum()
            row[split_name] = f"{count_split:,}"
        else:
            row[split_name] = format_continuous(df_split[col_name])

    # P-value calculation
    # For categorical specific values (e.g. Race=White), we compare that binary category across splits
    if calc_p:
        if is_categorical:
            if category_value is not None:
                df_full['temp_binary'] = (df_full[col_name] == category_value).astype(int)
                p_val = calculate_p_values(df_full, 'temp_binary', True)
            else:
                p_val = calculate_p_values(df_full, col_name, True)
        else:
            p_val = calculate_p_values(df_full, col_name, False)
        row['P-value'] = format_p_value(p_val)
    else:
        row['P-value'] = ""
        
    return row

def add_header_row(label):
    return {
        'Characteristic': f"\\textbf{{{label}}}",
        'Total Cohort': '', 'Training Set': '', 'Validation Set': '', 'Test Set': '', 'P-value': ''
    }

import pickle

def load_split_ids():
    """Load existing split IDs from pickle files."""
    splits = {}
    for split_name, file_name in [
        ('Training Set', 'icustay_ids_train.pkl'),
        ('Validation Set', 'icustay_ids_val.pkl'),
        ('Test Set', 'icustay_ids_test.pkl')
    ]:
        path = os.path.join(OUTPUT_DIR, file_name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                ids = pickle.load(f)
                # Handle if it's a dataframe/series/index
                if hasattr(ids, 'values'):
                    ids = ids.values
                splits[split_name] = ids
            logging.info(f"Loaded {len(ids)} IDs for {split_name}")
        else:
            logging.warning(f"Split file not found: {path}")
    return splits

def main():
    setup_logging()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df_patients = load_data()
    
    # Load predefined splits instead of recreating them
    split_ids = load_split_ids()
    
    if not split_ids:
        logging.error("No split files found. Cannot generate table based on exact cohort.")
        return

    # Assign splits and filter cohort
    df_patients['Split'] = np.nan
    
    # Map splits
    # Reset index level if icustay_id is in index (it usually is level 2 in MIMIC HDF5)
    # The load_data function returns df_patients with MultiIndex (subject_id, hadm_id, icustay_id)
    # We need to ensure we match on icustay_id
    
    # Check index structure
    if 'icustay_id' in df_patients.index.names:
        icustay_index = df_patients.index.get_level_values('icustay_id')
    elif 'icustay_id' in df_patients.columns:
        icustay_index = df_patients['icustay_id']
    else:
        logging.error("icustay_id not found in patient dataframe.")
        return

    valid_mask = np.zeros(len(df_patients), dtype=bool)
    
    for split_name, ids in split_ids.items():
        # Find which rows have these icustay_ids
        mask = icustay_index.isin(ids)
        df_patients.loc[mask, 'Split'] = split_name
        valid_mask |= mask
        
    # Filter to only patients in the splits
    df_patients = df_patients[valid_mask].copy()
    
    logging.info(f"Final cohort size: {len(df_patients)} (Train: {len(df_patients[df_patients['Split']=='Training Set'])}, Val: {len(df_patients[df_patients['Split']=='Validation Set'])}, Test: {len(df_patients[df_patients['Split']=='Test Set'])})")
    
    # --- Feature Engineering ---
    logging.info("Processing features...")
    df_patients['sex_female'] = (df_patients['gender'] == 'F').astype(int)
    df_patients['race_group'] = df_patients['ethnicity'].apply(group_ethnicity)
    df_patients['admission_group'] = df_patients['admission_type'].apply(group_admission_type)
    df_patients['service_group'] = df_patients['first_careunit'].apply(group_service)
    
    # Severity - Placeholder
    for col in ['oasis', 'sofa', 'los_icu']:
        if col not in df_patients.columns: df_patients[col] = np.nan
        
    # --- Generate Table Rows ---
    rows = []
    
    # Get Ns for headers
    n_total = len(df_patients)
    n_train = len(df_patients[df_patients['Split'] == 'Training Set'])
    n_val = len(df_patients[df_patients['Split'] == 'Validation Set'])
    n_test = len(df_patients[df_patients['Split'] == 'Test Set'])
    
    # 1. Demographics
    rows.append(add_header_row('Demographics'))
    rows.append(generate_table_row(df_patients, 'Age, median [IQR], y', 'age', is_categorical=False, indent=True))
    
    # Sex
    rows.append({'Characteristic': '\\hspace{1em}Sex, No.', 'Total Cohort': '', 'Training Set': '', 'P-value': format_p_value(calculate_p_values(df_patients, 'sex_female', True))})
    rows.append(generate_table_row(df_patients, 'Female', 'sex_female', is_categorical=True, indent=True, calc_p=False))
    # We could add Male here if desired, but usually Female % is sufficient for binary. 
    # To be explicit like Table 1 usually is:
    # rows.append(generate_table_row(df_patients, 'Male', ...)) 
    
    # Race/Ethnicity
    # Calculate overall p-value for the race_group variable first
    race_p = calculate_p_values(df_patients, 'race_group', True)
    rows.append({'Characteristic': '\\hspace{1em}Race and Ethnicity, No.', 'Total Cohort': '', 'Training Set': '', 'P-value': format_p_value(race_p)})
    
    for race in ['White', 'Black', 'Asian', 'Hispanic', 'Other']:
        # For individual rows, we don't show p-value if we showed the omnibus p-value above
        rows.append(generate_table_row(df_patients, race, 'race_group', is_categorical=True, category_value=race, indent=True, calc_p=False))

    # 2. Clinical Characteristics
    rows.append(add_header_row('Clinical Characteristics'))
    
    # Admission Type
    adm_p = calculate_p_values(df_patients, 'admission_group', True)
    rows.append({'Characteristic': '\\hspace{1em}Admission Type, No.', 'Total Cohort': '', 'Training Set': '', 'P-value': format_p_value(adm_p)})
    rows.append(generate_table_row(df_patients, 'Emergency/Urgent', 'admission_group', is_categorical=True, category_value='Emergency', indent=True, calc_p=False))
    rows.append(generate_table_row(df_patients, 'Elective', 'admission_group', is_categorical=True, category_value='Elective', indent=True, calc_p=False))
    
    # Service
    srv_p = calculate_p_values(df_patients, 'service_group', True)
    rows.append({'Characteristic': '\\hspace{1em}Primary Service, No.', 'Total Cohort': '', 'Training Set': '', 'P-value': format_p_value(srv_p)})
    rows.append(generate_table_row(df_patients, 'Medical', 'service_group', is_categorical=True, category_value='Medical', indent=True, calc_p=False))
    rows.append(generate_table_row(df_patients, 'Surgical', 'service_group', is_categorical=True, category_value='Surgical', indent=True, calc_p=False))
    
    # Severity
    if df_patients['oasis'].notna().any() or df_patients['sofa'].notna().any():
         rows.append({'Characteristic': '\\hspace{1em}Severity at Admission, median [IQR]', 'Total Cohort': '', 'Training Set': '', 'P-value': ''})
         if df_patients['oasis'].notna().any():
            rows.append(generate_table_row(df_patients, 'OASIS Score', 'oasis', is_categorical=False, indent=True))
         if df_patients['sofa'].notna().any():
            rows.append(generate_table_row(df_patients, 'SOFA Score', 'sofa', is_categorical=False, indent=True))
    
    # 3. Outcomes
    rows.append(add_header_row('Outcomes'))
    if 'mort_hosp' in df_patients.columns:
        rows.append(generate_table_row(df_patients, 'Hospital Mortality, No.', 'mort_hosp', is_categorical=True, indent=True))
    if 'readmission_30' in df_patients.columns:
        rows.append(generate_table_row(df_patients, '30-Day Readmission, No.', 'readmission_30', is_categorical=True, indent=True))
    if 'los_icu' in df_patients.columns and df_patients['los_icu'].notna().any():
        rows.append(generate_table_row(df_patients, 'Length of Stay, median [IQR], d', 'los_icu', is_categorical=False, indent=True))
        
    # Create DataFrame
    table_df = pd.DataFrame(rows)
    
    # Rename columns for the display/LaTeX
    col_map = {
        'Total Cohort': f'Total Cohort\\newline(N={n_total:,})',
        'Training Set': f'Training Set\\newline(N={n_train:,})',
        'Validation Set': f'Validation Set\\newline(N={n_val:,})',
        'Test Set': f'Test Set\\newline(N={n_test:,})',
        'P-value': 'P Value'
    }
    table_df = table_df.rename(columns=col_map)
    
    # Fill NaN with empty strings for clean look
    table_df = table_df.fillna('')

    # Save CSV (Raw)
    csv_path = os.path.join(OUTPUT_DIR, 'cohort_table_1.csv')
    table_df.to_csv(csv_path, index=False)
    logging.info(f"Table saved to {csv_path}")

    # Generate LaTeX
    latex_table = table_df.to_latex(
        index=False, 
        escape=False, # We manually escaped % and added latex commands
        column_format='lccccc', 
        header=[c.replace('\\newline', ' ') for c in table_df.columns] # Simple header for text representation if needed, but we override in wrapper
    )
    
    # Custom Table Header Generation for Multiline Headers
    headers = table_df.columns
    header_row = f"{headers[0]} & " + " & ".join([f"\\makecell[c]{{{h}}}" for h in headers[1:]]) + " \\\\"

    # Extract body only (remove booktabs wrapper generated by pandas if any, though standard to_latex is basic)
    # We will reconstruct the table with adjustbox and makecell
    body = []
    for _, row in table_df.iterrows():
        # Clean up row to string
        row_str = " & ".join([str(x) for x in row.values]) + " \\\\"
        body.append(row_str)
    body_str = "\n".join(body)

    latex_standalone = (
        "\\documentclass[border=0pt]{standalone}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{adjustbox}\n"
        "\\usepackage{makecell}\n"
        "\\usepackage{multirow}\n"
        "\\renewcommand\\theadfont{\\bfseries}\n"
        "\\begin{document}\n"
        "\\begin{adjustbox}{width=\\textwidth}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        + header_row + "\n"
        "\\midrule\n"
        + body_str + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{adjustbox}\n"
        "\\end{document}\n"
    )
    
    tex_path = os.path.join(OUTPUT_DIR, 'cohort_table_1.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_standalone)
    logging.info(f"LaTeX table saved to {tex_path}")

    # Markdown
    print("\nCohort Description Table:")
    print(table_df.to_markdown(index=False))

if __name__ == "__main__":
    main()

