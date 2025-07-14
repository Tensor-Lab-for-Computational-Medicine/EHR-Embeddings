import os
import sys
import pandas as pd
import numpy as np
import meds
import json
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timedelta
import tqdm
from sklearn.model_selection import train_test_split

# --- Configuration ---
# 1. Path to your input HDF5 file from mimic-extract
HDF_FILE_PATH = './data/raw/all_hourly_data.h5' 
# 2. Root directory where the new MEDS-formatted splits will be saved
MEDS_OUTPUT_DIR = './data/meds_cohort_split_filtered'
# 3. Configuration for the data splitting process
SPLIT_CONFIG = {
    'TARGET_VARIABLE': 'mort_hosp',
    'SEED': 42,
}
# 4. Configuration for time-window filtering
TIME_WINDOW_CONFIG = {
    'WINDOW_SIZE': 24,  # Observation window (e.g., first 48 hours of data)
    'GAP_TIME': 6,      # Gap after observation window (e.g., 6 hours)
}
# 5. (Optional) Set to True to run on a small subset of patients for quick testing
DEBUG_MODE = True
DEBUG_PATIENT_COUNT = 1000

def create_splits(df_patients, config):
    """
    Creates subject-level train/val/test splits to prevent data leakage.
    This function uses a fixed random seed to ensure the splits are always the same.
    """
    print("\nStep 3: Creating train/validation/test splits by subject_id...")
    
    if config['TARGET_VARIABLE'] not in df_patients.columns:
        raise ValueError(f"Target variable '{config['TARGET_VARIABLE']}' not found in the patient data.")
        
    subject_outcomes = df_patients.groupby('subject_id')[config['TARGET_VARIABLE']].max()
    subjects, outcomes = subject_outcomes.index.values, subject_outcomes.values
    
    train_val_subjects, test_subjects, _, _ = train_test_split(
        subjects, outcomes, test_size=0.25, random_state=config['SEED'], stratify=outcomes
    )
    
    train_subjects, val_subjects, _, _ = train_test_split(
        train_val_subjects, subject_outcomes.loc[train_val_subjects], 
        test_size=0.125, random_state=config['SEED'], 
        stratify=subject_outcomes.loc[train_val_subjects]
    )
    
    splits = {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects,
    }
    
    print(f"  --> Splits created: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)} subjects")
    return splits

def generate_and_save_labels(split_name, subject_ids, patients_df, config, meds_output_dir):
    """Generates and saves labels for a given split according to MEDS LabelSchema."""
    
    target_variable = config['TARGET_VARIABLE']
    window_size = TIME_WINDOW_CONFIG['WINDOW_SIZE']
    
    print(f"\n--- Generating labels for {split_name.upper()} split, target: {target_variable} ---")

    split_patients_df = patients_df[patients_df.index.get_level_values('subject_id').isin(subject_ids)]
    
    # Get the first admission time for each subject to define a prediction time
    first_admissions = split_patients_df.reset_index().sort_values('admittime').drop_duplicates('subject_id').set_index('subject_id')
    
    # Determine labels by taking the max outcome per subject
    subject_labels = split_patients_df.groupby('subject_id')[target_variable].max()
    
    labels_data = []
    for subject_id, label in subject_labels.items():
        if subject_id in first_admissions.index:
            admittime = first_admissions.loc[subject_id, 'admittime']
            prediction_time = admittime + timedelta(hours=window_size)
            
            labels_data.append({
                'subject_id': subject_id,
                'prediction_time': prediction_time,
                'boolean_value': bool(label)
            })

    if not labels_data:
        print(f"  --> No labels generated for {split_name} split.")
        return

    labels_df = pd.DataFrame(labels_data)
    
    # Define the task-specific output directory
    task_dir = os.path.join(meds_output_dir, 'tasks', target_variable, split_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # Write labels to a parquet file
    labels_path = os.path.join(task_dir, 'labels.parquet')

    # The MEDS LabelSchema defines several optional value columns. We must construct
    # a schema that contains only the columns present in our DataFrame to ensure compliance.
    label_schema = pa.schema([
        pa.field(meds.LabelSchema.subject_id_name, meds.LabelSchema.subject_id_dtype, nullable=False),
        pa.field(meds.LabelSchema.prediction_time_name, meds.LabelSchema.prediction_time_dtype, nullable=False),
        pa.field(meds.LabelSchema.boolean_value_name, meds.LabelSchema.boolean_value_dtype, nullable=False),
    ])

    table = pa.Table.from_pandas(labels_df, schema=label_schema)
    pq.write_table(table, labels_path)
    
    print(f"  --> Successfully wrote {len(labels_df)} labels for {split_name} split to: {labels_path}")

def convert_split_to_meds(split_name, subject_ids, all_data, output_dir):
    """Processes and writes a single data split to MEDS format, one file per patient."""
    print(f"\n--- Processing Split: {split_name.upper()} ---")

    patients_df = all_data['patients'][all_data['patients'].index.get_level_values('subject_id').isin(subject_ids)]
    codes_df = all_data['codes'][all_data['codes'].index.get_level_values('subject_id').isin(subject_ids)]
    interventions_df = all_data['interventions'][all_data['interventions'].index.get_level_values('subject_id').isin(subject_ids)]
    vitals_df = all_data['vitals'][all_data['vitals'].index.get_level_values('subject_id').isin(subject_ids)]
    
    all_events = []

    admission_times = patients_df[['admittime']].reset_index().drop_duplicates(subset=['subject_id', 'hadm_id'])
    admission_times['admittime'] = pd.to_datetime(admission_times['admittime'])

    if not codes_df.empty:
        codes_long = codes_df.reset_index().merge(admission_times, on=['subject_id', 'hadm_id'], how='left')
        codes_long['timestamp'] = codes_long.apply(lambda r: r['admittime'] + timedelta(hours=r['hours_in']), axis=1)
        codes_long = codes_long.explode('icd9_codes').dropna(subset=['icd9_codes'])
        for _, row in tqdm.tqdm(codes_long.iterrows(), total=len(codes_long), desc=f"  Codes ({split_name})"):
            all_events.append({
                "subject_id": row['subject_id'], "time": row['timestamp'], "code": f"ICD9CM/{row['icd9_codes']}"
            })
    
    if not interventions_df.empty:
        interventions_long = interventions_df.reset_index().melt(id_vars=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'], var_name='code', value_name='val')
        interventions_long = interventions_long[interventions_long['val'] == 1].merge(admission_times, on=['subject_id', 'hadm_id'], how='left')
        interventions_long['timestamp'] = interventions_long.apply(lambda r: r['admittime'] + timedelta(hours=r['hours_in']), axis=1)
        for _, row in tqdm.tqdm(interventions_long.iterrows(), total=len(interventions_long), desc=f"  Interventions ({split_name})"):
            all_events.append({
                "subject_id": row['subject_id'], "time": row['timestamp'], "code": f"intervention/{row['code']}"
            })

    if not vitals_df.empty:
        vitals_df.columns = vitals_df.columns.get_level_values(0)
        vitals_long = vitals_df.reset_index().melt(id_vars=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'], var_name='code', value_name='numeric_value')
        vitals_long = vitals_long.dropna(subset=['numeric_value']).merge(admission_times, on=['subject_id', 'hadm_id'], how='left')
        vitals_long['timestamp'] = vitals_long.apply(lambda r: r['admittime'] + timedelta(hours=r['hours_in']), axis=1)
        for _, row in tqdm.tqdm(vitals_long.iterrows(), total=len(vitals_long), desc=f"  Vitals/Labs ({split_name})"):
            all_events.append({
                "subject_id": row['subject_id'], "time": row['timestamp'], "code": f"mimic_extract_vitals_labs/{row['code']}", "numeric_value": row['numeric_value']
            })

    print(f"  --> Found {len(all_events)} total events for {split_name} split.")
    
    data_dir = os.path.join(output_dir, 'data', split_name)
    metadata_dir = os.path.join(output_dir, 'metadata')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    if all_events:
        events_df = pd.DataFrame(all_events)
        
        for subject_id, subject_df in tqdm.tqdm(events_df.groupby('subject_id'), desc=f"  Writing patient files ({split_name})"):
            subject_df = subject_df.sort_values(by='time').drop(columns=['subject_id'])
            
            # Ensure schema compliance
            subject_df['time'] = pd.to_datetime(subject_df['time'])
            subject_df['code'] = subject_df['code'].astype(str)

            if 'numeric_value' in subject_df.columns:
                subject_df['numeric_value'] = pd.to_numeric(subject_df['numeric_value'], errors='coerce').astype('float32')
            else:
                subject_df['numeric_value'] = np.nan
                subject_df['numeric_value'] = subject_df['numeric_value'].astype('float32')

            if 'text_value' not in subject_df.columns:
                subject_df['text_value'] = pd.NA
            subject_df['text_value'] = subject_df['text_value'].astype('string')

            # Add subject_id back in
            subject_df['subject_id'] = subject_id
            
            # Reorder columns to match MEDS schema
            cols = ['subject_id', 'time', 'code', 'numeric_value', 'text_value']
            subject_df = subject_df[[col for col in cols if col in subject_df.columns]]
            
            table = pa.Table.from_pandas(subject_df, schema=meds.DataSchema.schema())
            
            output_path = os.path.join(data_dir, f"{subject_id}.parquet")
            pq.write_table(table, output_path)

    # Write metadata
    metadata = {
        "dataset_name": f"MIMIC_Extract_to_MEDS ({split_name.capitalize()} Split, Filtered)",
        "dataset_version": "1.0.0",
        "meds_version": "0.4.0" 
    }
    metadata_path = os.path.join(metadata_dir, 'dataset.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"  --> Successfully wrote data for {split_name} cohort to: {data_dir}")

def main():
    """Main conversion pipeline."""
    if not os.path.exists(HDF_FILE_PATH):
        print(f"--- CRITICAL ERROR: File not found at '{HDF_FILE_PATH}' ---")
        sys.exit(1)

    print("--- Starting HDF5 to MEDS Conversion with Train/Val/Test Splits ---")
    
    with pd.HDFStore(HDF_FILE_PATH, 'r') as store:
        print("\nStep 1: Loading all data tables from HDF5 file...")
        patients_df = store.select('/patients')
        all_data = {
            'patients': patients_df,
            'codes': store.select('/codes'),
            'interventions': store.select('/interventions'),
            'vitals': store.select('/vitals_labs_mean'),
        }
        print("  --> All tables loaded.")

    if DEBUG_MODE:
        print(f"\n--- RUNNING IN DEBUG MODE (first {DEBUG_PATIENT_COUNT} patients) ---")
        unique_subjects = patients_df.index.get_level_values('subject_id').unique()
        if len(unique_subjects) > DEBUG_PATIENT_COUNT:
            selected_subjects = unique_subjects[:DEBUG_PATIENT_COUNT]
            for key in all_data:
                all_data[key] = all_data[key][all_data[key].index.get_level_values('subject_id').isin(selected_subjects)]
            patients_df = all_data['patients']

    # --- ADDED: Time-window filtering logic ---
    print("\nStep 2: Applying time-window filtering...")
    # Using vitals to determine length of stay as it is the densest time-series table
    patient_max_hours = all_data['vitals'].reset_index(level='hours_in').groupby(level='icustay_id')['hours_in'].max()
    
    # Identify ICU stays that are long enough for the observation window and gap
    min_required_hours = TIME_WINDOW_CONFIG['WINDOW_SIZE'] + TIME_WINDOW_CONFIG['GAP_TIME']
    valid_icustays = patient_max_hours[patient_max_hours > min_required_hours].index
    
    # Filter the patient demographic data
    original_patient_count = len(patients_df.index.get_level_values('subject_id').unique())
    patients_df = patients_df[patients_df.index.get_level_values('icustay_id').isin(valid_icustays)]
    all_data['patients'] = patients_df
    
    # Filter all time-series data to include only valid ICU stays and events within the observation window
    for key in ['codes', 'interventions', 'vitals']:
        df = all_data[key]
        df_filtered = df[
            (df.index.get_level_values('icustay_id').isin(valid_icustays)) &
            (df.index.get_level_values('hours_in') < TIME_WINDOW_CONFIG['WINDOW_SIZE'])
        ]
        all_data[key] = df_filtered

    filtered_patient_count = len(patients_df.index.get_level_values('subject_id').unique())
    print(f"  --> Time window filtered: {original_patient_count} -> {filtered_patient_count} patients remaining.")
    # --- END of filtering logic ---
    
    # This creates reproducible splits based on the SEED value using the filtered patient data
    splits = create_splits(patients_df, SPLIT_CONFIG)

    for split_name, subject_ids in splits.items():
        convert_split_to_meds(split_name, subject_ids, all_data, MEDS_OUTPUT_DIR)
        generate_and_save_labels(split_name, subject_ids, all_data['patients'], SPLIT_CONFIG, MEDS_OUTPUT_DIR)

    print("\n--- All Splits Processed Successfully! ---")

if __name__ == '__main__':
    try:
        import meds, tqdm, pandas, tables, pyarrow
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"--- DEPENDENCY ERROR: {e} ---")
        print("Please install required libraries: pip install meds tqdm pandas tables scikit-learn pyarrow")
        sys.exit(1)
        
    main()

