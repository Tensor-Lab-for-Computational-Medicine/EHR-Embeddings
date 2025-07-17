import os
import sys
from datetime import timedelta
import json
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import tqdm

# --- Configuration ---
HDF_FILE_PATH = './data/raw/all_hourly_data.h5'
MEDS_OUTPUT_DIR = './data/meds_cohort_split_filtered'
TARGET_VARIABLES = ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso']
SPLIT_CONFIG = {'STRATIFICATION_TARGET': 'mort_hosp', 'SEED': 42}
TIME_WINDOW_CONFIG = {'WINDOW_SIZE': 24, 'GAP_TIME': 6}
DEBUG_MODE = False
DEBUG_PATIENT_COUNT = 1000

# --- MEDS Schema Definitions ---
MEDS_FLAT_SCHEMA = pa.schema([
    pa.field('patient_id', pa.int64(), nullable=False),
    pa.field('event_id', pa.int64(), nullable=False),
    pa.field('time', pa.timestamp('us'), nullable=False),
    pa.field('code', pa.string(), nullable=False),
    pa.field('numeric_value', pa.float32(), nullable=True),
    pa.field('text_value', pa.string(), nullable=True),
    pa.field('datetime_value', pa.timestamp('us'), nullable=True),
])

LABEL_SCHEMA = pa.schema([
    pa.field('patient_id', pa.int64(), nullable=False),
    pa.field('prediction_time', pa.timestamp('us'), nullable=False),
    pa.field('boolean_value', pa.bool_(), nullable=True),
])

def load_and_preprocess_data(hdf_path: str) -> Dict[str, pd.DataFrame]:
    """Loads data from HDF5 and pre-computes derived labels."""
    print("\nStep 1: Loading and Preprocessing Data...")
    with pd.HDFStore(hdf_path, 'r') as store:
        all_data = {
            'patients': store.select('/patients'),
            'codes': store.select('/codes'),
            'interventions': store.select('/interventions'),
            'vitals': store.select('/vitals_labs_mean'),
        }

    patients_df = all_data['patients']
    if 'los_icu' in patients_df.columns:
        patients_df['los_3'] = (patients_df['los_icu'] > 3).astype(int)
        patients_df['los_7'] = (patients_df['los_icu'] > 7).astype(int)
        print("  --> Derived LOS labels (los_3, los_7).")

    interventions_df = all_data['interventions']
    if 'vent' in interventions_df.columns and 'vaso' in interventions_df.columns:
        intervention_labels = interventions_df.groupby('subject_id')[['vent', 'vaso']].max()
        intervention_labels = intervention_labels.rename(
            columns={'vent': 'intervention_vent', 'vaso': 'intervention_vaso'}
        )
        patients_df = patients_df.join(intervention_labels, on='subject_id')
        patients_df[['intervention_vent', 'intervention_vaso']] = patients_df[
            ['intervention_vent', 'intervention_vaso']
        ].fillna(0).astype(int)
        print("  --> Derived intervention labels.")

    all_data['patients'] = patients_df
    print("  --> All tables loaded and preprocessed.")
    return all_data

def filter_by_time_window(all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Filters data to include only the first `WINDOW_SIZE` hours of ICU stays
    that last longer than `WINDOW_SIZE + GAP_TIME`.
    """
    print("\nStep 2: Applying time-window filtering...")
    
    # 1. Identify valid ICU stays based on length
    patient_max_hours = all_data['vitals'].reset_index().groupby('icustay_id')['hours_in'].max()
    min_required_hours = TIME_WINDOW_CONFIG['WINDOW_SIZE'] + TIME_WINDOW_CONFIG['GAP_TIME']
    valid_icustays = patient_max_hours[patient_max_hours > min_required_hours].index
    
    # 2. Filter patients to only include valid ICU stays
    patients_df = all_data['patients']
    original_patient_count = patients_df.index.get_level_values('subject_id').nunique()
    
    filtered_patients = patients_df[patients_df.index.get_level_values('icustay_id').isin(valid_icustays)]
    
    # Get the unique admission IDs (hadm_id) associated with our valid ICU stays
    valid_hadm_ids = filtered_patients.index.get_level_values('hadm_id').unique()

    # 3. Filter all data tables based on the valid ICU stays and time window
    filtered_data = {'patients': filtered_patients}

    # Filter codes to only include those from the relevant hospital admissions
    filtered_data['codes'] = all_data['codes'][all_data['codes'].index.get_level_values('hadm_id').isin(valid_hadm_ids)]

    # Filter time-series data to the first `WINDOW_SIZE` hours for valid ICU stays
    for key in ['interventions', 'vitals']:
        df = all_data[key]
        time_window_mask = (
            (df.index.get_level_values('icustay_id').isin(valid_icustays)) &
            (df.index.get_level_values('hours_in') < TIME_WINDOW_CONFIG['WINDOW_SIZE'])
        )
        filtered_data[key] = df[time_window_mask]

    filtered_patient_count = filtered_patients.index.get_level_values('subject_id').nunique()
    print(f"  --> Time window filtered: {original_patient_count} -> {filtered_patient_count} patients remaining.")
    print(f"  --> Found {len(valid_icustays)} valid ICU stays.")
    return filtered_data

def create_splits(df_patients: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Creates subject-level train/val/test splits."""
    print("\nStep 3: Creating train/validation/test splits...")
    strat_target = SPLIT_CONFIG['STRATIFICATION_TARGET']
    if strat_target not in df_patients.columns:
        raise ValueError(f"Stratification target '{strat_target}' not found.")

    subject_outcomes = df_patients.groupby('subject_id')[strat_target].max()
    subjects, outcomes = subject_outcomes.index.to_numpy(), subject_outcomes.to_numpy()

    train_val_subjects, test_subjects = train_test_split(
        subjects, test_size=0.25, random_state=SPLIT_CONFIG['SEED'], stratify=outcomes
    )
    train_val_outcomes = subject_outcomes.loc[train_val_subjects]
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=0.125, random_state=SPLIT_CONFIG['SEED'], stratify=train_val_outcomes
    )

    splits = {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}
    print(f"  --> Splits created: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
    return splits

def _process_event_data(df: pd.DataFrame, admission_times: pd.DataFrame, code_prefix: str, value_col: str = None, melt_id_vars: List[str] = None) -> pd.DataFrame:
    """Helper to process different event types into a standard format."""
    if df.empty:
        return pd.DataFrame()
        
    df = df.reset_index()
    if melt_id_vars:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().melt(id_vars=melt_id_vars, var_name='code', value_name=value_col)
        df = df[df[value_col].notna()]
        if value_col == 'val': # From interventions
            df = df[df['val'] == 1]
    
    # FIX: Differentiate merge strategy based on whether icustay_id is already present
    if 'icustay_id' in df.columns:
        # For vitals/interventions, merge on all three keys to prevent creating conflicting columns
        df = df.merge(admission_times, on=['subject_id', 'hadm_id', 'icustay_id'], how='left')
    else:
        # For codes, merge on two keys to bring in the associated icustay_id(s)
        df = df.merge(admission_times, on=['subject_id', 'hadm_id'], how='left')
    
    df.dropna(subset=['icustay_id'], inplace=True)

    if 'hours_in' in df.columns:
        df['time'] = df['admittime'] + pd.to_timedelta(df['hours_in'], unit='h')
    else:
        df['time'] = df['admittime']
    
    if 'icd9_codes' in df.columns: # Explode for codes
        df = df.explode('icd9_codes').dropna(subset=['icd9_codes'])
        df['code'] = code_prefix + df['icd9_codes']
    else:
        df['code'] = code_prefix + df['code']
        
    cols_to_keep = ['subject_id', 'icustay_id', 'time', 'code']
    if value_col and value_col != 'val':
        df = df.rename(columns={value_col: 'numeric_value'})
        cols_to_keep.append('numeric_value')
        
    return df[cols_to_keep]

def convert_split_to_meds(split_name: str, subject_ids: np.ndarray, all_data: Dict[str, pd.DataFrame], output_dir: str):
    """Processes and writes a single data split to a MEDS-FLAT file."""
    print(f"\n--- Processing Split: {split_name.upper()} ---")
    
    split_data = {key: df[df.index.get_level_values('subject_id').isin(subject_ids)] for key, df in all_data.items()}
    
    admission_times = split_data['patients'].reset_index()[['subject_id', 'hadm_id', 'icustay_id', 'admittime']]
    admission_times['admittime'] = pd.to_datetime(admission_times['admittime'])

    processed_events = [
        _process_event_data(split_data['codes'], admission_times, "ICD9CM/"),
        _process_event_data(split_data['interventions'], admission_times, "intervention/", value_col='val', melt_id_vars=['subject_id', 'hadm_id', 'icustay_id', 'hours_in']),
        _process_event_data(split_data['vitals'], admission_times, "vitals_labs/", value_col='numeric_value', melt_id_vars=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
    ]
    
    events_df = pd.concat(processed_events, ignore_index=True)
    print(f"  --> Found {len(events_df)} total events for {split_name} split.")

    if events_df.empty:
        return

    events_df = events_df.sort_values(by=['subject_id', 'icustay_id', 'time'])
    events_df['event_id'] = events_df.groupby(['subject_id', 'icustay_id']).cumcount()
    events_df = events_df.rename(columns={'subject_id': 'patient_id'})

    for col in ['text_value', 'datetime_value']:
        events_df[col] = pd.NA
    
    final_df = events_df.astype({
        'patient_id': 'int64', 'event_id': 'int64', 'time': 'datetime64[us]',
        'code': 'string', 'numeric_value': 'float32', 'text_value': 'string',
        'datetime_value': 'datetime64[us]'
    }, errors='ignore')

    data_dir = os.path.join(output_dir, 'data', split_name)
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "data.parquet")
    table = pa.Table.from_pandas(final_df[MEDS_FLAT_SCHEMA.names], schema=MEDS_FLAT_SCHEMA, preserve_index=False)
    pq.write_table(table, output_path)
    print(f"  --> Successfully wrote data for {split_name} cohort to: {output_path}")

def generate_and_save_labels(split_name: str, subject_ids: np.ndarray, patients_df: pd.DataFrame, target: str, output_dir: str):
    """Generates and saves labels for a given split."""
    print(f"\n--- Generating labels for {split_name.upper()}, target: {target} ---")
    if target not in patients_df.columns:
        print(f"  --> WARNING: Target '{target}' not in patient data. Skipping.")
        return

    split_patients = patients_df[patients_df.index.get_level_values('subject_id').isin(subject_ids)]
    first_admissions = split_patients.reset_index().sort_values('admittime').drop_duplicates('subject_id')
    subject_labels = split_patients.groupby('subject_id')[target].max().reset_index()

    labels_df = pd.merge(subject_labels, first_admissions[['subject_id', 'admittime']], on='subject_id')
    labels_df['prediction_time'] = labels_df['admittime'] + timedelta(hours=TIME_WINDOW_CONFIG['WINDOW_SIZE'])
    labels_df = labels_df.rename(columns={'subject_id': 'patient_id', target: 'boolean_value'})
    labels_df['boolean_value'] = labels_df['boolean_value'].astype(bool)

    task_dir = os.path.join(output_dir, 'tasks', target, split_name)
    os.makedirs(task_dir, exist_ok=True)
    labels_path = os.path.join(task_dir, 'labels.parquet')
    
    table = pa.Table.from_pandas(labels_df[LABEL_SCHEMA.names], schema=LABEL_SCHEMA, preserve_index=False)
    pq.write_table(table, labels_path)
    print(f"  --> Wrote {len(labels_df)} labels to: {labels_path}")

def main():
    """Main conversion pipeline."""
    if not os.path.exists(HDF_FILE_PATH):
        sys.exit(f"--- CRITICAL ERROR: File not found at '{HDF_FILE_PATH}' ---")

    print("--- Starting HDF5 to MEDS-FLAT Conversion ---")
    
    all_data = load_and_preprocess_data(HDF_FILE_PATH)

    if DEBUG_MODE:
        print(f"\n--- RUNNING IN DEBUG MODE (first {DEBUG_PATIENT_COUNT} patients) ---")
        unique_subjects = all_data['patients'].index.get_level_values('subject_id').unique()
        if len(unique_subjects) > DEBUG_PATIENT_COUNT:
            selected_subjects = list(unique_subjects[:DEBUG_PATIENT_COUNT])
            all_data = {k: v[v.index.get_level_values('subject_id').isin(selected_subjects)] for k, v in all_data.items()}

    filtered_data = filter_by_time_window(all_data)
    splits = create_splits(filtered_data['patients'])

    for split_name, subject_ids in splits.items():
        convert_split_to_meds(split_name, subject_ids, filtered_data, MEDS_OUTPUT_DIR)
        for target in TARGET_VARIABLES:
            generate_and_save_labels(split_name, subject_ids, filtered_data['patients'], target, MEDS_OUTPUT_DIR)

    metadata_path = os.path.join(MEDS_OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            "dataset_name": "MIMIC_Extract_to_MEDS (Filtered)",
            "dataset_version": "1.0.0",
            "meds_version": "0.4.0"
        }, f, indent=4)

    print("\n--- All Splits Processed Successfully! ---")

if __name__ == '__main__':
    try:
        import meds, tqdm, pandas, tables, pyarrow
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        sys.exit(f"--- DEPENDENCY ERROR: {e} ---\nPlease install required libraries.")
    main()
