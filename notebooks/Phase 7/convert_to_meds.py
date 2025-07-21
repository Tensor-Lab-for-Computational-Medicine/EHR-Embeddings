import os
import sys
from datetime import timedelta
import json
from typing import Dict, List, Set

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
    pa.field('time', pa.timestamp('us'), nullable=False),
    pa.field('code', pa.string(), nullable=False),
    pa.field('numeric_value', pa.float32(), nullable=True),
    # FIX: Removed event_id, text_value, and datetime_value to match what's actually generated
    # and align with the minimal needs of the embedding script.
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
    patients_df['admittime'] = pd.to_datetime(patients_df['admittime'])
    
    # FIX: Calculate birth_date robustly, ensuring age is numeric
    age_for_dob_calc = pd.to_numeric(patients_df['age'], errors='coerce').clip(upper=90)
    age_in_days = (age_for_dob_calc * 365.25).fillna(0).astype(int)
    patients_df['birth_date'] = patients_df['admittime'] - pd.to_timedelta(age_in_days, unit='D')
    print("  --> Calculated approximate birth dates.")

    patients_df['los_3'] = (patients_df['los_icu'] > 3).astype(int)
    patients_df['los_7'] = (patients_df['los_icu'] > 7).astype(int)
    print("  --> Derived LOS labels (los_3, los_7).")

    interventions_df = all_data['interventions']
    intervention_labels = interventions_df.groupby('subject_id')[['vent', 'vaso']].max()
    intervention_labels = intervention_labels.rename(columns={'vent': 'intervention_vent', 'vaso': 'intervention_vaso'})
    patients_df = patients_df.join(intervention_labels, on='subject_id')
    patients_df[['intervention_vent', 'intervention_vaso']] = patients_df[['intervention_vent', 'intervention_vaso']].fillna(0).astype(int)
    print("  --> Derived intervention labels.")

    all_data['patients'] = patients_df
    print("  --> All tables loaded and preprocessed.")
    return all_data

# FIX: Refactored filtering logic to be more robust.
def get_valid_icustay_ids(vitals_df: pd.DataFrame) -> Set[int]:
    """Identifies ICU stay IDs that are long enough for prediction."""
    print("\nStep 2: Identifying valid ICU stays for filtering...")
    patient_max_hours = vitals_df.reset_index().groupby('icustay_id')['hours_in'].max()
    min_required_hours = TIME_WINDOW_CONFIG['WINDOW_SIZE'] + TIME_WINDOW_CONFIG['GAP_TIME']
    valid_icustays = patient_max_hours[patient_max_hours > min_required_hours].index
    print(f"  --> Found {len(valid_icustays)} ICU stays lasting longer than {min_required_hours} hours.")
    return set(valid_icustays)

def create_splits(df_patients: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Creates subject-level train/val/test splits."""
    print("\nStep 3: Creating train/validation/test splits...")
    strat_target = SPLIT_CONFIG['STRATIFICATION_TARGET']
    subject_outcomes = df_patients.groupby('subject_id')[strat_target].max()
    subjects, outcomes = subject_outcomes.index.to_numpy(), subject_outcomes.to_numpy()

    train_val_subjects, test_subjects = train_test_split(subjects, test_size=0.25, random_state=SPLIT_CONFIG['SEED'], stratify=outcomes)
    train_val_outcomes = subject_outcomes.loc[train_val_subjects]
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.125, random_state=SPLIT_CONFIG['SEED'], stratify=train_val_outcomes)

    splits = {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}
    print(f"  --> Splits created: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
    return splits

def _process_event_data(df: pd.DataFrame, admission_info: pd.DataFrame, code_prefix: str, value_col: str = None, melt_id_vars: List[str] = None) -> pd.DataFrame:
    """Helper to process different event types into a standard format."""
    if df.empty:
        return pd.DataFrame()
        
    df = df.reset_index()
    if melt_id_vars:
        df = df.melt(id_vars=melt_id_vars, var_name='code', value_name='value')
        df = df[df['value'].notna()]
        if value_col == 'val': # From interventions, keep only positive events
            df = df[df['value'] == 1]
    
    # Merge with admission info to get admittime and link to the correct ICU stay
    merge_on = ['subject_id', 'hadm_id', 'icustay_id'] if 'icustay_id' in df.columns else ['subject_id', 'hadm_id']
    df = df.merge(admission_info, on=merge_on, how='inner') # Use inner to be safe
    
    # Calculate event time
    if 'hours_in' in df.columns:
        df['time'] = df['admittime'] + pd.to_timedelta(df['hours_in'], unit='h')
    else: # For codes which are recorded at admission time
        df['time'] = df['admittime']
    
    # Format code
    if 'icd9_codes' in df.columns:
        df = df.explode('icd9_codes').dropna(subset=['icd9_codes'])
        df['code'] = code_prefix + df['icd9_codes']
    else:
        df['code'] = code_prefix + df['code']
        
    # Standardize output columns
    cols_to_keep = ['subject_id', 'time', 'code']
    if value_col and value_col != 'val':
        df = df.rename(columns={'value': 'numeric_value'})
        cols_to_keep.append('numeric_value')
        
    return df[cols_to_keep]

def convert_split_to_meds(split_name: str, subject_ids: np.ndarray, filtered_data: Dict[str, pd.DataFrame], output_dir: str):
    """Processes and writes a single data split to a MEDS-FLAT file."""
    print(f"\n--- Processing Split: {split_name.upper()} ---")
    
    # Isolate data for the current split
    split_patients = filtered_data['patients'][filtered_data['patients'].index.get_level_values('subject_id').isin(subject_ids)]
    split_icustays = set(split_patients.index.get_level_values('icustay_id'))
    split_hadms = set(split_patients.index.get_level_values('hadm_id'))
    
    # This dataframe is key for linking events to the correct time
    admission_info = split_patients.reset_index()[['subject_id', 'hadm_id', 'icustay_id', 'admittime']]
    
    # Filter event tables for the current split
    split_codes = filtered_data['codes'][filtered_data['codes'].index.get_level_values('hadm_id').isin(split_hadms)]
    split_interventions = filtered_data['interventions'][filtered_data['interventions'].index.get_level_values('icustay_id').isin(split_icustays)]
    split_vitals = filtered_data['vitals'][filtered_data['vitals'].index.get_level_values('icustay_id').isin(split_icustays)]

    processed_events = [
        _process_event_data(split_codes, admission_info, "ICD9CM/"),
        _process_event_data(split_interventions, admission_info, "intervention/", value_col='val', melt_id_vars=['subject_id', 'hadm_id', 'icustay_id', 'hours_in']),
        _process_event_data(split_vitals, admission_info, "vitals_labs/", value_col='numeric_value', melt_id_vars=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])
    ]
    
    events_df = pd.concat([df for df in processed_events if not df.empty], ignore_index=True)
    if events_df.empty:
        print(f"  --> No events found for split {split_name}. Skipping file write.")
        return

    print(f"  --> Found {len(events_df)} total events for {split_name} split.")

    events_df = events_df.sort_values(by=['subject_id', 'time'])
    events_df = events_df.rename(columns={'subject_id': 'patient_id'})
    
    # Ensure a 'numeric_value' column exists, filling with NaN if not present in some event types
    if 'numeric_value' not in events_df.columns:
        events_df['numeric_value'] = np.nan

    final_df = events_df.astype({'patient_id': 'int64', 'time': 'datetime64[us]', 'code': 'string', 'numeric_value': 'float32'})

    data_dir = os.path.join(output_dir, 'data', split_name)
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "data.parquet")
    table = pa.Table.from_pandas(final_df[MEDS_FLAT_SCHEMA.names], schema=MEDS_FLAT_SCHEMA, preserve_index=False)
    pq.write_table(table, output_path)
    print(f"  --> Successfully wrote data for {split_name} cohort to: {output_path}")

def generate_and_save_labels(split_name: str, subject_ids: np.ndarray, patients_df: pd.DataFrame, target: str, output_dir: str):
    """Generates and saves labels for a given split."""
    split_patients = patients_df[patients_df.index.get_level_values('subject_id').isin(subject_ids)]
    
    # We need one prediction time per patient; use their first valid ICU admission
    prediction_times = split_patients.reset_index().sort_values('admittime').drop_duplicates('subject_id')
    subject_labels = split_patients.groupby('subject_id')[target].max().reset_index()

    labels_df = pd.merge(subject_labels, prediction_times[['subject_id', 'admittime']], on='subject_id')
    labels_df['prediction_time'] = labels_df['admittime'] + timedelta(hours=TIME_WINDOW_CONFIG['WINDOW_SIZE'])
    labels_df = labels_df.rename(columns={'subject_id': 'patient_id', target: 'boolean_value'})
    labels_df['boolean_value'] = labels_df['boolean_value'].astype(bool)

    task_dir = os.path.join(output_dir, 'tasks', target, split_name)
    os.makedirs(task_dir, exist_ok=True)
    labels_path = os.path.join(task_dir, 'labels.parquet')
    
    table = pa.Table.from_pandas(labels_df[LABEL_SCHEMA.names], schema=LABEL_SCHEMA, preserve_index=False)
    pq.write_table(table, labels_path)


def main():
    """Main conversion pipeline."""
    if not os.path.exists(HDF_FILE_PATH):
        sys.exit(f"--- CRITICAL ERROR: File not found at '{HDF_FILE_PATH}' ---")

    print("--- Starting HDF5 to MEDS-FLAT Conversion ---")
    all_data = load_and_preprocess_data(HDF_FILE_PATH)

    # FIX: Centralized filtering logic. First, find all valid ICU stays from the entire dataset.
    valid_icustay_ids = get_valid_icustay_ids(all_data['vitals'])
    
    # Now, filter all dataframes based on this single source of truth.
    filtered_patients = all_data['patients'][all_data['patients'].index.get_level_values('icustay_id').isin(valid_icustay_ids)]
    
    if DEBUG_MODE:
        all_valid_ids = filtered_patients.index.get_level_values('subject_id').unique()
        debug_ids = np.random.choice(all_valid_ids, size=min(len(all_valid_ids), DEBUG_PATIENT_COUNT), replace=False)
        filtered_patients = filtered_patients[filtered_patients.index.get_level_values('subject_id').isin(debug_ids)]
        print(f"\n*** DEBUG MODE: Using a subset of {len(debug_ids)} patients. ***")

    # Re-calculate valid IDs based on the (potentially debug-limited) patient list.
    final_valid_icustays = set(filtered_patients.index.get_level_values('icustay_id'))
    final_valid_hadms = set(filtered_patients.index.get_level_values('hadm_id'))
    final_valid_subjects = set(filtered_patients.index.get_level_values('subject_id'))
    
    print(f"\nFiltering all tables to {len(final_valid_subjects)} patients across {len(final_valid_icustays)} valid ICU stays.")
    
    filtered_data = {
        'patients': filtered_patients,
        'codes': all_data['codes'][all_data['codes'].index.get_level_values('hadm_id').isin(final_valid_hadms)],
        'interventions': all_data['interventions'][
            (all_data['interventions'].index.get_level_values('icustay_id').isin(final_valid_icustays)) &
            (all_data['interventions'].index.get_level_values('hours_in') < TIME_WINDOW_CONFIG['WINDOW_SIZE'])
        ],
        'vitals': all_data['vitals'][
            (all_data['vitals'].index.get_level_values('icustay_id').isin(final_valid_icustays)) &
            (all_data['vitals'].index.get_level_values('hours_in') < TIME_WINDOW_CONFIG['WINDOW_SIZE'])
        ]
    }
    
    # Save patient metadata (the definitive list of patients who should be in the output)
    patients_to_save = filtered_data['patients'].reset_index()
    patients_metadata = patients_to_save[['subject_id', 'birth_date']].drop_duplicates(subset=['subject_id']).rename(columns={'subject_id': 'patient_id'})
    
    os.makedirs(MEDS_OUTPUT_DIR, exist_ok=True)
    metadata_output_path = os.path.join(MEDS_OUTPUT_DIR, "patients.parquet")
    patients_metadata.to_parquet(metadata_output_path)
    print(f"  --> Saved metadata for {len(patients_metadata)} patients to: {metadata_output_path}")

    # Create splits and generate data/labels
    splits = create_splits(filtered_data['patients'])
    for split_name, subject_ids in splits.items():
        convert_split_to_meds(split_name, subject_ids, filtered_data, MEDS_OUTPUT_DIR)
        print(f"\n--- Generating labels for {split_name.upper()} ---")
        for target in TARGET_VARIABLES:
            generate_and_save_labels(split_name, subject_ids, filtered_data['patients'], target, MEDS_OUTPUT_DIR)
        print(f"  --> Finished generating labels for {split_name}.")

    print("\n--- All Splits Processed Successfully! ---")

if __name__ == '__main__':
    main()