import os
import sys
from datetime import timedelta
import json
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import jsonschema
from sklearn.model_selection import train_test_split
import tqdm
from datetime import datetime
from meds import (
    CodeMetadataSchema,
    DataSchema,
    DatasetMetadataSchema,
    SubjectSplitSchema,
    LabelSchema,
    train_split,
    tuning_split,
    held_out_split,
)


# --- Configuration ---
HDF_FILE_PATH = './data/raw/all_hourly_data.h5'
MEDS_OUTPUT_DIR = './data/meds_cohort_split_filtered'
TARGET_VARIABLES = ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso']
SPLIT_CONFIG = {'STRATIFICATION_TARGET': 'mort_hosp', 'SEED': 42}
TIME_WINDOW_CONFIG = {'WINDOW_SIZE': 24, 'GAP_TIME': 6}
DEBUG_MODE = False
DEBUG_PATIENT_COUNT = 1000

# --- MEDS Schema Definitions ---
# Schemas are now imported from the meds library
DATA_SCHEMA = DataSchema(subject_id='subject_id', time='time', code='code')
LABEL_SCHEMA = LabelSchema(subject_id='subject_id', prediction_time='prediction_time', boolean_value='boolean_value')
SUBJECT_SPLIT_SCHEMA = SubjectSplitSchema(subject_id='subject_id', split='split')
CODE_METADATA_SCHEMA = CodeMetadataSchema(code='code')
DATASET_METADATA_SCHEMA = DatasetMetadataSchema()

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

def _process_event_data(df: pd.DataFrame, admission_info: pd.DataFrame, code_prefix: str, value_col: str = None, needs_melting: bool = False) -> pd.DataFrame:
    """Helper to process different event types into a standard format."""
    if df.empty:
        return pd.DataFrame()

    # CRITICAL FIX for MergeError: This converts the MultiIndex into columns.
    df = df.reset_index()

    # For tables like vitals/interventions, merge in patient identifiers.
    # We use a simple merge because we know these tables don't have subject_id/hadm_id to start with.
    if 'icustay_id' in df.columns and 'subject_id' not in df.columns:
        # We need subject_id, hadm_id, and admittime
        id_map = admission_info[['subject_id', 'hadm_id', 'icustay_id', 'admittime']].drop_duplicates()
        df = df.merge(id_map, on='icustay_id', how='inner')
    # For the codes table, we only need to add admittime.
    elif 'hadm_id' in df.columns and 'admittime' not in df.columns:
        id_map = admission_info[['hadm_id', 'admittime']].drop_duplicates(subset=['hadm_id'])
        df = df.merge(id_map, on='hadm_id', how='inner')


    if needs_melting:
        # These are the identifiers we want to keep after melting.
        # They should all exist in the DataFrame now thanks to the merge above.
        id_vars = ['subject_id', 'hadm_id', 'icustay_id', 'admittime', 'hours_in']
        df = df.melt(id_vars=id_vars, var_name='code', value_name='value')
        df = df[df['value'].notna()]
        if value_col == 'val': # For interventions, keep only positive events
            df = df[df['value'] == 1]

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

def generate_and_save_labels(patients_df: pd.DataFrame, target: str, output_dir: str):
    """Generates and saves labels for a given task for all patients."""
    # We need one prediction time per patient; use their first valid ICU admission
    prediction_times = patients_df.reset_index().sort_values('admittime').drop_duplicates('subject_id')
    subject_labels = patients_df.groupby('subject_id')[target].max().reset_index()

    labels_df = pd.merge(subject_labels, prediction_times[['subject_id', 'admittime']], on='subject_id')
    labels_df['prediction_time'] = labels_df['admittime'] + timedelta(hours=TIME_WINDOW_CONFIG['WINDOW_SIZE'])
    labels_df = labels_df.rename(columns={target: 'boolean_value'})
    labels_df['boolean_value'] = labels_df['boolean_value'].astype(bool)

    task_dir = os.path.join(output_dir, 'tasks', target)
    os.makedirs(task_dir, exist_ok=True)
    labels_path = os.path.join(task_dir, 'labels.parquet')

    table = pa.Table.from_pandas(labels_df[['subject_id', 'prediction_time', 'boolean_value']], schema=LABEL_SCHEMA.schema(), preserve_index=False)
    pq.write_table(table, labels_path)
    print(f"  --> Wrote labels for task '{target}' to: {labels_path}")


def main():
    """Main conversion pipeline."""
    if not os.path.exists(HDF_FILE_PATH):
        sys.exit(f"--- CRITICAL ERROR: File not found at '{HDF_FILE_PATH}' ---")

    print("--- Starting HDF5 to MEDS Conversion ---")

    # Create output directories required by MEDS format
    os.makedirs(MEDS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(MEDS_OUTPUT_DIR, 'metadata'), exist_ok=True)
    os.makedirs(os.path.join(MEDS_OUTPUT_DIR, 'data'), exist_ok=True)
    
    all_data = load_and_preprocess_data(HDF_FILE_PATH)

    valid_icustay_ids = get_valid_icustay_ids(all_data['vitals'])
    
    filtered_patients = all_data['patients'][all_data['patients'].index.get_level_values('icustay_id').isin(valid_icustay_ids)]
    
    if DEBUG_MODE:
        all_valid_ids = filtered_patients.index.get_level_values('subject_id').unique()
        debug_ids = np.random.choice(all_valid_ids, size=min(len(all_valid_ids), DEBUG_PATIENT_COUNT), replace=False)
        filtered_patients = filtered_patients[filtered_patients.index.get_level_values('subject_id').isin(debug_ids)]
        print(f"\n*** DEBUG MODE: Using a subset of {len(debug_ids)} patients. ***")

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
    
    # Step 4: Create and write metadata files
    print("\nStep 4: Creating metadata files...")
    dataset_metadata = {
        "dataset_name": "MIMIC-IV-Sample-ICU",
        "dataset_version": "1.0",
        "etl_name": "HDF5 to MEDS ETL",
        "etl_version": "2.0",
        "meds_version" : "0.3.3",
        "created_at" : str(datetime.now()),
    }
    jsonschema.validate(instance=dataset_metadata, schema=DATASET_METADATA_SCHEMA.schema())
    metadata_path = os.path.join(MEDS_OUTPUT_DIR, 'metadata', 'dataset.json')
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f)
    print(f"  --> Wrote dataset metadata to: {metadata_path}")

    splits = create_splits(filtered_data['patients'])
    split_dfs = []
    split_map = {'train': train_split, 'val': tuning_split, 'test': held_out_split}
    for split_name, subject_ids in splits.items():
        name = split_map[split_name]
        split_dfs.append(pd.DataFrame({'subject_id': subject_ids, 'split': name}))
    
    all_splits_df = pd.concat(split_dfs, ignore_index=True)
    splits_path = os.path.join(MEDS_OUTPUT_DIR, 'metadata', 'subject_splits.parquet')
    table = pa.Table.from_pandas(all_splits_df, schema=SUBJECT_SPLIT_SCHEMA.schema())
    pq.write_table(table, splits_path)
    print(f"  --> Wrote {len(all_splits_df)} subject splits to: {splits_path}")

    # Step 5: Process all events into a single dataframe
    print("\nStep 5: Processing all events...")
    # This admission_info DataFrame is key for adding identifiers to other tables
    admission_info = filtered_data['patients'].reset_index()[['subject_id', 'hadm_id', 'icustay_id', 'admittime']]
    
    # --- FIX START: Simplify the calls to the helper function ---
    processed_events = [
        # The 'codes' table does not need to be melted
        _process_event_data(filtered_data['codes'], admission_info, "DIAGNOSIS/ICD9CM/"),
        
        # 'interventions' needs melting, has a 'val' column to indicate presence
        _process_event_data(filtered_data['interventions'], admission_info, "PROCEDURE/intervention/", 
                            value_col='val', needs_melting=True),
        
        # 'vitals' needs melting, has continuous values
        _process_event_data(filtered_data['vitals'], admission_info, "MEASUREMENT/vitals_labs/", 
                            value_col='numeric_value', needs_melting=True)
    ]
    # --- FIX END ---
    
    
    events_df = pd.concat([df for df in processed_events if not df.empty], ignore_index=True)
    if events_df.empty:
        print("  --> No events found. Halting.")
        return

    print(f"  --> Processed {len(events_df)} total events.")

    events_df = events_df.sort_values(by=['subject_id', 'time'])
    
    if 'numeric_value' not in events_df.columns:
        events_df['numeric_value'] = np.nan

    final_df = events_df.astype({'subject_id': 'int64', 'time': 'datetime64[us]', 'code': 'string', 'numeric_value': 'float32'})

    data_path = os.path.join(MEDS_OUTPUT_DIR, 'data', "data.parquet")
    table = pa.Table.from_pandas(final_df[list(DATA_SCHEMA.keys())], schema=DATA_SCHEMA.schema(), preserve_index=False)
    pq.write_table(table, data_path)
    print(f"  --> Wrote event data to: {data_path}")

    unique_codes = final_df['code'].unique()
    codes_df = pd.DataFrame({'code': unique_codes, 'description': ''})
    codes_path = os.path.join(MEDS_OUTPUT_DIR, 'metadata', 'codes.parquet')
    table = pa.Table.from_pandas(codes_df, schema=CODE_METADATA_SCHEMA.schema())
    pq.write_table(table, codes_path)
    print(f"  --> Wrote {len(codes_df)} unique codes to: {codes_path}")

    # Step 6: Generate and save all labels
    print("\nStep 6: Generating and saving labels for all tasks...")
    for target in tqdm.tqdm(TARGET_VARIABLES, desc="Generating Labels"):
        generate_and_save_labels(filtered_data['patients'], target, MEDS_OUTPUT_DIR)
    print("  --> Finished generating all labels.")

    print("\n--- MEDS Conversion Completed Successfully! ---")

    print("\nRunning meds_reader to confirm MEDS extract is valid...")
    db_path = os.path.join(MEDS_OUTPUT_DIR, "processed_db")
    status = os.system(f"meds_reader_convert {MEDS_OUTPUT_DIR} {db_path} --num_threads 5")
    if status == 0:
        print(f"--- meds_reader validation successful! DB created at {db_path} ---")
    else:
        print("--- WARNING: meds_reader_convert failed. The output MEDS directory may not be valid. ---")


if __name__ == '__main__':
    main()