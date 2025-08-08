import os
import sys
from datetime import timedelta, datetime
import json
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import jsonschema
from sklearn.model_selection import train_test_split
import tqdm

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
# Set DEBUG_MODE to True to run on a small subset of patients
DEBUG_MODE = False
DEBUG_PATIENT_COUNT = 1000

# --- MEDS Schema Definitions (Corrected) ---
DATA_SCHEMA = DataSchema(
    subject_id='subject_id',
    time='time',
    code='code',
    numeric_value='numeric_value',
    text_value='text_value'
)
LABEL_SCHEMA = LabelSchema(
    subject_id="subject_id",
    prediction_time="prediction_time",
    boolean_value="boolean_value",
    integer_value="integer_value",
    float_value="float_value",
    categorical_value="categorical_value"
)
SUBJECT_SPLIT_SCHEMA = SubjectSplitSchema(subject_id='subject_id', split='split')
CODE_METADATA_SCHEMA = CodeMetadataSchema(code='code', description='description', parent_codes='parent_codes')
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

    age_for_dob_calc = pd.to_numeric(patients_df['age'], errors='coerce').clip(upper=90)
    age_in_days = (age_for_dob_calc * 365.25).fillna(0).astype(int)
    patients_df['birth_date'] = patients_df['admittime'] - pd.to_timedelta(age_in_days, unit='D')
    print("  --> Calculated approximate birth dates.")

    patients_df['los_3'] = (patients_df['los_icu'] > 3).astype(int)
    patients_df['los_7'] = (patients_df['los_icu'] > 7).astype(int)
    print("  --> Derived LOS labels (los_3, los_7).")

    # Create intervention labels with proper prevalent case handling
    interventions_df = all_data['interventions']
    
    # 1. Create labels from full ICU stay data (for patients who eventually need interventions)
    intervention_labels = interventions_df.groupby('subject_id')[['vent', 'vaso']].max()
    intervention_labels = intervention_labels.rename(columns={'vent': 'intervention_vent', 'vaso': 'intervention_vaso'})
    patients_df = patients_df.join(intervention_labels, on='subject_id')
    patients_df[['intervention_vent', 'intervention_vaso']] = patients_df[['intervention_vent', 'intervention_vaso']].fillna(0).astype(int)
    print("  --> Derived intervention labels from full ICU stay.")
    
    # 2. Identify prevalent cases (patients already on interventions in first 24 hours)
    print("  --> Identifying and excluding prevalent cases for intervention predictions...")
    window_size = TIME_WINDOW_CONFIG['WINDOW_SIZE']  # 24 hours
    prevalent_interventions = interventions_df[
        interventions_df.index.get_level_values('hours_in') < window_size
    ]
    
    # Find patients with interventions in the first 24 hours
    prevalent_vent_subjects = set(
        prevalent_interventions[prevalent_interventions['vent'] > 0]
        .index.get_level_values('subject_id').unique()
    )
    prevalent_vaso_subjects = set(
        prevalent_interventions[prevalent_interventions['vaso'] > 0]
        .index.get_level_values('subject_id').unique()
    )
    
    # 3. Set prevalent cases to NaN (exclude from prediction task)
    if prevalent_vent_subjects:
        vent_prevalent_mask = patients_df.index.get_level_values('subject_id').isin(prevalent_vent_subjects)
        patients_df.loc[vent_prevalent_mask, 'intervention_vent'] = np.nan
        print(f"    --> Excluded {vent_prevalent_mask.sum()} prevalent cases for 'intervention_vent' (already on ventilation in first {window_size}h)")
    
    if prevalent_vaso_subjects:
        vaso_prevalent_mask = patients_df.index.get_level_values('subject_id').isin(prevalent_vaso_subjects)
        patients_df.loc[vaso_prevalent_mask, 'intervention_vaso'] = np.nan
        print(f"    --> Excluded {vaso_prevalent_mask.sum()} prevalent cases for 'intervention_vaso' (already on vasopressors in first {window_size}h)")
    
    print("  --> Intervention labels created with proper prevalent case exclusion.")

    all_data['patients'] = patients_df
    print("  --> All tables loaded and preprocessed.")
    return all_data

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

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]

    if 'icustay_id' in df.columns and 'subject_id' not in df.columns:
        id_map = admission_info[['icustay_id', 'hadm_id', 'subject_id', 'admittime']].drop_duplicates(subset=['icustay_id'])
        df = df.merge(id_map, on='icustay_id', how='inner')
    elif 'hadm_id' in df.columns and 'admittime' not in df.columns:
        id_map = admission_info[['hadm_id', 'subject_id', 'admittime']].drop_duplicates(subset=['hadm_id'])
        cols_to_merge = ['hadm_id'] + [col for col in id_map.columns if col not in df.columns]
        df = df.merge(id_map[cols_to_merge], on='hadm_id', how='inner')

    if needs_melting:
        id_vars = ['subject_id', 'hadm_id', 'icustay_id', 'admittime', 'hours_in']
        id_vars = [col for col in id_vars if col in df.columns]
        df = df.melt(id_vars=id_vars, var_name='code', value_name='value')
        df = df[df['value'].notna()]
        if value_col == 'val':
            df = df[df['value'] == 1]

    if 'hours_in' in df.columns:
        df['time'] = df['admittime'] + pd.to_timedelta(df['hours_in'], unit='h')
    else:
        df['time'] = df['admittime']

    if 'icd9_codes' in df.columns:
        df = df.explode('icd9_codes').dropna(subset=['icd9_codes'])
        df['code'] = code_prefix + df['icd9_codes']
    else:
        df['code'] = code_prefix + df['code']

    cols_to_keep = ['subject_id', 'time', 'code']
    if value_col and value_col == 'numeric_value':
        df = df.rename(columns={'value': 'numeric_value'})
        cols_to_keep.append('numeric_value')

    return df[cols_to_keep]

def generate_and_save_labels(patients_df: pd.DataFrame, target: str, output_dir: str):
    """Generates and saves labels for a given task for all patients."""
    prediction_times = patients_df.reset_index().sort_values('admittime').drop_duplicates('subject_id')
    subject_labels = patients_df.groupby('subject_id')[target].max().reset_index()

    labels_df = pd.merge(subject_labels, prediction_times[['subject_id', 'admittime']], on='subject_id')
    labels_df['prediction_time'] = labels_df['admittime'] + timedelta(hours=TIME_WINDOW_CONFIG['WINDOW_SIZE'])
    labels_df = labels_df.rename(columns={target: 'boolean_value'})
    
    # Handle intervention tasks: preserve NaN values for prevalent cases
    if target in ['intervention_vent', 'intervention_vaso']:
        # Keep NaN values as they represent excluded prevalent cases
        # Only convert non-NaN values to boolean
        labels_df['boolean_value'] = labels_df['boolean_value'].apply(
            lambda x: bool(x) if pd.notna(x) else None
        )
        print(f"    --> Preserved {labels_df['boolean_value'].isna().sum()} prevalent cases as NaN for {target}")
    else:
        # For non-intervention tasks, convert to boolean as before
        labels_df['boolean_value'] = labels_df['boolean_value'].astype(bool)

    # Use the correct null type for each column
    labels_df['integer_value'] = pd.NA
    labels_df['float_value'] = np.nan
    labels_df['categorical_value'] = pd.NA

    task_dir = os.path.join(output_dir, 'tasks', target)
    os.makedirs(task_dir, exist_ok=True)
    labels_path = os.path.join(task_dir, 'labels.parquet')
    
    # Ensure the DataFrame being saved matches the full schema and has correct dtypes
    # Handle boolean_value column carefully to preserve NaN for intervention tasks
    if target in ['intervention_vent', 'intervention_vaso']:
        # Use nullable boolean type that can handle NaN
        labels_df['boolean_value'] = labels_df['boolean_value'].astype('boolean')
    else:
        labels_df['boolean_value'] = labels_df['boolean_value'].astype(bool)
    
    final_labels_df = labels_df[LABEL_SCHEMA.schema().names].astype({
        'integer_value': 'Int64',
        'float_value': 'float32',
        'categorical_value': 'string'
    })
    
    table = pa.Table.from_pandas(final_labels_df, schema=LABEL_SCHEMA.schema(), preserve_index=False)
    pq.write_table(table, labels_path)

def main():
    """Main conversion pipeline."""
    if not os.path.exists(HDF_FILE_PATH):
        sys.exit(f"--- CRITICAL ERROR: File not found at '{HDF_FILE_PATH}' ---")

    print("--- Starting HDF5 to MEDS Conversion ---")
    print("--- NOTE: Intervention predictions properly exclude prevalent cases (patients already on interventions in first 24h) ---")

    # Create base directories
    os.makedirs(MEDS_OUTPUT_DIR, exist_ok=True)
    metadata_dir = os.path.join(MEDS_OUTPUT_DIR, 'metadata')
    data_dir = os.path.join(MEDS_OUTPUT_DIR, 'data')
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    all_data = load_and_preprocess_data(HDF_FILE_PATH)

    # Step 2: Identify valid ICU stays
    valid_icustay_ids = get_valid_icustay_ids(all_data['vitals'])
    
    # Filter patients based on valid stays
    filtered_patients = all_data['patients'][all_data['patients'].index.get_level_values('icustay_id').isin(valid_icustay_ids)]
    
    # Apply debug settings if enabled
    if DEBUG_MODE:
        all_valid_ids = filtered_patients.index.get_level_values('subject_id').unique()
        debug_ids = np.random.choice(all_valid_ids, size=min(len(all_valid_ids), DEBUG_PATIENT_COUNT), replace=False)
        filtered_patients = filtered_patients[filtered_patients.index.get_level_values('subject_id').isin(debug_ids)]
        print(f"\n*** DEBUG MODE: Using a subset of {len(debug_ids)} patients. ***")

    # Finalize the sets of valid IDs
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
    
    # Step 3 & 4: Create metadata files
    print("\nStep 4: Creating metadata files...")
    dataset_metadata = {
        "dataset_name": "MIMIC-IV-Sample-ICU", "dataset_version": "1.0", "etl_name": "HDF5 to MEDS ETL",
        "etl_version": "2.1", "meds_version" : "0.3.3", "created_at" : str(datetime.now()),
    }
    jsonschema.validate(instance=dataset_metadata, schema=DATASET_METADATA_SCHEMA.schema())
    metadata_path = os.path.join(metadata_dir, 'dataset.json')
    with open(metadata_path, 'w') as f: json.dump(dataset_metadata, f)
    print(f"  --> Wrote dataset metadata to: {metadata_path}")

    # Create and save splits metadata
    splits = create_splits(filtered_data['patients']) # `splits` is now a dictionary
    split_dfs = []
    split_map = {'train': train_split, 'val': tuning_split, 'test': held_out_split}
    for split_name, subject_ids in splits.items():
        name = split_map[split_name]
        split_dfs.append(pd.DataFrame({'subject_id': subject_ids, 'split': name}))
    
    all_splits_df = pd.concat(split_dfs, ignore_index=True)
    splits_path = os.path.join(metadata_dir, 'subject_splits.parquet')
    pq.write_table(pa.Table.from_pandas(all_splits_df, schema=SUBJECT_SPLIT_SCHEMA.schema()), splits_path)
    print(f"  --> Wrote {len(all_splits_df)} subject splits to: {splits_path}")

    # Step 5: Process all event data into a single dataframe
    print("\nStep 5: Processing all events...")
    admission_info = filtered_data['patients'].reset_index()[['subject_id', 'hadm_id', 'icustay_id', 'admittime']]
    
    processed_events = [
        _process_event_data(filtered_data['codes'], admission_info, "DIAGNOSIS/ICD9CM/"),
        _process_event_data(filtered_data['interventions'], admission_info, "PROCEDURE/intervention/", 
                            value_col='val', needs_melting=True),
        _process_event_data(filtered_data['vitals'], admission_info, "MEASUREMENT/vitals_labs/", 
                            value_col='numeric_value', needs_melting=True)
    ]
    
    events_df = pd.concat([df for df in processed_events if not df.empty], ignore_index=True)
    if events_df.empty:
        print("  --> No events found. Halting.")
        return

    print(f"  --> Processed {len(events_df)} total events.")
    events_df = events_df.sort_values(by=['subject_id', 'time'])
    
    # Ensure all required columns exist and have the right type
    if 'numeric_value' not in events_df.columns: events_df['numeric_value'] = np.nan
    if 'text_value' not in events_df.columns: events_df['text_value'] = pd.NA
    final_df = events_df[DATA_SCHEMA.schema().names].astype({
        'subject_id': 'int64', 'time': 'datetime64[us]', 'code': 'string',
        'numeric_value': 'float32', 'text_value': 'string'
    })

    # --- FIX #1: WRITING PATIENT METADATA (STILL NEEDED) ---
    print("\n--- Writing Patient Metadata ---")
    patients_to_save = filtered_data['patients'].reset_index().rename(columns={'subject_id': 'patient_id'})
    patient_metadata_path = os.path.join(MEDS_OUTPUT_DIR, "patients.parquet")
    pq.write_table(pa.Table.from_pandas(patients_to_save, preserve_index=False), patient_metadata_path)
    print(f"  --> Wrote patient metadata to: {patient_metadata_path}")

    # --- START FIX #2: WRITING SPLIT EVENT DATA ---
    print("\n--- Writing Split Event Data ---")
    # The 'splits' dictionary holds the subject_ids for 'train', 'val', 'test'
    for split_name, subject_ids in splits.items():
        print(f"  --> Processing split: '{split_name}'")

        # Create the directory for the split (e.g., .../data/train)
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Filter the main events dataframe for subjects in the current split
        split_events_df = final_df[final_df['subject_id'].isin(subject_ids)]

        if split_events_df.empty:
            print(f"    --> WARNING: No events found for split '{split_name}'. Skipping file write.")
            continue

        # Define the output path for the split's data
        output_path = os.path.join(split_dir, 'data.parquet')
        
        # Create and write the pyarrow table
        table = pa.Table.from_pandas(split_events_df, schema=DATA_SCHEMA.schema(), preserve_index=False)
        pq.write_table(table, output_path)

        print(f"    --> Wrote {len(split_events_df)} events for {len(subject_ids)} subjects to: {output_path}")
    # --- END FIX #2 ---

    print("\n--- Writing Code Metadata ---")
    unique_codes = final_df['code'].unique()
    codes_df = pd.DataFrame({'code': unique_codes, 'description': ['' for _ in unique_codes], 'parent_codes': [[] for _ in unique_codes]})
    codes_path = os.path.join(metadata_dir, 'codes.parquet')
    pq.write_table(pa.Table.from_pandas(codes_df, schema=CODE_METADATA_SCHEMA.schema()), codes_path)
    print(f"  --> Wrote {len(codes_df)} unique codes to: {codes_path}")

    # Step 6: Generate labels
    print("\nStep 6: Generating and saving labels for all tasks...")
    for target in tqdm.tqdm(TARGET_VARIABLES, desc="Generating Labels"):
        generate_and_save_labels(filtered_data['patients'], target, MEDS_OUTPUT_DIR)
    print("  --> Finished generating all labels.")

    print("\n--- MEDS Conversion Completed Successfully! ---")
    
    print("\nRunning meds_reader to confirm MEDS extract is valid...")
    print("--- NOTE: This validation may fail due to the new split-directory structure, which is expected. ---")
    print("--- The new structure is required by the `generate_climbr_embeddings.py` script. ---")
    db_path = os.path.join(MEDS_OUTPUT_DIR, "processed_db")
    status = os.system(f"meds_reader_convert {MEDS_OUTPUT_DIR} {db_path} --num_threads 5")
    if status == 0:
        print(f"--- meds_reader validation successful! DB created at {db_path} ---")
    else:
        print("--- WARNING: meds_reader_convert failed as expected. The output directory is still likely valid for the embedding script. ---")

if __name__ == '__main__':
    main()