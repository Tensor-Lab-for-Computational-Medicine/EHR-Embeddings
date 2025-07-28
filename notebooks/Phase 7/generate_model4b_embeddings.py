# generate_model4b_embeddings.py
"""
Generate Model 4B embeddings from MEDS-FLAT data in XGBoost-compatible format.

This script:
1. Reads MEDS-FLAT patient event data (same input as CLIMBR Model 4A)
2. Serializes each ICU stay's events into chronological text strings
3. Generates embeddings using Google Generative AI API in batches
4. Saves embeddings with icustay_id as filenames for XGBoost compatibility
5. Creates label CSV files with icustay_id as index
6. Maintains full traceability metadata

Output structure:
- model_4b_text-embedding-004/
  - MODEL4B_P0/train|val|test/{icustay_id}.npy
  - labels/{task}_{split}_labels.csv
  - embedding_metadata.csv

Compatible with: xgboost_embedding_analysis.py
"""
import argparse
import datetime
import getpass
import logging
import math
import os
import time
from pathlib import Path

import google.generativeai as genai
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Default Configuration ---
# These can be overridden by command-line arguments
DEFAULT_OUTPUT_DIR = Path("./notebooks/Phase 7/model_4b_text-embedding-004")
DEFAULT_MEDS_DIR = Path("./data/meds_cohort_split_filtered")
DEFAULT_MODEL_NAME = "models/text-embedding-004"
DEFAULT_TASK_TYPE = "CLASSIFICATION"
DEFAULT_BATCH_SIZE = 32 # The API is efficient with batches
DEFAULT_RATE_LIMIT_DELAY_S = 0.25 # Seconds to wait between batches

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Model 4B embeddings from MEDS-flat data in XGBoost-compatible format.")
    parser.add_argument("--meds-dir", type=Path, default=DEFAULT_MEDS_DIR, help="Root directory of the MEDS-flat dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root directory to save the final .npy embedding files.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="The embedding model to use (e.g., 'models/text-embedding-004').")
    parser.add_argument("--task-type", type=str, default=DEFAULT_TASK_TYPE, help="The task type for the embedding model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of patients to process in a single API call.")
    parser.add_argument("--experimental-arm", type=str, default="MODEL4B_P0", help="Experimental arm name for XGBoost compatibility.")
    parser.add_argument("--debug", action='store_true', help="If set, process only the first few batches in each split for testing.")
    return parser.parse_args()

def setup_api_key():
    """Securely prompts for and configures the Google AI API key."""
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        print(f"api_key: {api_key}")
        if not api_key:
            logging.info("GOOGLE_API_KEY environment variable not found.")
            api_key = getpass.getpass('Please enter your Google AI Studio API key: ')
        genai.configure(api_key=api_key)
        logging.info("Successfully configured Google AI API key.")
    except Exception as e:
        logging.error(f"Failed to configure API key: {e}")
        exit(1)

def load_embedding_metadata(metadata_path):
    """Load existing embedding metadata if it exists."""
    if metadata_path.exists():
        try:
            return pd.read_csv(metadata_path)
        except Exception as e:
            logging.warning(f"Could not load existing metadata: {e}. Starting fresh.")
    return pd.DataFrame()

def save_embedding_metadata(metadata_df, metadata_path):
    """Save embedding metadata to CSV."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(metadata_path, index=False)

def create_embedding_labels_mapping(meds_dir, target_variables):
    """Create a mapping between subjects and their labels across all tasks."""
    mapping_data = []
    
    for task in target_variables:
        labels_path = meds_dir / "tasks" / task / "labels.parquet"
        if not labels_path.exists():
            logging.warning(f"Labels file not found for task {task}, skipping")
            continue
            
        labels_df = pd.read_parquet(labels_path)
        for _, row in labels_df.iterrows():
            mapping_data.append({
                'subject_id': row['subject_id'],
                'task': task,
                'prediction_time': row['prediction_time'],
                'label_value': row['boolean_value'] if pd.notna(row['boolean_value']) else None
            })
    
    return pd.DataFrame(mapping_data)

def create_xgboost_label_files(meds_dir, output_dir, target_variables, splits_data, patient_metadata):
    """Create XGBoost-compatible label CSV files with icustay_id as index."""
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for task in target_variables:
        labels_path = meds_dir / "tasks" / task / "labels.parquet"
        if not labels_path.exists():
            logging.warning(f"Labels file not found for task {task}, skipping")
            continue
            
        labels_df = pd.read_parquet(labels_path)
        
        # Create label files for each split
        for split_name, icustay_ids in splits_data.items():
            # Map icustay_ids back to subject_ids to get labels
            split_metadata = patient_metadata[patient_metadata['icustay_id'].isin(icustay_ids)]
            split_subject_ids = split_metadata['subject_id'].unique()
            
            split_labels = labels_df[labels_df['subject_id'].isin(split_subject_ids)].copy()
            
            if split_labels.empty:
                logging.warning(f"No labels found for task {task}, split {split_name}")
                continue
            
            # Merge with metadata to get icustay_id as index
            split_labels_with_icustay = pd.merge(
                split_metadata[['subject_id', 'icustay_id']], 
                split_labels, 
                on='subject_id', 
                how='inner'
            )
            
            # Prepare DataFrame with icustay_id as index and task as column
            split_labels_formatted = split_labels_with_icustay.set_index('icustay_id')[['boolean_value']].copy()
            split_labels_formatted.columns = [task]
            
            # Handle NaN values for intervention tasks
            if task in ['intervention_vent', 'intervention_vaso']:
                # Keep NaN values as they represent "prevalent cases" that should be excluded
                pass
            else:
                # For other tasks, fill NaN with False (assuming no event means negative outcome)
                split_labels_formatted[task] = split_labels_formatted[task].fillna(False)
            
            # Convert boolean to int (except for NaN values)
            split_labels_formatted[task] = split_labels_formatted[task].astype('Int64')
            
            # Save the label file
            label_file_path = labels_dir / f"{task}_{split_name}_labels.csv"
            split_labels_formatted.to_csv(label_file_path)
            logging.info(f"Created XGBoost label file: {label_file_path}")
    
    return labels_dir

def serialize_patient_events(subject_id: int, group_df: pd.DataFrame) -> str | None:
    """Processes a single subject's data and returns a serialized text string."""
    try:
        birth_date = group_df['birth_date'].iloc[0]
        if pd.isna(birth_date):
            logging.warning(f"Skipping subject {subject_id} due to missing birth date.")
            return None

        all_event_tuples = [(birth_date, "EVENT/BIRTH")]

        for _, row in group_df.iterrows():
            timestamp = row.get('time')
            code = str(row.get('code', '')).strip()
            numeric_val = row.get('numeric_value')

            if pd.isna(timestamp) or not code:
                continue

            event_str = f"{code}"
            if pd.notna(numeric_val) and isinstance(numeric_val, (int, float)) and math.isfinite(numeric_val):
                event_str += f" | {float(numeric_val):.4f}".rstrip('0').rstrip('.')
            all_event_tuples.append((timestamp, event_str))

        all_event_tuples.sort(key=lambda x: x[0])

        final_text_lines = [f"{ts.strftime('%Y-%m-%d %H:%M:%S')} | {event_data}" for ts, event_data in all_event_tuples]
        return "\n".join(final_text_lines)

    except Exception as e:
        logging.error(f"Failed to serialize events for subject {subject_id}. Error: {e}")
        return None

def process_batch(batch: list[tuple[int, int, str]], model_name: str, task_type: str, output_dir: Path, 
                 split: str, labels_mapping: pd.DataFrame, metadata_records: list, base_output_dir: Path,
                 experimental_arm: str, split_icustay_ids: set):
    """Takes a batch of (icustay_id, subject_id, text), gets embeddings, saves them, and tracks metadata."""
    if not batch:
        return

    icustay_ids = [item[0] for item in batch]
    subject_ids = [item[1] for item in batch]
    content_to_embed = [item[2] for item in batch]

    try:
        result = genai.embed_content(model=model_name, content=content_to_embed, task_type=task_type)
        embeddings = result['embedding']

        for i, embedding_vector in enumerate(embeddings):
            icustay_id = icustay_ids[i]
            subject_id = subject_ids[i]
            output_filepath = output_dir / f"{icustay_id}.npy"
            embedding_array = np.array(embedding_vector)
            np.save(output_filepath, embedding_array)
            
            # Add to split data for XGBoost label generation
            split_icustay_ids.add(icustay_id)
            
            # Create metadata records for traceability
            subject_labels = labels_mapping[labels_mapping['subject_id'] == subject_id]
            for _, label_row in subject_labels.iterrows():
                metadata_records.append({
                    'subject_id': subject_id,
                    'icustay_id': icustay_id,
                    'split': split,
                    'embedding_file': str(output_filepath.relative_to(base_output_dir)),
                    'task': label_row['task'],
                    'prediction_time': label_row['prediction_time'],
                    'label_value': label_row['label_value'],
                    'embedding_shape': embedding_array.shape,
                    'dtype': str(embedding_array.dtype),
                    'generated_at': datetime.datetime.now().isoformat(),
                    'model_name': model_name,
                    'task_type': task_type,
                    'experimental_arm': experimental_arm
                })

    except Exception as e:
        logging.error(f"API call failed for a batch of {len(icustay_ids)} ICU stays. Error: {e}")
        logging.error("Skipping this batch.")

def main():
    """Main function to orchestrate the embedding generation."""
    args = get_args()
    logging.info("--- Starting Integrated Embedding Generation for Model 4B ---")
    logging.info(f"  MEDS Data Dir: {args.meds_dir}")
    logging.info(f"  Output Dir: {args.output_dir}")
    logging.info(f"  Batch Size: {args.batch_size}")
    if args.debug:
        logging.warning("--- DEBUG MODE ENABLED: Processing only a few batches per split. ---")

    setup_api_key()

    # Load patient metadata once
    patient_metadata_path = args.meds_dir / "patients.parquet"
    if not patient_metadata_path.exists():
        logging.error(f"Fatal: Patient metadata not found at '{patient_metadata_path}'"); return
    
    patient_metadata = pd.read_parquet(patient_metadata_path)
    # Ensure consistent use of subject_id and keep icustay_id for XGBoost compatibility
    if 'patient_id' in patient_metadata.columns and 'subject_id' not in patient_metadata.columns:
        patient_metadata = patient_metadata.rename(columns={'patient_id': 'subject_id'})
    patient_metadata['birth_date'] = pd.to_datetime(patient_metadata['birth_date'])
    patient_metadata.drop_duplicates(subset=['subject_id', 'icustay_id'], keep='first', inplace=True)
    logging.info(f"Loaded metadata for {len(patient_metadata)} ICU stays.")

    # Create labels mapping for traceability
    target_variables = ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso']
    labels_mapping = create_embedding_labels_mapping(args.meds_dir, target_variables)
    logging.info(f"Created labels mapping with {len(labels_mapping)} label records")
    
    # Initialize metadata tracking
    metadata_path = args.output_dir / "embedding_metadata.csv"
    existing_metadata = load_embedding_metadata(metadata_path)
    new_metadata_records = []
    
    # Create XGBoost-compatible experimental arm structure
    experimental_arm_dir = args.output_dir / args.experimental_arm
    splits_data = {}

    # Process each data split
    data_path = args.meds_dir / "data"
    splits = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

    for split in splits:
        logging.info(f"\n--- Processing split: '{split}' ---")
        # Use experimental arm structure for XGBoost compatibility
        split_output_dir = experimental_arm_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        events_df = pd.read_parquet(data_path / split / "data.parquet")
        # Ensure consistent use of subject_id
        if 'patient_id' in events_df.columns and 'subject_id' not in events_df.columns:
            events_df.rename(columns={'patient_id': 'subject_id'}, inplace=True)

        merged_df = pd.merge(events_df, patient_metadata[['subject_id', 'icustay_id', 'birth_date']], on='subject_id', how='inner')
        if merged_df.empty:
            logging.warning(f"No matching patient events for split '{split}'. Skipping."); continue

        icustay_groups = merged_df.groupby(['icustay_id', 'subject_id'])
        split_icustay_ids = set()
        
        batch = []
        batches_processed = 0
        
        for (icustay_id, subject_id), group_df in tqdm(icustay_groups, desc=f"Embedding '{split}' ICU stays"):
            # Check if embedding already exists in metadata (more robust than just file existence)
            existing_record = existing_metadata[
                (existing_metadata['icustay_id'] == icustay_id) & 
                (existing_metadata['split'] == split)
            ] if not existing_metadata.empty else pd.DataFrame()
            
            embedding_file_path = split_output_dir / f"{icustay_id}.npy"
            if not existing_record.empty and embedding_file_path.exists():
                logging.debug(f"Embedding for icustay {icustay_id} (subject {subject_id}) in split {split} already exists, skipping")
                split_icustay_ids.add(icustay_id)  # Still add to splits_data
                continue

            serialized_text = serialize_patient_events(subject_id, group_df)
            if serialized_text:
                batch.append((icustay_id, subject_id, serialized_text))
            
            if len(batch) >= args.batch_size:
                process_batch(batch, args.model_name, args.task_type, split_output_dir,
                            split, labels_mapping, new_metadata_records, args.output_dir,
                            args.experimental_arm, split_icustay_ids)
                batch = [] # Reset batch
                time.sleep(DEFAULT_RATE_LIMIT_DELAY_S)
                batches_processed += 1
                if args.debug and batches_processed >= 3:
                    logging.info("Debug limit reached for this split.")
                    break
        
        # Process the final, potentially smaller, batch
        if batch and not (args.debug and batches_processed >= 3):
            process_batch(batch, args.model_name, args.task_type, split_output_dir,
                        split, labels_mapping, new_metadata_records, args.output_dir,
                        args.experimental_arm, split_icustay_ids)
        
        # Store icustay IDs for this split
        splits_data[split] = split_icustay_ids
        logging.info(f"Processed {len(split_icustay_ids)} ICU stays for split '{split}'")

    # Save updated metadata
    if new_metadata_records:
        new_metadata_df = pd.DataFrame(new_metadata_records)
        combined_metadata = pd.concat([existing_metadata, new_metadata_df], ignore_index=True)
        save_embedding_metadata(combined_metadata, metadata_path)
        logging.info(f"Saved metadata with {len(new_metadata_records)} new records to {metadata_path}")

    # Create XGBoost-compatible label files
    if splits_data:
        logging.info("Creating XGBoost-compatible label files...")
        labels_dir = create_xgboost_label_files(args.meds_dir, args.output_dir, target_variables, splits_data, patient_metadata)
        logging.info(f"XGBoost label files created in: {labels_dir}")

    logging.info("\n--- Model 4B embedding generation complete. ---")
    logging.info(f"Embeddings saved in XGBoost-compatible format: {experimental_arm_dir}")
    logging.info(f"XGBoost-compatible label files available in: {args.output_dir / 'labels'}")
    logging.info("Ready for use with xgboost_embedding_analysis.py")

if __name__ == '__main__':
    main()