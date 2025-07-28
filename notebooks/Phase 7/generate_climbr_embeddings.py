"""
Generate CLIMBR embeddings from MEDS-FLAT data in XGBoost-compatible format.

This script:
1. Loads CLIMBR model and processes MEDS-FLAT patient event data
2. Generates embeddings for each ICU stay (icustay_id)
3. Saves embeddings in XGBoost-compatible directory structure
4. Creates label CSV files with icustay_id as index
5. Maintains full traceability metadata

Output structure:
- embeddings_StanfordShahLab_clmbr-t-base/
  - CLIMBR_P0/train|val|test/{icustay_id}.npy
  - labels/{task}_{split}_labels.csv
  - embedding_metadata.csv

Compatible with: xgboost_embedding_analysis.py
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
import logging
import json
import math
import traceback

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import femr.models.processor
    import femr.models.tokenizer
    import femr.models.transformer
    
    import meds # Import meds to use the official birth_code
except ImportError as e:
    sys.exit(f"--- DEPENDENCY ERROR: {e} ---\nPlease install the 'femr' and 'meds' libraries.")

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate CLIMBR embeddings from MEDS-FLAT Parquet files in XGBoost-compatible format.")
    parser.add_argument("--meds-dir", type=Path, default=Path("./data/meds_cohort_split_filtered"), help="The root directory of the MEDS-FLAT dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("./notebooks/Phase 7"), help="The root directory to save the embeddings.")
    parser.add_argument("--model-name", type=str, default="StanfordShahLab/clmbr-t-base", help="The Hugging Face name of the CLIMBR model to use.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"], help="NumPy dtype to save the embeddings.")
    parser.add_argument("--debug-patient-id", type=int, default=None, help="Process only a single subject ID for debugging purposes.")
    parser.add_argument("--dry-run", action='store_true', help="Run the script for only the first valid patient and then exit.")
    parser.add_argument("--experimental-arm", type=str, default="CLIMBR_P0", help="Experimental arm name for XGBoost compatibility.")
    return parser.parse_args()

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

def move_batch_to_device(batch, device):
    """
    Recursively moves all tensors in a nested dictionary to the specified device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    else:
        return batch

def main():
    """Main function to load the model and generate embeddings."""
    args = get_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    try:
        logging.info(f"Loading model: {args.model_name}")
        tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(args.model_name)
        batch_processor = femr.models.processor.FEMRBatchProcessor(tokenizer)
        model = femr.models.transformer.FEMRModel.from_pretrained(args.model_name)
        model.to(device)
        model.eval()
        logging.info("Model loaded successfully")
    except Exception as e:
        sys.exit(f"Fatal: Failed to load model '{args.model_name}'. Error: {e}")

    patient_metadata_path = args.meds_dir / "patients.parquet"
    if not patient_metadata_path.exists():
        sys.exit(f"Fatal: Patient metadata not found at '{patient_metadata_path}'")
        
    patient_metadata = pd.read_parquet(patient_metadata_path)
    if 'patient_id' in patient_metadata.columns and 'subject_id' not in patient_metadata.columns:
        patient_metadata = patient_metadata.rename(columns={'patient_id': 'subject_id'})
    # Keep both subject_id and icustay_id for XGBoost compatibility
    patient_metadata.drop_duplicates(subset=['subject_id', 'icustay_id'], keep='first', inplace=True)
    logging.info("Patient metadata loaded successfully.")
    
    # Create labels mapping for traceability
    target_variables = ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso']
    labels_mapping = create_embedding_labels_mapping(args.meds_dir, target_variables)
    logging.info(f"Created labels mapping with {len(labels_mapping)} label records")
    
    model_name_safe = args.model_name.replace('/', '_')
    output_dir_root = args.output_dir / f"embeddings_{model_name_safe}"
    
    # For XGBoost compatibility, use experimental arm structure
    experimental_arm_dir = output_dir_root / args.experimental_arm
    
    # Initialize or load embedding metadata
    metadata_path = output_dir_root / "embedding_metadata.csv"
    existing_metadata = load_embedding_metadata(metadata_path)
    new_metadata_records = []
    
    data_path = args.meds_dir / "data"
    splits = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    # Collect subject IDs by split for XGBoost label generation
    splits_data = {}
    
    if args.debug_patient_id:
        logging.warning(f"--- DEBUG MODE ON: Processing ONLY subject_id = {args.debug_patient_id} ---")
        logging.getLogger().setLevel(logging.DEBUG)

    for split in splits:
        logging.info(f"Generating embeddings for split '{split}'")
        split_data_path = data_path / split / "data.parquet"
        if not split_data_path.exists(): 
            logging.warning(f"Data file not found for split '{split}', skipping.")
            continue

        # Use experimental arm structure for XGBoost compatibility
        split_output_dir = experimental_arm_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        patient_events_df = pd.read_parquet(split_data_path)
        # Ensure we're using subject_id consistently
        if 'patient_id' in patient_events_df.columns and 'subject_id' not in patient_events_df.columns:
            patient_events_df.rename(columns={'patient_id': 'subject_id'}, inplace=True)
        
        if 'birth_date' not in patient_metadata.columns:
            sys.exit(f"Fatal: 'birth_date' column not found in '{patient_metadata_path}'. Please check `convert_to_meds.py`.")
            
        merged_df = pd.merge(patient_events_df, patient_metadata[['subject_id', 'icustay_id', 'birth_date']], on='subject_id', how='inner')
        if merged_df.empty: 
            logging.warning(f"No matching patient events for split '{split}'. Skipping.")
            continue

        if args.debug_patient_id:
            merged_df = merged_df[merged_df['subject_id'] == args.debug_patient_id]
            if merged_df.empty: 
                logging.error(f"Debug subject ID {args.debug_patient_id} not found in split '{split}'.")
                continue
        
        icustay_groups = merged_df.groupby(['icustay_id', 'subject_id'])
        
        # Collect icustay IDs for this split for XGBoost label generation
        split_icustay_ids = set()
        processed_count = 0
        
        for (icustay_id, subject_id), group in tqdm(icustay_groups, desc=f"Embedding '{split}' ICU stays"):
            # Output path (XGBoost compatible - using icustay_id as filename)
            output_path = split_output_dir / f"{icustay_id}.npy"
            
            # Check if embedding already exists in metadata (more robust than just file existence)
            existing_record = existing_metadata[
                (existing_metadata['icustay_id'] == icustay_id) & 
                (existing_metadata['split'] == split)
            ] if not existing_metadata.empty else pd.DataFrame()
            
            if not existing_record.empty and output_path.exists() and not (args.debug_patient_id or args.dry_run):
                logging.debug(f"Embedding for icustay {icustay_id} (subject {subject_id}) in split {split} already exists, skipping")
                split_icustay_ids.add(icustay_id)  # Still add to splits_data
                continue
            
            try:
                birth_date_dt = pd.to_datetime(group['birth_date'].iloc[0])
                if pd.isna(birth_date_dt):
                    logging.warning(f"Skipping icustay {icustay_id} (subject {subject_id}) due to missing birth date.")
                    continue
                
                birth_event = {
                    'time': birth_date_dt.to_pydatetime(),
                    'measurements': [{'code': meds.birth_code}]
                }
                
                clinical_events = []
                group['time'] = pd.to_datetime(group['time'], errors='coerce')
                group.dropna(subset=['time', 'code'], inplace=True)

                for timestamp, events_at_time in group.groupby('time'):
                    measurements = []
                    for _, row in events_at_time.iterrows():
                        code = str(row['code']).strip()
                        if pd.notna(row.get('numeric_value')) and math.isfinite(float(row['numeric_value'])):
                            measurements.append({'code': code, 'numeric_value': float(row['numeric_value'])})
                        else:
                            measurements.append({'code': code})
                    if measurements:
                        clinical_events.append({'time': timestamp.to_pydatetime(), 'measurements': measurements})
                
                all_events = [birth_event] + clinical_events
                all_events.sort(key=lambda x: x['time'])
                
                # DEBUG: Check events before FEMR processing
                if processed_count <= 5:
                    logging.info(f"DEBUG - Patient {icustay_id}: {len(all_events)} total events ({len(clinical_events)} clinical + 1 birth)")
                    if len(all_events) > 1:
                        logging.info(f"DEBUG - Patient {icustay_id}: first event time = {all_events[0]['time']}")
                        logging.info(f"DEBUG - Patient {icustay_id}: second event time = {all_events[1]['time']}")
                        logging.info(f"DEBUG - Patient {icustay_id}: second event measurements = {len(all_events[1]['measurements'])}")
                        logging.info(f"DEBUG - Patient {icustay_id}: first 3 codes = {[m['code'][:50] for m in all_events[1]['measurements'][:3]]}")
                
                example_patient = {
                    'patient_id': int(icustay_id),  # Using icustay_id for FEMR processing
                    'events': all_events,
                }
                
                raw_batch = batch_processor.convert_patient(example_patient, tensor_type="pt")
                
                if not raw_batch.get('transformer') or not raw_batch['transformer']['valid_tokens'].any():
                    logging.warning(f"ICU stay {icustay_id} (subject {subject_id}) resulted in zero valid tokens after processing. Skipping.")
                    continue
                

                
                collated_batch = batch_processor.collate([raw_batch])
                batch = move_batch_to_device(collated_batch, device)
                
                with torch.no_grad():
                    _, result = model(**batch)
                

                
                # Fix: Use mean pooling over valid tokens instead of last token
                representations = result['representations']
                if len(representations.shape) == 2:  # [seq_len, hidden_dim]
                    # Get valid token mask if available
                    if 'valid_tokens' in raw_batch.get('transformer', {}):
                        valid_mask = raw_batch['transformer']['valid_tokens']
                        if valid_mask.any():
                            # Apply mask and take mean of valid tokens
                            valid_representations = representations[valid_mask, :]
                            embedding = valid_representations.mean(dim=0).cpu().numpy().astype(args.dtype)
                        else:
                            # Fallback to mean of all tokens
                            embedding = representations.mean(dim=0).cpu().numpy().astype(args.dtype)
                    else:
                        # Fallback to mean pooling of all tokens
                        embedding = representations.mean(dim=0).cpu().numpy().astype(args.dtype)
                else:
                    # Handle unexpected shape
                    embedding = representations.flatten().cpu().numpy().astype(args.dtype)
                

                
                np.save(output_path, embedding)
                
                # Add to split data for XGBoost label generation
                split_icustay_ids.add(icustay_id)
                processed_count += 1
                
                # Create metadata record for traceability
                subject_labels = labels_mapping[labels_mapping['subject_id'] == subject_id]
                for _, label_row in subject_labels.iterrows():
                    new_metadata_records.append({
                        'subject_id': subject_id,
                        'icustay_id': icustay_id,
                        'split': split,
                        'embedding_file': str(output_path.relative_to(args.output_dir)),
                        'task': label_row['task'],
                        'prediction_time': label_row['prediction_time'],
                        'label_value': label_row['label_value'],
                        'embedding_shape': embedding.shape,
                        'dtype': str(embedding.dtype),
                        'generated_at': datetime.datetime.now().isoformat(),
                        'model_name': args.model_name,
                        'experimental_arm': args.experimental_arm
                    })
                
                # logging.info(f"Generated embedding for patient {patient_id} with shape {embedding.shape}")
                
                if args.dry_run and processed_count >= 3:
                    logging.info(f"--- Dry run: processed {processed_count} ICU stays from split '{split}'. Moving to next split. ---")
                    break  # Exit the ICU stay loop, but continue to next split
                
                if args.debug_patient_id:
                    logging.info(f"--- Successfully processed debug icustay {icustay_id} (subject {subject_id}). Exiting. ---")
                    # Save metadata before exiting
                    if new_metadata_records:
                        new_metadata_df = pd.DataFrame(new_metadata_records)
                        combined_metadata = pd.concat([existing_metadata, new_metadata_df], ignore_index=True)
                        save_embedding_metadata(combined_metadata, metadata_path)
                    sys.exit(0)

            except Exception as e:
                logging.error(f"FATAL ERROR during FEMR processing for icustay {icustay_id} (subject {subject_id}).")
                logging.error(f"The error was: {e}")
                logging.error(traceback.format_exc())
                if args.debug_patient_id:
                    sys.exit(1)
        
        # Store icustay IDs for this split
        splits_data[split] = split_icustay_ids
        logging.info(f"Processed {len(split_icustay_ids)} ICU stays for split '{split}'")
    
    if args.dry_run:
        total_processed = sum(len(ids) for ids in splits_data.values())
        logging.info(f"--- Dry run complete: processed {total_processed} total ICU stays across all splits ---")
    
    # Save updated metadata
    if new_metadata_records:
        new_metadata_df = pd.DataFrame(new_metadata_records)
        combined_metadata = pd.concat([existing_metadata, new_metadata_df], ignore_index=True)
        save_embedding_metadata(combined_metadata, metadata_path)
        logging.info(f"Saved metadata with {len(new_metadata_records)} new records to {metadata_path}")
    
    # Create XGBoost-compatible label files
    if splits_data:
        logging.info("Creating XGBoost-compatible label files...")
        labels_dir = create_xgboost_label_files(args.meds_dir, output_dir_root, target_variables, splits_data, patient_metadata)
        logging.info(f"XGBoost label files created in: {labels_dir}")
    
    dry_run_suffix = " (DRY RUN)" if args.dry_run else ""
    logging.info(f"--- CLIMBR embedding generation complete{dry_run_suffix}. ---")
    logging.info(f"Embeddings saved in XGBoost-compatible format: {experimental_arm_dir}")
    logging.info(f"XGBoost-compatible label files available in: {output_dir_root / 'labels'}") 
    logging.info("Ready for use with xgboost_embedding_analysis.py")

if __name__ == '__main__':
    main()