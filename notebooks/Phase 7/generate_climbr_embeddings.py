import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import femr.models.processor
    import femr.models.tokenizer
    import femr.models.transformer
except ImportError as e:
    sys.exit(f"--- DEPENDENCY ERROR: {e} ---\nPlease install the 'femr' library.")


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate CLIMBR embeddings from MEDS-FLAT Parquet files.")
    parser.add_argument("--meds-dir", type=Path, default=Path("./data/meds_cohort_split_filtered"), help="The root directory of the MEDS-FLAT dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("./notebooks/Phase 7"), help="The root directory to save the embeddings.") # Updated path
    parser.add_argument("--model-name", type=str, default="StanfordShahLab/clmbr-t-base", help="The Hugging Face name of the CLIMBR model to use.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"], help="NumPy dtype to save the embeddings.")
    return parser.parse_args()

def main():
    """Main function to load the model and generate embeddings."""
    args = get_args()

    # --- Device and Model Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Using device: {device} ---")

    try:
        print(f"--- Loading model: {args.model_name} ---")
        tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(args.model_name)
        batch_processor = femr.models.processor.FEMRBatchProcessor(tokenizer)
        model = femr.models.transformer.FEMRModel.from_pretrained(args.model_name)
        model.to(device)
        model.eval()
        print("--- Model loaded successfully ---")
    except Exception as e:
        sys.exit(f"Fatal: Failed to load model '{args.model_name}'. Error: {e}")

    # --- Load Patient Metadata ---
    patient_metadata_path = args.meds_dir / "patients.parquet"
    if not patient_metadata_path.exists():
        sys.exit(f"Fatal: Patient metadata not found at '{patient_metadata_path}'. Please run the conversion script first.")
    patient_metadata = pd.read_parquet(patient_metadata_path)
    print("--- Patient metadata loaded successfully ---")
    
    # --- Data Processing ---
    model_name_safe = args.model_name.replace('/', '_')
    output_dir_root = args.output_dir / f"embeddings_{model_name_safe}"
    data_path = args.meds_dir / "data"
    if not data_path.exists():
        sys.exit(f"Fatal: Input data path not found at '{data_path}'. Please run the conversion script first.")
        
    splits = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

    for i, split in enumerate(splits):
        print(f"\n--- Generating embeddings for split {i + 1} of {len(splits)} ('{split}') ---")
        split_data_path = data_path / split / "data.parquet"
        if not split_data_path.exists():
            print(f"  --> WARNING: Parquet file not found at '{split_data_path}'. Skipping split.")
            continue

        split_output_dir = output_dir_root / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        patient_events_df = pd.read_parquet(split_data_path)

        # FIX: Perform a robust INNER MERGE. This is the key fix.
        # It guarantees that we only proceed with patients who exist in both the event data
        # and the metadata file, completely eliminating the "missing birthdate" error.
        merged_df = pd.merge(patient_events_df, patient_metadata, on='patient_id', how='inner')
        merged_df.dropna(subset=['birth_date'], inplace=True) # Final safety check

        if merged_df.empty:
            print(f"  --> WARNING: No patients with valid birth dates found after merging for split '{split}'. Skipping.")
            continue
        
        patient_groups = merged_df.groupby('patient_id')
        for patient_id, group in tqdm(patient_groups, desc=f"Embedding '{split}' patients"):
            output_path = split_output_dir / f"{patient_id}.npy"
            if output_path.exists():
                continue

            # Convert flat MEDS to nested MEDS for CLIMBR
            events = []
            for time, measurements in group.groupby('time'):
                event_measurements = [{'code': row['code'], 'numeric_value': row['numeric_value']} if pd.notna(row['numeric_value']) else {'code': row['code']} for _, row in measurements.iterrows()]
                events.append({'time': time.to_pydatetime(), 'measurements': event_measurements})
            
            # The 'birth_date' is guaranteed to be present and valid due to the inner merge.
            example_patient = {
                'patient_id': int(patient_id),
                'events': events,
                'birth_date': group['birth_date'].iloc[0].to_pydatetime()
            }
            
            try:
                raw_batch = batch_processor.convert_patient(example_patient, tensor_type="pt")
                batch = {k: v.to(device) for k, v in batch_processor.collate([raw_batch]).items()}
                
                with torch.no_grad():
                    _, result = model(**batch)
                    
                embedding = result['representations'][0, -1, :].cpu().numpy().astype(args.dtype)
                np.save(output_path, embedding)
            except Exception as e:
                # This error should now only be triggered by model-specific issues, not data ones.
                print(f"  --> ERROR: Could not process patient {patient_id}. Skipping. Error: {e}")

    print("\n--- Embedding generation complete. ---")

if __name__ == '__main__':
    main()