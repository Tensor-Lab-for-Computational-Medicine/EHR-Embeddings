import argparse
import datetime
import os
import sys
import time
import getpass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

# --- Configuration ---
# These can be adjusted as needed
MODEL_NAME = "models/text-embedding-004"
TASK_TYPE = "RETRIEVAL_DOCUMENT"
BATCH_SIZE = 1000  # Number of patients to process in one API call
RATE_LIMIT_DELAY = 1  # Base delay in seconds between batches
TOTAL_WORKERS = 1 # Set to > 1 for multi-machine parallel processing

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'
)

def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Google AI embeddings from MEDS-FLAT Parquet files.")
    parser.add_argument(
        "--meds-dir",
        type=Path,
        default=Path("./data/meds_cohort_split_filtered"),
        help="The root directory of the MEDS-FLAT dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/meds_cohort_split_filtered"),
        help="The root directory to save the embeddings."
    )
    parser.add_argument(
        '--worker-id',
        type=int,
        required=True,
        help='The unique ID for this worker (e.g., 0, 1, or 2).'
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="NumPy dtype to save the embeddings."
    )
    return parser.parse_args()

def setup_api_key(worker_id: int):
    """Securely prompts for and configures the Google AI API key."""
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            api_key = getpass.getpass('Please enter your Google AI Studio API key: ')
        genai.configure(api_key=api_key)
        logging.info("Successfully configured Google AI API key.", extra={'worker_id': worker_id})
    except Exception as e:
        logging.error(f"Failed to configure API key: {e}", extra={'worker_id': worker_id})
        sys.exit(1)

def main():
    """Main function to load the model and generate embeddings."""
    args = get_args()
    worker_id = args.worker_id
    extra_dict = {'worker_id': worker_id}

    if worker_id >= TOTAL_WORKERS:
        logging.error(f"Worker ID {worker_id} is invalid. It must be less than TOTAL_WORKERS ({TOTAL_WORKERS}).", extra=extra_dict)
        sys.exit(1)

    # --- Setup ---
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'))

    setup_api_key(worker_id)

    model_name_safe = MODEL_NAME.replace('/', '_')
    output_dir_root = args.output_dir / f"embeddings_{model_name_safe}"
    
    logging.info(f"--- Starting Embedding Generation (Worker {worker_id}/{TOTAL_WORKERS - 1}) ---", extra=extra_dict)
    logging.info(f"Using model: {MODEL_NAME}", extra=extra_dict)
    logging.info(f"Embeddings will be saved to: {output_dir_root}", extra=extra_dict)

    # --- Data Processing ---
    data_path = args.meds_dir / "data"
    if not data_path.exists():
        sys.exit(f"Fatal: Input data path not found at '{data_path}'. Please run the conversion script first.")
        
    splits = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

    for split in splits:
        logging.info(f"\n--- Processing '{split}' split ---", extra=extra_dict)
        split_data_path = data_path / split / "data.parquet"
        if not split_data_path.exists():
            logging.warning(f"Parquet file not found at '{split_data_path}'. Skipping split.", extra=extra_dict)
            continue

        split_output_dir = output_dir_root / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        patient_events_df = pd.read_parquet(split_data_path)
        
        # --- Assign Patients to Worker ---
        all_patient_ids = sorted(patient_events_df['patient_id'].unique())
        worker_patient_ids = [pid for i, pid in enumerate(all_patient_ids) if i % TOTAL_WORKERS == worker_id]
        
        if not worker_patient_ids:
            logging.info("No patients assigned to this worker for this split. Skipping.", extra=extra_dict)
            continue

        logging.info(f"Worker {worker_id} assigned {len(worker_patient_ids)} patients for '{split}' split.", extra=extra_dict)

        patient_batches = [worker_patient_ids[i:i + BATCH_SIZE] for i in range(0, len(worker_patient_ids), BATCH_SIZE)]

        for batch_of_pids in tqdm(patient_batches, desc=f"Embedding '{split}' batches"):
            
            # --- Check if batch is already processed ---
            output_paths = [split_output_dir / f"{pid}.npy" for pid in batch_of_pids]
            if all(p.exists() for p in output_paths):
                continue

            # --- Prepare content for API ---
            batch_content = []
            for patient_id in batch_of_pids:
                # Serialize patient events into a single string
                patient_codes = patient_events_df[patient_events_df['patient_id'] == patient_id]['code']
                serialized_text = " ".join(patient_codes.tolist())
                batch_content.append(serialized_text)
            
            # --- Call API and Save ---
            try:
                result = genai.embed_content(model=MODEL_NAME, content=batch_content, task_type=TASK_TYPE)
                embeddings = result['embedding']

                for i, embedding_vector in enumerate(embeddings):
                    np.save(output_paths[i], np.array(embedding_vector, dtype=args.dtype))
                
                # Respect rate limits
                time.sleep(RATE_LIMIT_DELAY * TOTAL_WORKERS)

            except Exception as e:
                logging.error(f"Failed to process batch. PIDs: {batch_of_pids}. Error: {e}", extra=extra_dict)
                if "429" in str(e) or "500" in str(e) or "503" in str(e):
                    logging.warning("API error hit. Sleeping for 15 seconds.", extra=extra_dict)
                    time.sleep(15)
                continue
            
    logging.info(f"--- Worker {worker_id} has completed its assigned files. ---", extra=extra_dict)

if __name__ == '__main__':
    main()
