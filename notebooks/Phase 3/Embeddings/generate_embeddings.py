# generate_embeddings.py
"""
Main script to generate embeddings in parallel.
This script is robust and restartable. It automatically detects and skips
any files that have already been processed, allowing it to resume if interrupted.
"""
import os
import time
import logging
import getpass
import argparse
import numpy as np
import google.generativeai as genai
from tqdm import tqdm
from config import (
    SERIALIZED_DATA_DIR,
    BASE_OUTPUT_DIR,
    MODEL_NAME,
    TASK_TYPE,
    DRY_RUN,
    BATCH_SIZE,
    RATE_LIMIT_DELAY,
    TOTAL_WORKERS
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'
)

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
        exit(1)

def find_files_to_process(worker_id: int, total_workers: int):
    """Finds all .txt files and returns a unique subset for the specified worker."""
    all_files = [os.path.join(root, filename) for root, _, files in os.walk(SERIALIZED_DATA_DIR) for filename in files if filename.endswith('.txt')]
    all_files.sort()
    return [filepath for i, filepath in enumerate(all_files) if i % total_workers == worker_id]

def create_batches(data_list, batch_size):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate embeddings in parallel based on config.py.")
    parser.add_argument('--worker-id', type=int, required=True, help='The unique ID for this worker (e.g., 0, 1, or 2).')
    args = parser.parse_args()

    worker_id = args.worker_id
    total_workers = TOTAL_WORKERS
    extra_dict = {'worker_id': worker_id}

    if worker_id >= total_workers:
        logging.error(f"Worker ID {worker_id} is invalid. It must be less than TOTAL_WORKERS ({total_workers}) set in config.py.", extra=extra_dict)
        exit(1)

    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'))

    model_name_sanitized = MODEL_NAME.replace('/', '_')
    EMBEDDING_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"embeddings_{model_name_sanitized}")
    
    logging.info(f"--- Starting Embedding Generation (Worker {worker_id}/{total_workers - 1}) ---", extra=extra_dict)
    logging.info(f"Using model: {MODEL_NAME}", extra=extra_dict)
    logging.info(f"Embeddings will be saved to: {EMBEDDING_OUTPUT_DIR}", extra=extra_dict)

    setup_api_key(worker_id=worker_id)
    
    files_to_process = find_files_to_process(worker_id, total_workers)
    if not files_to_process:
        logging.info("No files assigned to this worker. Exiting.", extra=extra_dict)
        return
    
    file_batches = list(create_batches(files_to_process, BATCH_SIZE))
    
    if DRY_RUN:
        logging.warning("DRY RUN ENABLED. Processing only the first batch.", extra=extra_dict)
        file_batches = file_batches[:1]

    logging.info(f"Worker {worker_id} assigned {len(files_to_process)} files, processing in {len(file_batches)} batches of up to {BATCH_SIZE} files each.", extra=extra_dict)

    effective_delay = RATE_LIMIT_DELAY * total_workers
    logging.info(f"Base delay is {RATE_LIMIT_DELAY}s. With {total_workers} workers, effective delay is {effective_delay:.2f}s.", extra=extra_dict)

    for batch_of_files in tqdm(file_batches, desc=f"Worker {worker_id} Batches"):
        
        # --- START: New Robustness Logic ---
        
        # 1. Determine the expected output paths for the current batch
        expected_output_paths = []
        for input_filepath in batch_of_files:
            relative_path = os.path.relpath(input_filepath, SERIALIZED_DATA_DIR)
            output_filepath = os.path.join(EMBEDDING_OUTPUT_DIR, os.path.splitext(relative_path)[0] + '.npy')
            expected_output_paths.append(output_filepath)

        # 2. Check if all files in this batch have already been processed
        if all(os.path.exists(p) for p in expected_output_paths):
            continue # Skip this entire batch, as it's already done
            
        # --- END: New Robustness Logic ---

        try:
            batch_content = [open(f, 'r', encoding='utf-8').read() for f in batch_of_files]
            
            result = genai.embed_content(model=MODEL_NAME, content=batch_content, task_type=TASK_TYPE)
            embeddings = result['embedding']

            for i, embedding_vector in enumerate(embeddings):
                output_filepath = expected_output_paths[i] # Use path generated above
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                np.save(output_filepath, np.array(embedding_vector))
            
            time.sleep(effective_delay)

        except Exception as e:
            logging.error(f"Failed to process batch. Files will be retried on next run. Error: {e}", extra=extra_dict)
            if "429" in str(e) or "500" in str(e) or "503" in str(e):
                logging.warning("API error hit. Sleeping for 15 seconds.", extra=extra_dict)
                time.sleep(15)
            continue
            
    if DRY_RUN:
        logging.info("--- Dry run complete. ---", extra=extra_dict)
    else:
        logging.info(f"--- Worker {worker_id} has completed its assigned files. ---", extra=extra_dict)

if __name__ == '__main__':
    main()