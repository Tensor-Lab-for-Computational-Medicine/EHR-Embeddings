# generate_embeddings.py
"""
Main script to generate embeddings in parallel for all serialized text representations.

This script takes a worker ID as a command-line argument to divide the total
list of files among a set number of workers. Each worker processes its own
unique subset of files.
"""
import os
import time
import logging
import getpass
import argparse
import numpy as np
import google.generativeai as genai
from tqdm import tqdm
from config_embedding import (
    SERIALIZED_DATA_DIR,
    EMBEDDING_OUTPUT_DIR,
    MODEL_NAME,
    TASK_TYPE,
    DRY_RUN,
    RATE_LIMIT_DELAY
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'
)

def setup_api_key():
    """Securely prompts for and configures the Google AI API key."""
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            api_key = getpass.getpass('Please enter your Google AI Studio API key: ')
        genai.configure(api_key=api_key)
        logging.info("Successfully configured Google AI API key.")
    except Exception as e:
        logging.error(f"Failed to configure API key: {e}")
        exit(1)

def find_files_to_process(worker_id: int, total_workers: int):
    """
    Finds all .txt files and returns a unique subset for the specified worker.
    """
    all_files = []
    for root, _, files in os.walk(SERIALIZED_DATA_DIR):
        for filename in files:
            if filename.endswith('.txt'):
                all_files.append(os.path.join(root, filename))
    
    # Sort the list to ensure consistent assignment across all machines
    all_files.sort()

    # Assign files to this worker using the modulo operator
    assigned_files = [
        filepath for i, filepath in enumerate(all_files) 
        if i % total_workers == worker_id
    ]
    
    if DRY_RUN:
        if not assigned_files:
            logging.warning("This worker has no files assigned. This is normal if the total file count is less than the worker count.")
            return []
        logging.warning(f"DRY RUN ENABLED. Worker {worker_id} will process only its first assigned file.")
        return [assigned_files[0]] # Return only the first file for this worker
        
    return assigned_files

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate embeddings in parallel across multiple workers.")
    parser.add_argument('--worker-id', type=int, required=True, help='The ID for this worker (e.g., 0, 1, or 2).')
    parser.add_argument('--total-workers', type=int, default=3, help='The total number of workers splitting the job.')
    args = parser.parse_args()

    # Make the worker_id available for logging
    extra_dict = {'worker_id': args.worker_id}
    logger = logging.getLogger()
    handler = logger.handlers[0]
    formatter = logging.Formatter('%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    
    logging.info(f"--- Starting Phase IV: Embedding Generation (Worker {args.worker_id}/{args.total_workers-1}) ---", extra=extra_dict)
    
    setup_api_key()
    
    files_to_process = find_files_to_process(args.worker_id, args.total_workers)
    if not files_to_process:
        logging.info("No files assigned to this worker. Exiting.", extra=extra_dict)
        return

    logging.info(f"Found {len(files_to_process)} text files assigned to this worker.", extra=extra_dict)

    for input_filepath in tqdm(files_to_process, desc=f"Worker {args.worker_id} Progress"):
        try:
            relative_path = os.path.relpath(input_filepath, SERIALIZED_DATA_DIR)
            output_filepath = os.path.join(EMBEDDING_OUTPUT_DIR, relative_path)
            output_filepath = os.path.splitext(output_filepath)[0] + '.npy'
            
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

            with open(input_filepath, 'r', encoding='utf-8') as f:
                text_content = f.read()

            result = genai.embed_content(
                model=MODEL_NAME,
                content=text_content,
                task_type=TASK_TYPE
            )
            
            embedding_vector = np.array(result['embedding'])
            np.save(output_filepath, embedding_vector)
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logging.error(f"Failed to process {input_filepath}: {e}", extra=extra_dict)
            continue
            
    if DRY_RUN:
        logging.info("--- Dry run complete. ---", extra=extra_dict)
    else:
        logging.info(f"--- Worker {args.worker_id} has completed its assigned files. ---", extra=extra_dict)

if __name__ == '__main__':
    main()
