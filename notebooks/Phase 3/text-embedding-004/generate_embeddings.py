# generate_embeddings.py
"""
Main script to generate embeddings in parallel using an efficient BATCHING strategy.
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
    BATCH_SIZE,
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
    parser = argparse.ArgumentParser(description="Generate embeddings in parallel across multiple workers.")
    parser.add_argument('--worker-id', type=int, required=True, help='The ID for this worker (e.g., 0, 1, or 2).')
    parser.add_argument('--total-workers', type=int, default=3, help='The total number of workers splitting the job.')
    args = parser.parse_args()

    extra_dict = {'worker_id': args.worker_id}
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'))
    
    logging.info(f"--- Starting Embedding Generation (Worker {args.worker_id}/{args.total_workers-1}) ---", extra=extra_dict)
    
    setup_api_key()
    
    files_to_process = find_files_to_process(args.worker_id, args.total_workers)
    if not files_to_process:
        logging.info("No files assigned to this worker. Exiting.", extra=extra_dict)
        return

    file_batches = list(create_batches(files_to_process, BATCH_SIZE))
    
    if DRY_RUN:
        logging.warning("DRY RUN ENABLED. Processing only the first batch.", extra=extra_dict)
        file_batches = file_batches[:1]

    logging.info(f"Worker {args.worker_id} assigned {len(files_to_process)} files, processing in {len(file_batches)} batches of up to {BATCH_SIZE} files each.", extra=extra_dict)

    for batch_of_files in tqdm(file_batches, desc=f"Worker {args.worker_id} Batches"):
        try:
            batch_content = []
            batch_output_paths = []

            for input_filepath in batch_of_files:
                with open(input_filepath, 'r', encoding='utf-8') as f:
                    batch_content.append(f.read())
                
                relative_path = os.path.relpath(input_filepath, SERIALIZED_DATA_DIR)
                output_filepath = os.path.join(EMBEDDING_OUTPUT_DIR, relative_path)
                output_filepath = os.path.splitext(output_filepath)[0] + '.npy'
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                batch_output_paths.append(output_filepath)

            # Generate embeddings for the entire batch in one API call
            result = genai.embed_content(model=MODEL_NAME, content=batch_content, task_type=TASK_TYPE)
            embeddings = result['embedding']

            # Save the embeddings from the batch
            for i, embedding_vector in enumerate(embeddings):
                np.save(batch_output_paths[i], np.array(embedding_vector))
            
            # Sleep once per batch to respect rate limits
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logging.error(f"Failed to process a batch. Error: {e}", extra=extra_dict)
            logging.error("Skipping this batch and continuing...", extra=extra_dict)
            continue
            
    if DRY_RUN:
        logging.info("--- Dry run complete. ---", extra=extra_dict)
    else:
        logging.info(f"--- Worker {args.worker_id} has completed its assigned files. ---", extra=extra_dict)

if __name__ == '__main__':
    main()