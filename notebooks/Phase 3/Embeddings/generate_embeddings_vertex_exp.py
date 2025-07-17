# generate_embeddings_vertex_exp.py
"""
Main script to generate embeddings in parallel using the Vertex AI SDK.
This script is adapted for single-request models like 'text-embedding-large-exp-03-07'
and uses a reactive retry mechanism to handle rate limits.
"""
import os
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import vertexai
from vertexai.language_models import TextEmbeddingModel

from config_vertex_exp import (
    SERIALIZED_DATA_DIR,
    BASE_OUTPUT_DIR,
    PROJECT_ID,
    LOCATION,
    MODEL_NAME,
    DRY_RUN,
    TOTAL_WORKERS,
    MAX_RETRIES,
    BACKOFF_SECONDS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s')

def setup_vertex_ai(worker_id: int):
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        logging.info("Successfully initialized Vertex AI SDK.", extra={'worker_id': worker_id})
    except Exception as e:
        logging.error(f"Vertex AI initialization failed. Error: {e}", extra={'worker_id': worker_id})
        exit(1)

def find_files_to_process(worker_id: int, total_workers: int):
    all_files = [os.path.join(root, filename) for root, _, files in os.walk(SERIALIZED_DATA_DIR) for filename in files if filename.endswith('.txt')]
    all_files.sort()
    return [filepath for i, filepath in enumerate(all_files) if i % total_workers == worker_id]

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings in parallel using Vertex AI.")
    parser.add_argument('--worker-id', type=int, required=True, help='The unique ID for this worker.')
    args = parser.parse_args()

    worker_id = args.worker_id
    total_workers = TOTAL_WORKERS
    extra_dict = {'worker_id': worker_id}
    
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'))
    
    model_name_sanitized = MODEL_NAME.replace('/', '_')
    EMBEDDING_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"embeddings_{model_name_sanitized}")
    
    setup_vertex_ai(worker_id=worker_id)
    
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    files_to_process = find_files_to_process(worker_id, total_workers)

    if DRY_RUN:
        logging.warning("DRY RUN ENABLED. Processing only the first 5 files.", extra=extra_dict)
        files_to_process = files_to_process[:5]

    logging.info(f"--- Starting Embedding Generation (Worker {worker_id}/{total_workers - 1}) ---", extra=extra_dict)
    logging.info(f"Worker {worker_id} assigned {len(files_to_process)} files to process.", extra=extra_dict)
    logging.info(f"Proactive rate limit delays are disabled. Using reactive retries on error.", extra=extra_dict)

    # --- MODIFIED: Loop processes one file at a time ---
    for input_filepath in tqdm(files_to_process, desc=f"Worker {worker_id} Files"):
        relative_path = os.path.relpath(input_filepath, SERIALIZED_DATA_DIR)
        output_filepath = os.path.join(EMBEDDING_OUTPUT_DIR, os.path.splitext(relative_path)[0] + '.npy')

        if os.path.exists(output_filepath):
            continue

        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                with open(input_filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # --- MODIFIED: API call with a single text string in a list ---
                embeddings = model.get_embeddings([content])
                vector = embeddings[0].values

                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                np.save(output_filepath, np.array(vector))

                success = True
                # --- MODIFIED: Proactive time.sleep() after success is removed ---

            except Exception as e:
                retries += 1
                logging.error(f"Attempt {retries}/{MAX_RETRIES} failed for file {input_filepath}. Error: {e}", extra=extra_dict)
                if retries < MAX_RETRIES:
                    logging.warning(f"Sleeping for {BACKOFF_SECONDS}s before retrying...", extra=extra_dict)
                    time.sleep(BACKOFF_SECONDS)
                else:
                    logging.error(f"All {MAX_RETRIES} retries failed for {input_filepath}. Skipping permanently.", extra=extra_dict)
    
    logging.info(f"--- Worker {worker_id} has completed its assigned files. ---", extra=extra_dict)

if __name__ == '__main__':
    main()