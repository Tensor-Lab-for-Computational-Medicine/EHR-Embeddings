# generate_embeddings_vertex.py
"""
Main script to generate embeddings in parallel using the Vertex AI SDK.
This definitive version is robust, restartable, resilient, and efficient.
It automatically retries failed calls, skips completed files, and uses an
adaptive, token-aware rate limiter to maximize throughput.
"""
import os
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque

import vertexai
from vertexai.language_models import TextEmbeddingModel

from config_vertex import (
    SERIALIZED_DATA_DIR,
    BASE_OUTPUT_DIR,
    PROJECT_ID,
    LOCATION,
    MODEL_NAME,
    TOKEN_LIMIT_PER_MINUTE,
    DRY_RUN,
    TOTAL_WORKERS,
    MAX_RETRIES,
    BACKOFF_SECONDS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'
)

def setup_vertex_ai(worker_id: int):
    """Initializes the Vertex AI SDK."""
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        logging.info("Successfully initialized Vertex AI SDK.", extra={'worker_id': worker_id})
    except Exception as e:
        logging.error(f"Vertex AI initialization failed. Please run 'gcloud auth application-default login'. Error: {e}", extra={'worker_id': worker_id})
        exit(1)

def find_files_to_process(worker_id: int, total_workers: int):
    """Finds this worker's assigned subset of .txt files."""
    all_files = [os.path.join(root, filename) for root, _, files in os.walk(SERIALIZED_DATA_DIR) for filename in files if filename.endswith('.txt')]
    all_files.sort()
    return [filepath for i, filepath in enumerate(all_files) if i % total_workers == worker_id]

def estimate_tokens(text: str) -> int:
    """Provides a simple, fast estimation of token count."""
    return len(text) // 4

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate embeddings in parallel using Vertex AI.")
    parser.add_argument('--worker-id', type=int, required=True, help='The unique ID for this worker (e.g., 0, 1, or 2).')
    args = parser.parse_args()

    worker_id = args.worker_id
    total_workers = TOTAL_WORKERS
    extra_dict = {'worker_id': worker_id}
    
    # --- Setup logging, paths, and SDK ---
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s [Worker %(worker_id)s] [%(levelname)s] - %(message)s'))
    
    model_name_sanitized = MODEL_NAME.replace('/', '_')
    EMBEDDING_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"embeddings_{model_name_sanitized}")
    
    setup_vertex_ai(worker_id=worker_id)
    
    # --- Adaptive Rate Limiter Setup ---
    per_worker_token_limit = TOKEN_LIMIT_PER_MINUTE / total_workers
    request_history = deque()
    
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    files_to_process = find_files_to_process(worker_id, total_workers)
    
    if DRY_RUN:
        logging.warning("DRY RUN ENABLED. Processing only the first 5 files.", extra=extra_dict)
        files_to_process = files_to_process[:5]

    logging.info(f"--- Starting Embedding Generation (Worker {worker_id}/{total_workers - 1}) ---", extra=extra_dict)
    logging.info(f"Using model: {MODEL_NAME}", extra=extra_dict)
    logging.info(f"Embeddings will be saved to: {EMBEDDING_OUTPUT_DIR}", extra=extra_dict)
    logging.info(f"Worker {worker_id} assigned {len(files_to_process)} files to process.", extra=extra_dict)
    logging.info(f"Worker {worker_id} targeting a rate of ~{per_worker_token_limit:,.0f} tokens/minute.", extra=extra_dict)

    for input_filepath in tqdm(files_to_process, desc=f"Worker {worker_id} Files"):
        relative_path = os.path.relpath(input_filepath, SERIALIZED_DATA_DIR)
        output_filepath = os.path.join(EMBEDDING_OUTPUT_DIR, os.path.splitext(relative_path)[0] + '.npy')

        # --- RE-INTRODUCED: Restartability Check ---
        # This is the first thing we do in the loop.
        if os.path.exists(output_filepath):
            continue # Skip this file entirely, it's already done.
        
        try:
            with open(input_filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # --- Adaptive Rate Limiting Logic ---
            tokens_in_request = estimate_tokens(content)
            
            while request_history and request_history[0]['timestamp'] < time.time() - 60:
                request_history.popleft()

            current_tokens_in_window = sum(item['tokens'] for item in request_history)
            if current_tokens_in_window + tokens_in_request > per_worker_token_limit:
                time_to_wait = (request_history[0]['timestamp'] + 60) - time.time()
                if time_to_wait > 0:
                    logging.warning(f"Approaching token limit. Pausing for {time_to_wait:.2f}s.", extra=extra_dict)
                    time.sleep(time_to_wait)

        except Exception as e:
            logging.error(f"Error during pre-flight check for {input_filepath}: {e}", extra=extra_dict)
            continue
        
        # --- Automatic Retry Logic ---
        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                embeddings = model.get_embeddings([content])
                vector = embeddings[0].values
                
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                np.save(output_filepath, np.array(vector))

                success = True
                request_history.append({'timestamp': time.time(), 'tokens': tokens_in_request})

            except Exception as e:
                retries += 1
                logging.error(f"Attempt {retries}/{MAX_RETRIES} failed for {input_filepath}. Error: {e}", extra=extra_dict)
                if retries < MAX_RETRIES:
                    logging.warning(f"Sleeping for {BACKOFF_SECONDS}s before retrying...", extra=extra_dict)
                    time.sleep(BACKOFF_SECONDS)
                else:
                    logging.error(f"All {MAX_RETRIES} retries failed for {input_filepath}. Skipping permanently.", extra=extra_dict)
            
    logging.info(f"--- Worker {worker_id} has completed its assigned files. ---", extra=extra_dict)

if __name__ == '__main__':
    main()