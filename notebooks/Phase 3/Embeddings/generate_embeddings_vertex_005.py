# generate_embeddings_vertex_005.py
"""
Main script to generate embeddings in parallel using the Vertex AI SDK.
This script uses DYNAMIC BATCHING to create batches that respect the
API's token limits, making it robust to variable file sizes.
"""
import os
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from config_vertex_005 import (
    SERIALIZED_DATA_DIR,
    BASE_OUTPUT_DIR,
    PROJECT_ID,
    LOCATION,
    MODEL_NAME,
    TASK_TYPE,
    MAX_FILES_PER_BATCH,
    MAX_TOKENS_PER_BATCH,
    DRY_RUN,
    RATE_LIMIT_DELAY,
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

def estimate_tokens(text: str) -> int:
    """A simple estimation of token count."""
    return len(text) // 4

# --- NEW: Dynamic Batching Function ---
def create_dynamic_batches(file_paths: list, max_tokens: int, max_files: int):
    """Creates batches of files that respect token and file count limits."""
    batch = []
    current_tokens = 0
    for filepath in file_paths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            estimated_tokens = estimate_tokens(content)

            # A single file cannot exceed the token limit
            if estimated_tokens > max_tokens:
                logging.warning(f"Skipping file {filepath} as its estimated token count ({estimated_tokens}) exceeds the batch limit ({max_tokens}).")
                continue

            # If adding the next file exceeds limits, yield the current batch
            if batch and (current_tokens + estimated_tokens > max_tokens or len(batch) >= max_files):
                yield batch
                batch = []
                current_tokens = 0

            batch.append({'path': filepath, 'content': content})
            current_tokens += estimated_tokens

        except Exception as e:
            logging.error(f"Could not read or process file {filepath}: {e}")
            continue
    
    # Yield the last remaining batch if it's not empty
    if batch:
        yield batch

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
    
    # --- MODIFIED: Use the new dynamic batching function ---
    file_batches = list(create_dynamic_batches(files_to_process, MAX_TOKENS_PER_BATCH, MAX_FILES_PER_BATCH))

    if DRY_RUN:
        logging.warning("DRY RUN ENABLED. Processing only the first batch.", extra=extra_dict)
        file_batches = file_batches[:1]

    logging.info(f"--- Starting Embedding Generation (Worker {worker_id}/{total_workers - 1}) ---", extra=extra_dict)
    logging.info(f"Worker {worker_id} assigned {len(files_to_process)} files, creating {len(file_batches)} dynamic batches.", extra=extra_dict)

    effective_delay = RATE_LIMIT_DELAY * total_workers

    for batch_of_files in tqdm(file_batches, desc=f"Worker {worker_id} Batches"):
        # --- MODIFIED: Logic to handle the new batch format ---
        batch_input_paths = [item['path'] for item in batch_of_files]
        expected_output_paths = []
        for input_filepath in batch_input_paths:
            relative_path = os.path.relpath(input_filepath, SERIALIZED_DATA_DIR)
            output_filepath = os.path.join(EMBEDDING_OUTPUT_DIR, os.path.splitext(relative_path)[0] + '.npy')
            expected_output_paths.append(output_filepath)

        if all(os.path.exists(p) for p in expected_output_paths):
            continue

        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                instances = [
                    TextEmbeddingInput(task_type=TASK_TYPE, title=os.path.basename(item['path']), text=item['content'])
                    for item in batch_of_files
                ]
                
                embeddings = model.get_embeddings(instances)

                for i, embedding in enumerate(embeddings):
                    output_filepath = expected_output_paths[i]
                    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                    np.save(output_filepath, np.array(embedding.values))

                success = True
                time.sleep(effective_delay)

            except Exception as e:
                retries += 1
                first_file = batch_input_paths[0] if batch_input_paths else "N/A"
                logging.error(f"Attempt {retries}/{MAX_RETRIES} failed for batch starting with {first_file}. Error: {e}", extra=extra_dict)
                if retries < MAX_RETRIES:
                    logging.warning(f"Sleeping for {BACKOFF_SECONDS}s before retrying...", extra=extra_dict)
                    time.sleep(BACKOFF_SECONDS)
                else:
                    logging.error(f"All retries failed for this batch. Skipping permanently.", extra=extra_dict)
    
    logging.info(f"--- Worker {worker_id} has completed its assigned files. ---", extra=extra_dict)

if __name__ == '__main__':
    main()