# run_batch_embeddings.py (Final Version using google-genai library)
"""
A simplified, all-in-one script to generate text embeddings using the
Vertex AI Batch Prediction API.

This script is restartable and self-cleaning. It checks for existing .npy
output files, only processes the missing files, and cleans up temporary
input files from Google Cloud Storage after submitting a job.

This version is updated to use the 'google-genai' library, which is the
latest and most direct way to interact with the Gemini batch API.
"""
import os
import json
import logging
import time
import numpy as np

# Third-party libraries
from tqdm import tqdm
from google.cloud import storage
from google.api_core import exceptions
# --- Use the newer google-genai library ---
from google import genai
from google.genai.types import CreateBatchJobConfig


# --- User Configuration ---
# TODO: IMPORTANT: Replace this with the name of your GCS bucket.
GCS_BUCKET_NAME = "embeddings-bucket-2025"
# --------------------------

# Import settings from your config file
try:
    from config_vertex import (
        SERIALIZED_DATA_DIR,
        BASE_OUTPUT_DIR,
        PROJECT_ID,
        LOCATION,
        MODEL_NAME  # Should be 'text-embedding-004' or a valid model
    )
except ImportError:
    print("Error: Could not import 'config_vertex.py'. Make sure it's in the same directory.")
    exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

def get_files_to_process() -> list:
    """
    Compares source .txt files with existing .npy files and returns a list
    of source files that still need to be processed.
    """
    logging.info("Scanning for existing embeddings to avoid rework...")

    model_name_sanitized = MODEL_NAME.replace('/', '_')
    local_output_dir = os.path.join(BASE_OUTPUT_DIR, f"embeddings_{model_name_sanitized}")

    if not os.path.exists(local_output_dir):
        logging.info("Output directory does not exist. All files will be processed.")
        os.makedirs(local_output_dir, exist_ok=True)

    all_source_files = {
        os.path.relpath(os.path.join(root, filename), SERIALIZED_DATA_DIR)
        for root, _, files in os.walk(SERIALIZED_DATA_DIR)
        for filename in files if filename.endswith('.txt')
    }

    existing_output_files = {
        os.path.relpath(os.path.join(root, filename), local_output_dir).replace('.npy', '.txt')
        for root, _, files in os.walk(local_output_dir)
        for filename in files if filename.endswith('.npy')
    }

    missing_files = sorted(list(all_source_files - existing_output_files))

    return [os.path.join(SERIALIZED_DATA_DIR, f) for f in missing_files]


def prepare_and_upload_data(files_to_process: list, bucket_name: str) -> str:
    """
    Scans specified files, creates a JSONL file, and uploads it to GCS.
    """
    logging.info(f"Step 1: Preparing data for {len(files_to_process)} missing files...")

    jsonl_records = []
    for filepath in tqdm(files_to_process, desc="  -> Formatting data"):
        relative_path = os.path.relpath(filepath, SERIALIZED_DATA_DIR)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        record = {"content": content, "source_file_path": relative_path}
        jsonl_records.append(json.dumps(record))

    jsonl_string = "\n".join(jsonl_records)
    gcs_input_filename = f"batch-input/{MODEL_NAME.replace('/', '_')}-{int(time.time())}.jsonl"

    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(gcs_input_filename)

        logging.info(f"Uploading prepared data to gs://{bucket_name}/{gcs_input_filename}")
        blob.upload_from_string(jsonl_string, content_type="application/jsonl")

        return f"gs://{bucket_name}/{gcs_input_filename}"
    except exceptions.NotFound:
        logging.error(f"GCS Bucket '{bucket_name}' not found. Please create it first.")
        raise


def delete_gcs_blob(bucket_name: str, blob_name: str):
    """Deletes a blob from the specified GCS bucket."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if blob.exists():
            logging.info(f"Cleaning up temporary input file: gs://{bucket_name}/{blob_name}")
            blob.delete()
    except Exception as e:
        logging.warning(f"Could not delete GCS file gs://{bucket_name}/{blob_name}. "
                        f"You may want to delete it manually. Error: {e}")


def submit_and_wait_for_job(gcs_input_uri: str, bucket_name: str, client: genai.Client) -> str:
    """
    Submits the batch job using the google-genai client and waits for completion.
    """
    logging.info("Step 2: Submitting batch prediction job to Vertex AI...")

    gcs_output_prefix = f"gs://{bucket_name}/batch-output/job-{int(time.time())}/"

    # Create the batch job using the newer, simpler client
    batch_job = client.batches.create(
        model=f"models/{MODEL_NAME}",  # The genai client expects the 'models/' prefix
        src=gcs_input_uri,
        config=CreateBatchJobConfig(dest=gcs_output_prefix),
    )
    logging.info(f"Job '{batch_job.name}' submitted.")

    # Clean up the temporary input file from GCS
    input_blob_name = gcs_input_uri.replace(f"gs://{bucket_name}/", "")
    delete_gcs_blob(bucket_name, input_blob_name)

    logging.info("Waiting for job to complete... (This may take a long time)")

    # Poll the job status until it is no longer running
    while batch_job.state == genai.JobState.RUNNING:
        time.sleep(10)  # Wait 10 seconds between checks
        batch_job = client.batches.get(name=batch_job.name)

    # Check the final status
    if batch_job.state == genai.JobState.SUCCEEDED:
        logging.info("Batch job succeeded.")
        return batch_job.dest.gcs_uri
    else:
        raise RuntimeError(f"Batch job failed with state: {batch_job.state} and error: {batch_job.error}")


def download_and_process_results(gcs_output_prefix: str):
    """
    Downloads prediction results from GCS and saves them as .npy files locally.
    """
    logging.info("Step 3: Downloading and processing results...")

    model_name_sanitized = MODEL_NAME.replace('/', '_')
    local_output_dir = os.path.join(BASE_OUTPUT_DIR, f"embeddings_{model_name_sanitized}")

    uri_parts = gcs_output_prefix.replace("gs://", "").split("/", 1)
    bucket_name = uri_parts[0]
    prefix = uri_parts[1]

    storage_client = storage.Client(project=PROJECT_ID)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    result_blobs = [b for b in blobs if "predictions.jsonl" in b.name]
    if not result_blobs:
        raise FileNotFoundError(f"No result files found in {gcs_output_prefix}")

    for blob in tqdm(result_blobs, desc="  -> Processing result files"):
        jsonl_content = blob.download_as_text()
        for line in jsonl_content.strip().split('\n'):
            if not line:
                continue

            prediction = json.loads(line)
            source_file_path = prediction['request']['source_file_path']
            # The output structure is slightly different with this client
            embedding_vector = prediction['response']['candidates'][0]['embedding']['values']

            output_filepath = os.path.join(
                local_output_dir,
                os.path.splitext(source_file_path)[0] + '.npy'
            )

            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            np.save(output_filepath, np.array(embedding_vector))


def main():
    """Main execution function using the google-genai client."""
    if GCS_BUCKET_NAME == "your-gcs-bucket-name-here":
        logging.error("Please update the 'GCS_BUCKET_NAME' variable in this script before running.")
        return

    try:
        # Step 0: Check for work to do
        files_to_process = get_files_to_process()
        if not files_to_process:
            logging.info("All files have already been processed. Nothing to do.")
            return
        logging.info(f"Found {len(files_to_process)} files that need to be processed.")

        # Initialize the new, specialized client
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

        # Step 1: Prepare and upload data
        gcs_input_uri = prepare_and_upload_data(files_to_process, GCS_BUCKET_NAME)

        # Step 2: Submit job and wait
        gcs_output_prefix = submit_and_wait_for_job(gcs_input_uri, GCS_BUCKET_NAME, client)

        # Step 3: Process results
        download_and_process_results(gcs_output_prefix)
        logging.info("\n--- Success! All steps completed. ---")

    except Exception as e:
        logging.error(f"\n--- An error occurred during the process ---", exc_info=True)


if __name__ == "__main__":
    main()