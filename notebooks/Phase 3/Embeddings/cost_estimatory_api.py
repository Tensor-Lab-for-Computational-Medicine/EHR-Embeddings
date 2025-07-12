# cost_estimator_api_sample.py
"""
Provides a FAST and ACCURATE cost estimation by sampling a subset of files,
using the API to get a precise token count for the sample, and then
extrapolating the total cost.
"""
import os
import random
import logging
import time
import getpass
import google.generativeai as genai
from tqdm import tqdm

# --- Configuration ---

# The directory containing the serialized text files from Phase 3.
DATA_DIR = 'notebooks/Phase 3/phase_3_serialized_data'

# The model used for counting tokens. This should match your embedding model.
MODEL_NAME = 'models/gemini-embedding-exp-03-07'

# --- Estimation Settings ---
# Number of random files to sample to calculate the average size.
# Increase this for a more accurate estimate (e.g., to 1000).
SAMPLE_SIZE = 100
SEED = 42 # For reproducible random sampling

# --- API Settings ---
# Number of documents to send in a single count_tokens API call.
BATCH_SIZE = 100
# Delay between batches to respect API rate limits.
RATE_LIMIT_DELAY = 1.0

# --- Official Gemini Embedding Pricing ---
PRICE_PER_1000_TOKENS = 0.00012  # $0.00012 per 1,000 tokens (batch rate)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)

def setup_api_key():
    """
    Securely prompts for the user's API key and configures the genai library.
    """
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            api_key = getpass.getpass('Please enter your Google AI Studio or Vertex AI API key: ')
        genai.configure(api_key=api_key)
        logging.info("Successfully configured API key.")
    except Exception as e:
        logging.error(f"Failed to configure API key: {e}")
        exit(1)


def create_batches(data_list, batch_size):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

def fast_api_estimate_cost(directory: str):
    """
    Estimates embedding cost by using the API to count tokens on a sample.
    """
    setup_api_key()
    logging.info(f"Starting fast estimation by sampling from: {directory}")
    
    all_files = [os.path.join(root, filename) for root, _, files in os.walk(directory) for filename in files if filename.endswith('.txt')]

    if not all_files:
        logging.error(f"No .txt files found in '{directory}'. Please check the DATA_DIR path.")
        return
        
    total_files = len(all_files)
    logging.info(f"Found a total of {total_files:,} files.")

    random.seed(SEED)
    actual_sample_size = min(SAMPLE_SIZE, total_files)
    sampled_files = random.sample(all_files, actual_sample_size)
    
    logging.info(f"Counting exact tokens for {actual_sample_size:,} random files using the API...")

    total_sampled_tokens = 0
    file_batches = list(create_batches(sampled_files, BATCH_SIZE))

    # FIX: Instantiate the model object first.
    model = genai.GenerativeModel(MODEL_NAME)

    for batch_of_files in tqdm(file_batches, desc="Analyzing sample batches"):
        try:
            batch_content = [open(filepath, 'r', encoding='utf-8').read() for filepath in batch_of_files]
            # FIX: Call count_tokens on the model instance and access the attribute.
            response = model.count_tokens(contents=batch_content)
            total_sampled_tokens += response.total_tokens
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            logging.warning(f"Could not process a batch: {e}")
            continue
    
    if actual_sample_size == 0:
        average_tokens_per_file = 0
    else:
        average_tokens_per_file = total_sampled_tokens / actual_sample_size
        
    estimated_total_tokens = average_tokens_per_file * total_files
    
    total_cost = (estimated_total_tokens / 1000) * PRICE_PER_1000_TOKENS
    
    # --- Display Results ---
    print("\n" + "="*60)
    print("      Vertex AI Gemini Embedding Cost Estimation (API Sample)")
    print("="*60)
    print(f"Model Assumed:           {MODEL_NAME}")
    print(f"Pricing Model:           Batch Rate @ ${PRICE_PER_1000_TOKENS} per 1,000 tokens")
    print(f"Token Count Method:      Exact API count on a random sample")
    print("-"*60)
    print(f"Total Files in Dataset:  {total_files:,}")
    print(f"Files Sampled:           {actual_sample_size:,}")
    print(f"Average Tokens per File: {average_tokens_per_file:,.2f}")
    print(f"Estimated Total Tokens:  {int(estimated_total_tokens):,}")
    print(f"Estimated Total Cost:    ${total_cost:.2f}")
    print("="*60)

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        logging.error(f"The specified directory does not exist: {DATA_DIR}")
        logging.error("Please update the DATA_DIR variable in this script.")
    else:
        fast_api_estimate_cost(DATA_DIR)
