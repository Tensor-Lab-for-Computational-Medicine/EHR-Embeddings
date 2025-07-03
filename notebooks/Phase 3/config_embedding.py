# config_embedding.py
"""
Configuration settings for Phase IV: Embedding Generation.
"""

# --- Core Paths ---

# The input directory containing the serialized text files from Phase 3.
# This should point to the output of your previous script.
SERIALIZED_DATA_DIR = 'notebooks/Phase 3/phase_3_serialized_data'

# The output directory where all generated embeddings will be saved.
# This script will create this directory and its subfolders.
EMBEDDING_OUTPUT_DIR = 'notebooks/Phase 4/phase_4_embeddings'


# --- Google AI Studio API Settings ---

# The embedding model to use. 'text-embedding-004' is the latest standard.
MODEL_NAME = 'models/text-embedding-004'

# The task type helps the model produce embeddings optimized for this use case.
# 'RETRIEVAL_DOCUMENT' is ideal for creating representations of documents
# that will be used in downstream models (classification, clustering, etc.).
TASK_TYPE = 'RETRIEVAL_DOCUMENT'


# --- Script Settings ---

# Set to True to run the script on only ONE file to test the API key and full process.
# Set to False to run on all files in the SERIALIZED_DATA_DIR.
DRY_RUN = True

# Delay in seconds between API calls to respect rate limits (e.g., 60 requests/minute).
# A value of 1.1 is a safe starting point.
RATE_LIMIT_DELAY = 1.1