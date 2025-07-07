# config_embedding_gemini_exp.py
"""
Configuration settings for Phase IV: Embedding Generation.

Adapted to use the experimental 'gemini-embedding-exp-03-07' model.
"""

# --- Core Paths ---

# The input directory containing the serialized text files from Phase 3.
# This should point to the output of your previous script.
SERIALIZED_DATA_DIR = 'notebooks/Phase 3/phase_3_serialized_data'

# The output directory where all generated embeddings will be saved.
# A NEW FOLDER is specified to keep these experimental embeddings separate.
EMBEDDING_OUTPUT_DIR = 'notebooks/Phase 4/phase_4_embeddings_gemini_exp'


# --- Google AI Studio API Settings ---

# The embedding model to use. Changed to the specified experimental model.
MODEL_NAME = 'models/gemini-embedding-exp-03-07'

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