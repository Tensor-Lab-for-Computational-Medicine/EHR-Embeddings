# config_vertex_exp.py
"""
Configuration settings for Vertex AI Embedding Generation
using the experimental 'text-embedding-large-exp-03-07' model.
"""

# --- Core Paths ---
SERIALIZED_DATA_DIR = 'notebooks/Phase 3/phase_3_serialized_data'
BASE_OUTPUT_DIR = 'notebooks/Phase 4'


# --- Google Cloud Vertex AI Settings ---
PROJECT_ID = "expanded-aria-465718-d1"
LOCATION = "us-central1"
MODEL_NAME = 'text-embedding-large-exp-03-07'


# --- Batching and Rate Limit Settings ---
# NOTE: The following settings are not used by the new file-by-file script.
# MAX_FILES_PER_BATCH = 250
# MAX_TOKENS_PER_BATCH = 18000
# RATE_LIMIT_DELAY = 1.0


# --- Script Settings ---
DRY_RUN = False


# --- Parallel Processing Settings ---
TOTAL_WORKERS = 1


# --- Retry Settings ---
# These settings are now the primary way the script handles rate limits.
MAX_RETRIES = 5
BACKOFF_SECONDS = 15 # Time to wait after hitting a rate limit error.