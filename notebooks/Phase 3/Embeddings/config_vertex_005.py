# config_vertex_005.py
"""
Configuration settings for Vertex AI Embedding Generation
using the 'text-embedding-005' model.
"""

# --- Core Paths ---
SERIALIZED_DATA_DIR = 'notebooks/Phase 3/phase_3_serialized_data'
BASE_OUTPUT_DIR = 'notebooks/Phase 4'


# --- Google Cloud Vertex AI Settings ---
PROJECT_ID = "nth-wording-462614-s0"
LOCATION = "us-central1"
MODEL_NAME = 'text-embedding-005'
# The task type helps the model produce better embeddings for your use case.
# Common types: "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "CLASSIFICATION"
TASK_TYPE = "RETRIEVAL_DOCUMENT"


# --- Script Settings ---
# text-embedding-005 supports batching up to 250 documents.
# Using the max batch size is most efficient.
BATCH_SIZE = 250

DRY_RUN = False

# Base delay in seconds between API calls for a single worker.
# Appropriate for RPM-based quotas.
RATE_LIMIT_DELAY = 0


# --- Parallel Processing Settings ---
TOTAL_WORKERS = 1


# --- Retry Settings ---
MAX_RETRIES = 5
BACKOFF_SECONDS = 15

# --- NEW: Batching and Rate Limit Settings ---
# The API has two limits per request: max files and max tokens.
# We will respect both.
MAX_FILES_PER_BATCH = 100
MAX_TOKENS_PER_BATCH = 12000 # Set slightly below the 20,000 limit for a safety margin

