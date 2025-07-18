# config_vertex.py
"""
Configuration settings for Vertex AI Embedding Generation.
"""

# --- Core Paths ---
# The input directory containing the serialized text files.
SERIALIZED_DATA_DIR = 'notebooks/Phase 3/phase_3_serialized_data'

# The base output directory. The script will create a model-specific sub-folder.
BASE_OUTPUT_DIR = 'notebooks/Phase 4'


# --- Google Cloud Vertex AI Settings ---

# Your Google Cloud project ID.
PROJECT_ID = "expanded-aria-465718-d1"

# The location of your project (e.g., "us-central1").
LOCATION = "us-central1"

# The embedding model to use.
MODEL_NAME = 'google/gemini-embedding-001'


# --- NEW: Token-based Rate Limit ---
# The total number of tokens allowed per minute for the entire project.
TOKEN_LIMIT_PER_MINUTE = 400000


# --- Script Settings ---
DRY_RUN = False


# --- Parallel Processing Settings ---
TOTAL_WORKERS = 1


# --- Retry Settings ---
MAX_RETRIES = 10
BACKOFF_SECONDS = 10