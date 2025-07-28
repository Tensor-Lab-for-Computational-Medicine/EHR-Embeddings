"""
Configuration settings for Phase IV: Embedding Generation.

Adapted to use the experimental 'gemini-embedding-exp-03-07' model.
Found the following models suitable for embedding:
  - models/embedding-001
  - models/text-embedding-004
  - models/gemini-embedding-exp-03-07
  - models/gemini-embedding-exp
"""
# config.py


# --- Core Paths ---

# The input directory containing the serialized text files.
SERIALIZED_DATA_DIR = 'notebooks/Phase 3/phase_3_serialized_data'

# The base output directory. The script will create a model-specific
# sub-folder inside this directory (e.g., 'embeddings_models_text-embedding-004').
BASE_OUTPUT_DIR = 'notebooks/Phase 4/Classification_Embeddings'


# --- Google AI Studio API Settings ---

# The embedding model to use.
# Simply change this line to switch models.
# Example 1: 'models/text-embedding-004' (Latest Stable)
# Example 2: 'models/gemini-embedding-exp-03-07' (Experimental)
MODEL_NAME = 'models/text-embedding-004'

# The task type helps the model produce optimized embeddings.
# 'RETRIEVAL_DOCUMENT' is ideal for document representation.
TASK_TYPE = 'CLASSIFICATION'


# --- Script Settings ---

# The number of documents to send in a single API request.
# 100 is a safe and efficient value for current models.
BATCH_SIZE = 128

# Set to True to run the script on only ONE BATCH to test the full process.
# Set to False to run on all files.
DRY_RUN = False

# Base delay in seconds between API calls for a single worker.
# The script automatically multiplies this by the number of workers for safety.
RATE_LIMIT_DELAY = 0

# --- Parallel Processing Settings ---

# The total number of workers you plan to run in parallel.
# Ensure this matches the number of terminals you open to run the script.
TOTAL_WORKERS = 1