Instructions for Running the Vertex AI Embedding Script
This guide explains the one-time setup required to run the generate_embeddings_gemini.py script correctly. The script now uses your gcloud login (Application Default Credentials) and requires specific environment variables to target the Vertex AI backend.

Prerequisites
Google Cloud SDK: Ensure you have gcloud installed and authenticated. If you haven't already, run:

gcloud auth application-default login

Project Files: Make sure you have generate_embeddings_gemini.py and config_embedding_gemini.py.

Step 1: Set Environment Variables
Before running the script, you must set three environment variables in your terminal. This tells the Python library to use Vertex AI.

Replace your-gcp-project-id with your actual Google Cloud Project ID.

# Replace with your actual project ID
export GOOGLE_CLOUD_PROJECT=nth-wording-462614-s0

# The location for Vertex AI embeddings is typically 'global'
export GOOGLE_CLOUD_LOCATION=global

# This is the magic variable that switches to the Vertex AI backend
export GOOGLE_GENAI_USE_VERTEXAI=True

Note: These variables are set for your current terminal session only. If you close your terminal, you will need to set them again.

Step 2: Run the Embedding Script
Once the environment variables are set, you can run the script just as before. Each person should use their assigned worker ID.

Person 1 (Worker 0) runs:

python "notebooks/Phase 3/generate_embeddings_gemini.py" --worker-id 0

Person 2 (Worker 1) runs:

python "notebooks/Phase 3/generate_embeddings_gemini.py" --worker-id 1

Person 3 (Worker 2) runs:

python "notebooks/Phase 3/generate_embeddings_gemini.py" --worker-id 2

The script will now use your gcloud credentials and correctly send requests to the Vertex AI API without asking for an API key.