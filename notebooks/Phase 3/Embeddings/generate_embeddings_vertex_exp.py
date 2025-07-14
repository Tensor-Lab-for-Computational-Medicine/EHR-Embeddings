# generate_embeddings_batched.py
"""
Generates embeddings for train, test, and validation sets using a
Hugging Face SentenceTransformer model.

This script reads text files, processes them in batches, and saves the
resulting embeddings as .npy files, mirroring the input directory structure.
It will automatically use a CUDA-enabled GPU if available.
All configurations are hardcoded at the top of the file.
"""
import os
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# EDIT THESE VARIABLES TO MATCH YOUR SETUP

# Model: The Hugging Face SentenceTransformer model to use.
MODEL_NAME = "abhinand/MedEmbed-small-v0.1"

# --- Path Configuration ---
# These paths are relative to the project's root directory (e.g., where you run the python command from).
# The base input directory containing all serialized text files (including train/test/val subfolders).
BASE_INPUT_DIR = 'notebooks/Phase 3/phase_3_serialized_data'
# The base output directory where the embeddings will be saved.
BASE_OUTPUT_DIR = 'notebooks/Phase 4'


# Processing: The number of text files to process in a single batch.
BATCH_SIZE = 1000

# --- End of Configuration ---


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)
# ---

def main():
    """
    Main function to find all text files, load the model, and process the files in batches.
    """
    logging.info("--- Starting Embedding Generation ---")

    # --- GPU/Device Setup ---
    # Check for CUDA-enabled GPU and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # Load the SentenceTransformer model from Hugging Face onto the selected device
    try:
        logging.info(f"Loading model: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME, device=device)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Fatal: Failed to load Hugging Face model. Error: {e}")
        return

    # --- File Discovery ---
    # Define the main output directory for this model's embeddings
    embedding_output_dir = os.path.join(BASE_OUTPUT_DIR, f"embeddings_{MODEL_NAME.replace('/', '_')}")
    logging.info(f"Output will be saved in: {embedding_output_dir}")

    # Recursively find all .txt files and check if their embeddings already exist
    filepaths_to_process = []
    if not os.path.isdir(BASE_INPUT_DIR):
        logging.error(f"Base input directory not found: {BASE_INPUT_DIR}. Exiting.")
        return

    logging.info(f"Scanning for text files in {BASE_INPUT_DIR}...")
    for root, _, files in os.walk(BASE_INPUT_DIR):
        for filename in files:
            if filename.endswith('.txt'):
                input_path = os.path.join(root, filename)
                
                # Construct the corresponding output path, preserving the directory structure
                relative_path = os.path.relpath(input_path, BASE_INPUT_DIR)
                output_path = os.path.join(embedding_output_dir, os.path.splitext(relative_path)[0] + '.npy')

                # Only add the file to the list if its embedding doesn't already exist
                if not os.path.exists(output_path):
                    filepaths_to_process.append(input_path)

    if not filepaths_to_process:
        logging.info("All embeddings are already generated. Nothing to do.")
        return

    logging.info(f"Found {len(filepaths_to_process)} new text files to process.")

    # --- Batch Processing ---
    for i in tqdm(range(0, len(filepaths_to_process), BATCH_SIZE), desc="Generating Embeddings"):
        batch_paths = filepaths_to_process[i:i + BATCH_SIZE]
        batch_texts = []
        
        # Read the content of each file in the batch
        for path in batch_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    batch_texts.append(f.read())
            except Exception as e:
                logging.error(f"Could not read file {path}. Skipping. Error: {e}")
                batch_texts.append("") # Add empty string as a placeholder to maintain batch alignment

        # Generate embeddings for the entire batch of texts
        try:
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)

            # Save each embedding to its corresponding .npy file
            for path, embedding in zip(batch_paths, batch_embeddings):
                relative_path = os.path.relpath(path, BASE_INPUT_DIR)
                output_path = os.path.join(embedding_output_dir, os.path.splitext(relative_path)[0] + '.npy')
                
                # Ensure the output directory exists before saving
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, embedding)

        except Exception as e:
            logging.error(f"Failed to process batch starting with file {batch_paths[0]}. Error: {e}")

    logging.info("--- All files processed. Embedding generation complete. ---")


if __name__ == '__main__':
    main()
