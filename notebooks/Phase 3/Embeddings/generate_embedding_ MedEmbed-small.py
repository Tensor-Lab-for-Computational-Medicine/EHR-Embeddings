# generate_embeddings_v2.py
"""
Generates embeddings for a dataset of text files using a Hugging Face
SentenceTransformer model.

This script scans an input directory for .txt files, processes them in batches,
and saves the resulting embeddings as .npy files, mirroring the input
directory structure in a specified output directory.

It automatically uses a CUDA-enabled GPU if available and skips files for which
embeddings already exist. All configurations are handled via command-line
arguments.
"""
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate sentence embeddings from text files.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="abhinand/MedEmbed-small-v0.1",
        help="The Hugging Face SentenceTransformer model to use."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("notebooks/Phase 3/phase_3_serialized_data"),
        help="Base input directory containing text files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/Phase 4"),
        help="Base output directory to save embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of files to process in a single batch."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="NumPy dtype to save the embeddings. float16 saves space and can be faster."
    )
    return parser.parse_args()

def discover_files_to_process(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """
    Scans the input directory for .txt files and returns a list of
    (input_path, output_path) tuples for files that need processing.
    """
    logging.info(f"Scanning for text files in '{input_dir}'...")
    
    txt_files = list(input_dir.rglob('*.txt'))
    if not txt_files:
        logging.warning(f"No .txt files found in {input_dir}.")
        return []

    files_to_process = []
    for input_path in txt_files:
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix('.npy')
        
        if not output_path.exists():
            files_to_process.append((input_path, output_path))
    
    logging.info(f"Found {len(files_to_process)} new files to process out of {len(txt_files)} total.")
    return files_to_process

def main():
    """Main function to load the model and generate embeddings."""
    args = get_args()
    
    # --- Device Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # --- Model Loading ---
    try:
        logging.info(f"Loading model: {args.model_name}...")
        model = SentenceTransformer(args.model_name, device=device)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Fatal: Failed to load model '{args.model_name}'. Error: {e}")
        return

    # --- Prepare Output Directory ---
    model_name_safe = args.model_name.replace('/', '_')
    output_dir_root = args.output_dir / f"embeddings_{model_name_safe}"
    
    # --- File Discovery ---
    files_to_process = discover_files_to_process(args.input_dir, output_dir_root)
    if not files_to_process:
        logging.info("All embeddings are already generated. Nothing to do.")
        return

    # --- Processing ---
    logging.info(f"Starting embedding generation with batch size {args.batch_size}...")
    
    num_files = len(files_to_process)
    for i in tqdm(range(0, num_files, args.batch_size), desc="Embedding Batches"):
        batch = files_to_process[i:i + args.batch_size]
        batch_input_paths = [item[0] for item in batch]
        batch_output_paths = [item[1] for item in batch]

        # Read file contents for the batch
        batch_texts = []
        for path in batch_input_paths:
            try:
                batch_texts.append(path.read_text(encoding='utf-8'))
            except Exception as e:
                logging.error(f"Could not read file {path}. Skipping. Error: {e}")
                batch_texts.append("") # Placeholder to maintain alignment

        # Generate and save embeddings
        try:
            batch_embeddings = model.encode(
                batch_texts,
                show_progress_bar=False,
                # Using tensor conversion can be slightly faster on GPU
                convert_to_tensor=True 
            )
            
            # Move embeddings to CPU and convert to NumPy for saving
            batch_embeddings_np = batch_embeddings.cpu().numpy().astype(args.dtype)
            
            for output_path, embedding in zip(batch_output_paths, batch_embeddings_np):
                # Ensure the parent directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, embedding)

        except Exception as e:
            logging.error(f"Failed to process batch starting with {batch_input_paths[0]}. Error: {e}")
    
    logging.info("--- Embedding generation complete. ---")

if __name__ == '__main__':
    main()