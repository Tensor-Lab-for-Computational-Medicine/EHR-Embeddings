# create_text_datasets.py
"""
Final main script to generate and serialize text datasets for all experimental arms.
This version includes the new 'F3' (Summary + Structured) representation.
"""
import os
import pickle
import pandas as pd
import logging
from tqdm import tqdm
from config import (
    PREPROCESSED_DATA_DIR,
    SERIALIZED_OUTPUT_DIR,
    REFERENCE_RANGES_PATH,
    PROMPTS,
    get_cache_prefix
)
from text_generator import generate_patient_representation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_preprocessed_data(prefix: str) -> dict:
    """Loads all necessary preprocessed data from pickle files."""
    logging.info(f"Loading data with prefix: {prefix}")
    datasets = {}
    required_files = {
        'X_train': 'X_train.pkl', 'X_val': 'X_val.pkl', 'X_test': 'X_test.pkl',
        'y_train': 'y_train.pkl', 'y_val': 'y_val.pkl', 'y_test': 'y_test.pkl',
        'scaler': 'scaler.pkl', 'label_encoders': 'label_encoders.pkl'
    }
    try:
        for name, filename in required_files.items():
            path = os.path.join(PREPROCESSED_DATA_DIR, f"{prefix}_{filename}")
            with open(path, 'rb') as f:
                datasets[name] = pickle.load(f)
            logging.info(f"Successfully loaded {filename}")
    except FileNotFoundError as e:
        logging.error(f"Fatal: Could not find required file: {e.filename}")
        exit(1)
        
    os.makedirs(SERIALIZED_OUTPUT_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        labels_path = os.path.join(SERIALIZED_OUTPUT_DIR, f'{split}_labels.csv')
        datasets[f'y_{split}'].to_csv(labels_path)
    logging.info("Saved train/val/test labels to CSV for easy lookup.")
    return datasets

def reverse_transform_data(df: pd.DataFrame, scaler, label_encoders: dict) -> pd.DataFrame:
    """Applies inverse transform for scaled and encoded features."""
    df_reversed = df.copy()
    numeric_cols = [col for col in scaler.get_feature_names_out() if col in df_reversed.columns]
    if numeric_cols:
        df_reversed[numeric_cols] = scaler.inverse_transform(df_reversed[numeric_cols])

    for encoded_col, encoder_info in label_encoders.items():
        if encoded_col in df_reversed.columns:
            le = encoder_info['encoder']
            original_col_name = encoded_col.replace('_encoded', '').title()
            df_reversed[original_col_name] = le.inverse_transform(df_reversed[encoded_col].astype(int))
            df_reversed.drop(columns=[encoded_col], inplace=True)
    return df_reversed

def main():
    """Main execution function."""
    DRY_RUN = False
    DRY_RUN_SAMPLE_SIZE = 1 

    logging.info("--- Starting Phase III: Text Dataset Serialization ---")
    if DRY_RUN:
        logging.warning(f"DRY RUN ENABLED: Processing {DRY_RUN_SAMPLE_SIZE} patient(s) per split.")
        
    cache_prefix = get_cache_prefix(dry_run=False)
    data = load_preprocessed_data(cache_prefix)

    try:
        df_ranges = pd.read_csv(REFERENCE_RANGES_PATH).set_index('feature_name')
        logging.info(f"Successfully loaded reference ranges from {REFERENCE_RANGES_PATH}")
    except FileNotFoundError:
        logging.error(f"Fatal: Could not find reference range file at '{REFERENCE_RANGES_PATH}'")
        exit(1)

    # FIX: Add 'F3' to the list of representations to generate.
    for representation in ['F1', 'F2', 'F3']:
        for prompt_key, prompt_text in PROMPTS.items():
            exp_name = f"{representation}_{prompt_key}"
            logging.info(f"Processing Experiment Arm: {exp_name}")
            
            for split in ['train', 'val', 'test']:
                output_dir = os.path.join(SERIALIZED_OUTPUT_DIR, exp_name, split)
                os.makedirs(output_dir, exist_ok=True)
                
                df_X_original = data[f'X_{split}'].head(DRY_RUN_SAMPLE_SIZE) if DRY_RUN else data[f'X_{split}']
                
                logging.info(f"Reversing transformations for '{split}' split...")
                df_X_readable = reverse_transform_data(df_X_original, data['scaler'], data['label_encoders'])
                
                logging.info(f"Generating {len(df_X_readable)} files for {exp_name}/{split}...")
                
                for icustay_id, patient_series in tqdm(df_X_readable.iterrows(), total=len(df_X_readable)):
                    gender = patient_series.get('Gender', 'Unknown')
                    
                    representation_text = generate_patient_representation(
                        patient_series, representation, df_ranges, gender
                    )
                    
                    final_text = f"{prompt_text}\n\n{representation_text}".strip()
                    
                    with open(os.path.join(output_dir, f"{icustay_id}.txt"), 'w', encoding='utf-8') as f:
                        f.write(final_text)

    logging.info("--- All experimental datasets have been successfully serialized. ---")

if __name__ == '__main__':
    main()