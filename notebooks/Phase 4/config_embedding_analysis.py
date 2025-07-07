# config_embedding_analysis.py
"""
Configuration for Phase V: Training XGBoost on Text Embeddings.
"""
import os

class Config:
    # --- Paths ---
    # Input directory for the embeddings generated in Phase 4
    EMBEDDING_DIR = 'notebooks/Phase 4/phase_4_embeddings'
    
    # Input directory for the label files (.csv) generated in Phase 3
    LABEL_DIR = 'notebooks/Phase 3/phase_3_serialized_data'
    
    # Output directory for models, results, and logs from this analysis
    OUTPUT_DIR = 'notebooks/Phase 5/embedding_model_results'

    # --- Experiment Setup ---
    # Set to True to run the analysis on only the FIRST experimental condition
    # on a SMALL SUBSET of the data.
    DRY_RUN = False
    
    # FIX: Added the missing attribute
    # Number of samples to use from each split (train/val/test) during a dry run.
    DRY_RUN_SUBSET_SIZE = 100 
    
    # Define all experimental arms to be tested.
    REPRESENTATIONS = ['F1', 'F2', 'F3']
    PROMPTS = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
    
    # --- XGBoost & Optuna Settings ---
    TARGET_VARIABLE = 'mort_hosp'
    SEED = 42
    N_OPTUNA_TRIALS = 10
    OPTUNA_TIMEOUT = 3600
    REUSE_EXISTING_STUDY = True

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)