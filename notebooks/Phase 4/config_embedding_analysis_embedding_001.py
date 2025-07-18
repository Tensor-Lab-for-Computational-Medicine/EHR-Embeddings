# config_embedding_analysis.py
"""
Configuration for Phase V: Training XGBoost on Text Embeddings.
"""
import os

class Config:
    # --- Paths ---
    # Input directory for the embeddings generated in Phase 4
    EMBEDDING_DIR = 'notebooks/Phase 4/embeddings_models_embedding-001'
    
    # Input directory for the label files (.csv) generated in Phase 3
    LABEL_DIR = 'notebooks/Phase 3/phase_3_serialized_data'
    
    # Base output directory
    BASE_OUTPUT_DIR = 'notebooks/Phase 5/embedding_model_results/embedding-001'

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
    USE_GPU = True
    TARGET_VARIABLE = 'los_3'
    TARGET_VARIABLES = ['mort_hosp', 'los_3', 'los_7']
    SEED = 42
    N_OPTUNA_TRIALS = 10
    OPTUNA_TIMEOUT = 3600
    REUSE_EXISTING_STUDY = True

    def __init__(self, target_variable: str = None):

        # Override the default only when the caller supplies a meaningful value
        if target_variable:
            self.TARGET_VARIABLE = target_variable

        # Build an OUTPUT_DIR that is specific to the selected target variable
        self.OUTPUT_DIR = os.path.join(self.BASE_OUTPUT_DIR, self.TARGET_VARIABLE)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)