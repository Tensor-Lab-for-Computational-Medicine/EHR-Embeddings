"""
Configuration file for XGBoost analysis using Model 4B embeddings.
This config adapts the XGBoost analysis script to work with Model 4B embeddings.
"""

import os
from pathlib import Path

class Config:
    # === PATHS ===
    BASE_DIR = Path("./notebooks/Phase 7")
    EMBEDDING_DIR = BASE_DIR / "model_4b_text-embedding-004"  # Adjust model name as needed
    LABEL_DIR = EMBEDDING_DIR / "labels"
    OUTPUT_DIR = BASE_DIR / "xgboost_results_model4b"
    
    # === EXPERIMENTAL SETUP ===
    # For Model 4B, we typically have one experimental arm
    REPRESENTATIONS = ["MODEL4B"]  # Single representation type
    PROMPTS = ["P0"]  # Single prompt/parameter setting
    
    # === TARGET VARIABLE ===
    # Change this to run analysis for different tasks
    TARGET_VARIABLE = 'mort_hosp'  # Options: 'mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso'
    
    # === MODEL HYPERPARAMETERS ===
    SEED = 42
    N_OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 3600  # 1 hour timeout for hyperparameter optimization
    REUSE_EXISTING_STUDY = True
    USE_GPU = True  # Set to False if no GPU available
    
    # === DEBUGGING ===
    DRY_RUN = False  # Set to True for quick testing
    DRY_RUN_SUBSET_SIZE = 100  # Number of samples per class in dry run mode
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Verify that required directories exist
        if not self.EMBEDDING_DIR.exists():
            raise FileNotFoundError(f"Embedding directory not found: {self.EMBEDDING_DIR}")
        if not self.LABEL_DIR.exists():
            raise FileNotFoundError(f"Label directory not found: {self.LABEL_DIR}")

# Create directories when this module is imported
config = Config()
config.__post_init__() 