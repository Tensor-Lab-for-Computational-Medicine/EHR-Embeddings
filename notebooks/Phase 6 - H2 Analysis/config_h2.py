# config_h2.py
"""
Configuration for H2 Analysis: Hybrid Models and Orthogonality Testing
"""
import os

class ConfigH2:
    def __init__(self):
        # Get the root directory (go up from Phase 6 to root)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # --- Paths (all relative to root directory) ---
        # Baseline numerical model outputs (Phase 1-2)
        self.BASELINE_MODEL_DIR = os.path.join(self.ROOT_DIR, 'notebooks', 'Phase 1 and 2', 'phase_1_outputs')
        self.BASELINE_MODEL_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'model_1_xgboost_baseline.pkl')
        self.BASELINE_RESULTS_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'results_xgboost_baseline.pkl')
        
        # Numerical feature data paths
        self.NUMERICAL_DATA_PREFIX = 'preprocessed_mort_hosp_trends_True_window_24_gap_6_seed_42'
        self.X_TRAIN_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_train.pkl')
        self.X_VAL_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_val.pkl')
        self.X_TEST_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_test.pkl')
        self.Y_TRAIN_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_train.pkl')
        self.Y_VAL_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_val.pkl')
        self.Y_TEST_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_test.pkl')
        
        # Champion embedding model (identified as F3_P2 from text-embedding-004)
        self.CHAMPION_EMBEDDING_MODEL = 'text-embedding-004'
        self.CHAMPION_ARM = 'F3_P2'
        self.EMBEDDING_MODEL_DIR = os.path.join(self.ROOT_DIR, 'notebooks', 'Phase 5', 'embedding_model_results')
        self.CHAMPION_MODEL_PATH = os.path.join(self.EMBEDDING_MODEL_DIR, f'model_{self.CHAMPION_ARM}.pkl')
        self.CHAMPION_RESULTS_PATH = os.path.join(self.EMBEDDING_MODEL_DIR, f'results_{self.CHAMPION_ARM}.pkl')
        
        # Embedding data paths for champion model
        self.EMBEDDING_DATA_DIR = os.path.join(self.ROOT_DIR, 'notebooks', 'Phase 4', 'phase_4_embeddings', self.CHAMPION_ARM)
        
        # Label data paths
        self.LABEL_DIR = os.path.join(self.ROOT_DIR, 'notebooks', 'Phase 3', 'phase_3_serialized_data')
        
        # Output directory for H2 analysis (local to current directory)
        self.OUTPUT_DIR = 'h2_results'
        
        # --- Experiment Settings ---
        self.TARGET_VARIABLE = 'mort_hosp'
        self.SEED = 42
        
        # Hyperparameter tuning settings
        self.N_OPTUNA_TRIALS = 20
        self.OPTUNA_TIMEOUT = 3600
        
        # Statistical testing settings
        self.CORRELATION_THRESHOLD = 0.4  # H2 threshold for "weakly correlated"
        self.N_BOOTSTRAP = 1000
        self.CONFIDENCE_LEVEL = 0.95
        
        # Debugging/testing
        self.DRY_RUN = False
        self.DRY_RUN_SAMPLES = 1000
        
        # Create output directory
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
    def validate_paths(self):
        """Validate that all required files exist"""
        required_files = [
            self.BASELINE_MODEL_PATH,
            self.BASELINE_RESULTS_PATH,
            self.X_TRAIN_NUM_PATH,
            self.X_VAL_NUM_PATH, 
            self.X_TEST_NUM_PATH,
            self.Y_TRAIN_PATH,
            self.Y_VAL_PATH,
            self.Y_TEST_PATH,
            self.CHAMPION_MODEL_PATH,
            self.CHAMPION_RESULTS_PATH
        ]
        
        print(f"🔍 Checking files from root directory: {self.ROOT_DIR}")
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
            else:
                print(f"✅ Found: {os.path.relpath(file_path, self.ROOT_DIR)}")
        
        if missing_files:
            print(f"❌ Missing files:")
            for file_path in missing_files:
                print(f"   - {os.path.relpath(file_path, self.ROOT_DIR)}")
            raise FileNotFoundError(f"Missing required files: {[os.path.relpath(f, self.ROOT_DIR) for f in missing_files]}")
        
        # Check embedding directories exist
        embedding_dirs = [
            os.path.join(self.EMBEDDING_DATA_DIR, split) 
            for split in ['train', 'val', 'test']
        ]
        missing_dirs = []
        for dir_path in embedding_dirs:
            if not os.path.isdir(dir_path):
                missing_dirs.append(dir_path)
            else:
                print(f"✅ Found directory: {os.path.relpath(dir_path, self.ROOT_DIR)}")
        
        if missing_dirs:
            print(f"❌ Missing directories:")
            for dir_path in missing_dirs:
                print(f"   - {os.path.relpath(dir_path, self.ROOT_DIR)}")
            raise FileNotFoundError(f"Missing embedding directories: {[os.path.relpath(d, self.ROOT_DIR) for d in missing_dirs]}")
        
        print("✅ All required files and directories found!")
        return True