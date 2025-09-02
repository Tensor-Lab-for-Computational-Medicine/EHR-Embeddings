"""
Configuration for H2 Analysis: Hybrid Models and Orthogonality Testing
<<<<<<< HEAD
Updated to use text-embedding-005 model with minimal structure changes
=======
Updated to match actual file structure
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
"""
import os

class ConfigH2:
    def __init__(self):
        # Get the root directory - we're in Phase 6, go up to notebooks
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming this file is in /notebooks/Phase 6 - H2 Analysis/
        self.NOTEBOOKS_DIR = os.path.abspath(os.path.join(current_dir, '..'))
        self.ROOT_DIR = os.path.abspath(os.path.join(self.NOTEBOOKS_DIR, '..'))
        
<<<<<<< HEAD
        # --- Paths (keeping same structure as before) ---
        # Baseline numerical model outputs (Phase 1-2)
        self.BASELINE_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 1 and 2', 'phase_1_outputs')
        # Check if there's a readmission-specific baseline model, otherwise use generic one
        self.BASELINE_MODEL_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'readmission_30', 'model_1_xgboost_baseline.pkl')
        self.BASELINE_RESULTS_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'readmission_30', 'results_xgboost_baseline.pkl')
        # Fallback to generic model if readmission-specific doesn't exist
        if not os.path.exists(self.BASELINE_MODEL_PATH):
            self.BASELINE_MODEL_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'model_1_xgboost_baseline.pkl')
            self.BASELINE_RESULTS_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'results_xgboost_baseline.pkl')
=======
        # --- Paths (all relative to notebooks directory) ---
        # Baseline numerical model outputs (Phase 1-2)
        self.BASELINE_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 1 and 2', 'phase_1_outputs')
        self.BASELINE_MODEL_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'model_1_xgboost_baseline.pkl')
        self.BASELINE_RESULTS_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'results_xgboost_baseline.pkl')
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
        
        # Numerical feature data paths - using readmission_30 files
        self.NUMERICAL_DATA_PREFIX = 'preprocessed_mort_hosp_los_3_los_7_readmission_30_intervention_vent_intervention_vaso_trends_True_window_24_gap_6_seed_42'
        self.X_TRAIN_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_train.pkl')
        self.X_VAL_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_val.pkl')
        self.X_TEST_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_test.pkl')
        self.Y_TRAIN_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_train.pkl')
        self.Y_VAL_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_val.pkl')
        self.Y_TEST_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_test.pkl')
        
<<<<<<< HEAD
        # --- Champion embedding model using text-embedding-005 ---
        self.CHAMPION_EMBEDDING_MODEL = 'text-embedding-005'
        self.CHAMPION_ARM = 'F2_P5'  # Best performing arm for readmission_30
        
        # --- Embedding model paths - text-embedding-005 models in Phase 5 ---
        self.EMBEDDING_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 5', 'embedding_model_results', 'text-embedding-005', 'readmission_30')
        self.CHAMPION_MODEL_PATH = os.path.join(self.EMBEDDING_MODEL_DIR, f'model_{self.CHAMPION_ARM}.pkl')
        self.CHAMPION_RESULTS_PATH = os.path.join(self.EMBEDDING_MODEL_DIR, f'results_{self.CHAMPION_ARM}.pkl')
        
        # --- Embedding data paths - text-embedding-005 embeddings in Phase 4 ---
        self.EMBEDDING_DATA_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 4', 'embeddings_text-embedding-005', 'F2_P5')
        
        # Label data paths (unchanged - same labels for both 004 and 005)
        self.LABEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 3', 'phase_3_serialized_data')
        
        # Output directory for H2 analysis (separate for 005)
        self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results_005_readmission')
=======
        # Champion embedding model (identified as F3_P2 from text-embedding-004)
        self.CHAMPION_EMBEDDING_MODEL = 'text-embedding-004'
        self.CHAMPION_ARM = 'F3_P2'
        
        # Embedding model paths - models are in Phase 5
        self.EMBEDDING_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 5', 'embedding_model_results')
        self.CHAMPION_MODEL_PATH = os.path.join(self.EMBEDDING_MODEL_DIR, f'model_{self.CHAMPION_ARM}.pkl')
        self.CHAMPION_RESULTS_PATH = os.path.join(self.EMBEDDING_MODEL_DIR, f'results_{self.CHAMPION_ARM}.pkl')
        
        # Embedding data paths - embeddings are in Phase 4
        self.EMBEDDING_DATA_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 4', 'phase_4_embeddings', self.CHAMPION_ARM)
        
        # Label data paths
        self.LABEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 3', 'phase_3_serialized_data')
        
        # Output directory for H2 analysis (local to current directory)
        self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results')
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
        
        # --- Experiment Settings ---
        self.TARGET_VARIABLE = 'readmission_30'
        self.SEED = 42
        self.CLASSIFICATION_THRESHOLD = 0.5
        
        # Hyperparameter tuning settings
        self.N_OPTUNA_TRIALS = 20
        self.OPTUNA_TIMEOUT = 3600
        
        # Statistical testing settings
        self.CORRELATION_THRESHOLD = 0.4  # H2 threshold for "weakly correlated"
        self.N_BOOTSTRAP = 1000
        self.CONFIDENCE_LEVEL = 0.95
        
        # =============================================================================
        # SUBGROUP DISCOVERY SETTINGS (NEW FOR H2b ANALYSIS)
        # =============================================================================
<<<<<<< HEAD
        # Enable/disable subgroup discovery
        self.USE_SUBGROUP_DISCOVERY = True
        
        # Subgroup Discovery Algorithm Parameters
        self.SUBGROUP_MIN_SUPPORT = 0.05
        self.SUBGROUP_MAX_DEPTH = 3
        self.SUBGROUP_TOP_K = 10
        
        # Quality thresholds for H2b hypothesis evaluation
        self.SUBGROUP_MIN_QUALITY = 0.1
        self.SUBGROUP_MIN_COVERAGE_PCT = 5
        self.SUBGROUP_MIN_LIFT = 1.5
=======
        # Enable/disable subgroup discovery (will fall back to univariate if False or if pysubgroup not installed)
        self.USE_SUBGROUP_DISCOVERY = True
        
        # Subgroup Discovery Algorithm Parameters
        self.SUBGROUP_MIN_SUPPORT = 0.05    # Minimum support: subgroup must cover at least 5% of population
        self.SUBGROUP_MAX_DEPTH = 3         # Maximum depth of conjunctive rules (e.g., depth 3 = up to 3 ANDed conditions)
        self.SUBGROUP_TOP_K = 10            # Number of top subgroups to discover per analysis
        
        # Quality thresholds for H2b hypothesis evaluation
        self.SUBGROUP_MIN_QUALITY = 0.1     # Minimum WRAcc quality score to consider a pattern meaningful
        self.SUBGROUP_MIN_COVERAGE_PCT = 5  # Minimum coverage percentage for a meaningful pattern
        self.SUBGROUP_MIN_LIFT = 1.5        # Minimum lift (ratio to baseline) for a meaningful pattern
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
        
        # Number of analyses that must yield meaningful patterns to support H2b
        self.SUBGROUP_MIN_MEANINGFUL_ANALYSES = 2
        
        # =============================================================================
        
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
        
<<<<<<< HEAD
        print(f"[INFO] Checking files from notebooks directory: {self.NOTEBOOKS_DIR}")
=======
        print(f"🔍 Checking files from notebooks directory: {self.NOTEBOOKS_DIR}")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
<<<<<<< HEAD
                print(f"[ERROR] Missing: {os.path.relpath(file_path, self.NOTEBOOKS_DIR)}")
            else:
                print(f"[SUCCESS] Found: {os.path.relpath(file_path, self.NOTEBOOKS_DIR)}")
=======
                print(f"❌ Missing: {os.path.relpath(file_path, self.NOTEBOOKS_DIR)}")
            else:
                print(f"✅ Found: {os.path.relpath(file_path, self.NOTEBOOKS_DIR)}")
        
        # Note: We need to check for X_train file
        if self.X_TRAIN_NUM_PATH in missing_files:
            # Check if it exists without the prefix
            alt_path = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_train.pkl')
            if not os.path.exists(alt_path):
                print(f"⚠️  Note: X_train file missing - this might be the source of data leakage if train+val were combined")
        
        if missing_files:
            print(f"\n❌ Missing {len(missing_files)} required files")
            # Don't raise error immediately - some files might be optional
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
        
        # Check embedding directories exist
        embedding_dirs = [
            os.path.join(self.EMBEDDING_DATA_DIR, split) 
            for split in ['train', 'val', 'test']
        ]
        missing_dirs = []
        for dir_path in embedding_dirs:
            if not os.path.isdir(dir_path):
                missing_dirs.append(dir_path)
<<<<<<< HEAD
                print(f"[ERROR] Missing directory: {os.path.relpath(dir_path, self.NOTEBOOKS_DIR)}")
            else:
                # Count files in directory
                num_files = len([f for f in os.listdir(dir_path) if f.endswith('.npy')])
                print(f"[SUCCESS] Found directory: {os.path.relpath(dir_path, self.NOTEBOOKS_DIR)} ({num_files} embeddings)")
=======
                print(f"❌ Missing directory: {os.path.relpath(dir_path, self.NOTEBOOKS_DIR)}")
            else:
                # Count files in directory
                num_files = len([f for f in os.listdir(dir_path) if f.endswith('.npy')])
                print(f"✅ Found directory: {os.path.relpath(dir_path, self.NOTEBOOKS_DIR)} ({num_files} embeddings)")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
        
        # Check label files
        label_files = ['train_labels.csv', 'val_labels.csv', 'test_labels.csv']
        for label_file in label_files:
            label_path = os.path.join(self.LABEL_DIR, label_file)
            if os.path.exists(label_path):
<<<<<<< HEAD
                print(f"[SUCCESS] Found label file: {label_file}")
            else:
                print(f"[ERROR] Missing label file: {label_file}")
=======
                print(f"✅ Found label file: {label_file}")
            else:
                print(f"❌ Missing label file: {label_file}")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
                missing_files.append(label_path)
        
        if missing_dirs:
            raise FileNotFoundError(f"Missing embedding directories: {[os.path.relpath(d, self.NOTEBOOKS_DIR) for d in missing_dirs]}")
        
        if missing_files:
<<<<<<< HEAD
            print(f"\n[WARNING]  Warning: {len(missing_files)} files are missing.")
            print("The analysis may fail if these are required.")
        else:
            print("\n[SUCCESS] All required files and directories found!")
=======
            print(f"\n⚠️  Warning: {len(missing_files)} files are missing.")
            print("The analysis may fail if these are required.")
        else:
            print("\n✅ All required files and directories found!")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
        
        return len(missing_files) == 0
    
    def check_data_leakage(self):
        """Check for potential data leakage by examining file sizes and dates"""
        import datetime
        
<<<<<<< HEAD
        print("\n[INFO] Checking for potential data leakage indicators...")
        
        # Check if X_train exists
        if not os.path.exists(self.X_TRAIN_NUM_PATH):
            print("[WARNING]  WARNING: X_train file not found!")
=======
        print("\n🔍 Checking for potential data leakage indicators...")
        
        # Check if X_train exists
        if not os.path.exists(self.X_TRAIN_NUM_PATH):
            print("⚠️  WARNING: X_train file not found!")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
            return True
        
        # Check file sizes to see if they make sense
        try:
            train_size = os.path.getsize(self.X_TRAIN_NUM_PATH) / (1024*1024)  # MB
            val_size = os.path.getsize(self.X_VAL_NUM_PATH) / (1024*1024)
            test_size = os.path.getsize(self.X_TEST_NUM_PATH) / (1024*1024)
            
<<<<<<< HEAD
            print(f"[DATA] Data file sizes:")
=======
            print(f"📊 Data file sizes:")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
            print(f"   X_train: {train_size:.1f} MB")
            print(f"   X_val: {val_size:.1f} MB")
            print(f"   X_test: {test_size:.1f} MB")
            
            # Check modification times
            train_time = datetime.datetime.fromtimestamp(os.path.getmtime(self.X_TRAIN_NUM_PATH))
            val_time = datetime.datetime.fromtimestamp(os.path.getmtime(self.X_VAL_NUM_PATH))
            model_time = datetime.datetime.fromtimestamp(os.path.getmtime(self.BASELINE_MODEL_PATH))
            
<<<<<<< HEAD
            print(f"\n[TIME] File modification times:")
=======
            print(f"\n📅 File modification times:")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
            print(f"   X_train: {train_time}")
            print(f"   X_val: {val_time}")
            print(f"   Model: {model_time}")
            
            # If model was modified after data files, it might be OK
            if model_time > max(train_time, val_time):
<<<<<<< HEAD
                print("\n[SUCCESS] Model was trained after data files were created")
            else:
                print("\n[WARNING]  WARNING: Model file is older than data files!")
=======
                print("\n✅ Model was trained after data files were created")
            else:
                print("\n⚠️  WARNING: Model file is older than data files!")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
                
        except Exception as e:
            print(f"Error checking files: {e}")
            
<<<<<<< HEAD
        print("\n[NOTE] Note: The 100% validation AUROC suggests the model may have been:")
=======
        print("\n💡 Note: The 100% validation AUROC suggests the model may have been:")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
        print("   1. Trained on combined train+val data")
        print("   2. Evaluated on data it was trained on")
        print("   3. Subject to target leakage through feature engineering")
        
        return False
    
    def validate_subgroup_discovery(self):
        """Check if subgroup discovery can be used"""
        try:
            import pysubgroup
<<<<<<< HEAD
            print("\n[SUCCESS] pysubgroup is installed - Subgroup Discovery analysis available")
=======
            print("\n✅ pysubgroup is installed - Subgroup Discovery analysis available")
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
            
            if self.USE_SUBGROUP_DISCOVERY:
                print("   Subgroup Discovery is ENABLED")
                print(f"   - Min support: {self.SUBGROUP_MIN_SUPPORT*100:.0f}% of population")
                print(f"   - Max rule depth: {self.SUBGROUP_MAX_DEPTH} conditions")
                print(f"   - Top K patterns: {self.SUBGROUP_TOP_K}")
                print(f"   - Min quality (WRAcc): {self.SUBGROUP_MIN_QUALITY}")
                print(f"   - Min lift: {self.SUBGROUP_MIN_LIFT}x baseline")
            else:
                print("   Subgroup Discovery is DISABLED in config (using univariate analysis)")
            
            return True
            
        except ImportError:
<<<<<<< HEAD
            print("\n[WARNING]  pysubgroup not installed - will fall back to univariate analysis")
            print("   To enable Subgroup Discovery: pip install pysubgroup")
            return False

# Main function for testing/debugging
if __name__ == "__main__":
    print("="*60)
    print("H2 CONFIGURATION DEBUG")
    print("="*60)
    
    config = ConfigH2()
    
    print("\n📁 KEY PATHS:")
    print(f"   Notebooks dir: {config.NOTEBOOKS_DIR}")
    print(f"   Target variable: {config.TARGET_VARIABLE}")
    print(f"   Champion arm: {config.CHAMPION_ARM}")
    print(f"   Embedding model: {config.CHAMPION_EMBEDDING_MODEL}")
    
    print("\n" + "="*60)
    print("PATH VALIDATION")
    print("="*60)
    all_valid = config.validate_paths()
    
    if all_valid:
        print("\n" + "="*60)
        print("DATA LEAKAGE CHECK")
        print("="*60)
        config.check_data_leakage()
        
        print("\n" + "="*60)
        print("SUBGROUP DISCOVERY CHECK")
        print("="*60)
        config.validate_subgroup_discovery()
    else:
        print("\n[WARNING]  Fix missing files before proceeding with analysis")
    
    # Additional debugging: check if we need to load labels separately
    print("\n" + "="*60)
    print("LABEL EXTRACTION INFO")
    print("="*60)
    print("[INFO] Note: Since TARGET_VARIABLE is 'readmission_30' but numerical data is from 'mort_hosp',")
    print("   you'll need to extract readmission_30 labels from the CSV files in your analysis script:")
    print("\n   import pandas as pd")
    print("   train_labels = pd.read_csv(os.path.join(config.LABEL_DIR, 'train_labels.csv'))")
    print("   y_train = train_labels['readmission_30'].values")
    print("   # Repeat for val and test sets")
    
    print("\n" + "="*60)
    print("READY STATUS")
    print("="*60)
    if all_valid:
        print("[SUCCESS] Configuration is ready for H2 analysis!")
        print(f"   Output will be saved to: {config.OUTPUT_DIR}")
    else:
        print("[ERROR] Configuration has issues that need to be resolved")
=======
            print("\n⚠️  pysubgroup not installed - will fall back to univariate analysis")
            print("   To enable Subgroup Discovery: pip install pysubgroup")
            return False
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
