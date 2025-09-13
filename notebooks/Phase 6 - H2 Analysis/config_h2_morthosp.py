# config_h2.py
"""
Configuration for H2 Analysis: Hybrid Models and Orthogonality Testing
Updated to match actual file structure
"""
import os


class ConfigH2:
   def __init__(self):
       # Get the root directory - we're in Phase 6, go up to notebooks
       current_dir = os.path.dirname(os.path.abspath(__file__))
       # Assuming this file is in /notebooks/Phase 6 - H2 Analysis/
       self.NOTEBOOKS_DIR = os.path.abspath(os.path.join(current_dir, '..'))
       self.ROOT_DIR = os.path.abspath(os.path.join(self.NOTEBOOKS_DIR, '..'))
      
       # --- Paths (all relative to notebooks directory) ---
       # Baseline numerical model outputs (Phase 1-2)
       self.BASELINE_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 1 and 2', 'phase_1_outputs')
       self.BASELINE_MODEL_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'model_1_xgboost_baseline_calibrated.pkl')
       self.BASELINE_RESULTS_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'results_xgboost_baseline.pkl')
      
       # Numerical feature data paths
       self.NUMERICAL_DATA_PREFIX = 'preprocessed_mort_hosp_trends_True_window_24_gap_6_seed_42'
       self.X_TRAIN_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_train.pkl')
       self.X_VAL_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_val.pkl')
       self.X_TEST_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_test.pkl')
       self.Y_TRAIN_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_train.pkl')
       self.Y_VAL_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_val.pkl')
       self.Y_TEST_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_test.pkl')
      
       # Champion embedding model (identified as F3_P5 from text-embedding-004)
       self.CHAMPION_ARM = 'F3_P5'
      
       # Embedding model paths - models are in Phase 5
       self.EMBEDDING_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 5', 'embedding_model_results')
       self.CHAMPION_MODEL_PATH = os.path.join(self.EMBEDDING_MODEL_DIR, f'model_{self.CHAMPION_ARM}_calibrated.pkl')
       self.CHAMPION_RESULTS_PATH = os.path.join(self.EMBEDDING_MODEL_DIR, f'results_{self.CHAMPION_ARM}.pkl')
      
       # Embedding data paths - embeddings are in Phase 4
       self.EMBEDDING_DATA_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 4', 'phase_4_embeddings', self.CHAMPION_ARM)
      
       # Label data paths
       self.LABEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 3', 'phase_3_serialized_data')
      
       # Output directory for H2 analysis (local to current directory)
       self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results')
      
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
      
       # =============================================================================
       # SUBGROUP DISCOVERY SETTINGS (NEW FOR H2b ANALYSIS)
       # =============================================================================
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
      
       print(f"🔍 Checking files from notebooks directory: {self.NOTEBOOKS_DIR}")
       missing_files = []
       for file_path in required_files:
           if not os.path.exists(file_path):
               missing_files.append(file_path)
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
      
       # Check embedding directories exist
       embedding_dirs = [
           os.path.join(self.EMBEDDING_DATA_DIR, split)
           for split in ['train', 'val', 'test']
       ]
       missing_dirs = []
       for dir_path in embedding_dirs:
           if not os.path.isdir(dir_path):
               missing_dirs.append(dir_path)
               print(f"❌ Missing directory: {os.path.relpath(dir_path, self.NOTEBOOKS_DIR)}")
           else:
               # Count files in directory
               num_files = len([f for f in os.listdir(dir_path) if f.endswith('.npy')])
               print(f"✅ Found directory: {os.path.relpath(dir_path, self.NOTEBOOKS_DIR)} ({num_files} embeddings)")
      
       # Check label files
       label_files = ['train_labels.csv', 'val_labels.csv', 'test_labels.csv']
       for label_file in label_files:
           label_path = os.path.join(self.LABEL_DIR, label_file)
           if os.path.exists(label_path):
               print(f"✅ Found label file: {label_file}")
           else:
               print(f"❌ Missing label file: {label_file}")
               missing_files.append(label_path)
      
       if missing_dirs:
           raise FileNotFoundError(f"Missing embedding directories: {[os.path.relpath(d, self.NOTEBOOKS_DIR) for d in missing_dirs]}")
      
       if missing_files:
           print(f"\n⚠️  Warning: {len(missing_files)} files are missing.")
           print("The analysis may fail if these are required.")
       else:
           print("\n✅ All required files and directories found!")
      
       return len(missing_files) == 0
  
   def check_data_leakage(self):
       """Check for potential data leakage by examining file sizes and dates"""
       import datetime
      
       print("\n🔍 Checking for potential data leakage indicators...")
      
       # Check if X_train exists
       if not os.path.exists(self.X_TRAIN_NUM_PATH):
           print("⚠️  WARNING: X_train file not found!")
           return True
      
       # Check file sizes to see if they make sense
       try:
           train_size = os.path.getsize(self.X_TRAIN_NUM_PATH) / (1024*1024)  # MB
           val_size = os.path.getsize(self.X_VAL_NUM_PATH) / (1024*1024)
           test_size = os.path.getsize(self.X_TEST_NUM_PATH) / (1024*1024)
          
           print(f"📊 Data file sizes:")
           print(f"   X_train: {train_size:.1f} MB")
           print(f"   X_val: {val_size:.1f} MB")
           print(f"   X_test: {test_size:.1f} MB")
          
           # Check modification times
           train_time = datetime.datetime.fromtimestamp(os.path.getmtime(self.X_TRAIN_NUM_PATH))
           val_time = datetime.datetime.fromtimestamp(os.path.getmtime(self.X_VAL_NUM_PATH))
           model_time = datetime.datetime.fromtimestamp(os.path.getmtime(self.BASELINE_MODEL_PATH))
          
           print(f"\n📅 File modification times:")
           print(f"   X_train: {train_time}")
           print(f"   X_val: {val_time}")
           print(f"   Model: {model_time}")
          
           # If model was modified after data files, it might be OK
           if model_time > max(train_time, val_time):
               print("\n✅ Model was trained after data files were created")
           else:
               print("\n⚠️  WARNING: Model file is older than data files!")
              
       except Exception as e:
           print(f"Error checking files: {e}")
          
       print("\n💡 Note: The 100% validation AUROC suggests the model may have been:")
       print("   1. Trained on combined train+val data")
       print("   2. Evaluated on data it was trained on")
       print("   3. Subject to target leakage through feature engineering")
      
       return False
  
   def validate_subgroup_discovery(self):
       """Check if subgroup discovery can be used"""
       try:
           import pysubgroup
           print("\n✅ pysubgroup is installed - Subgroup Discovery analysis available")
          
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
           print("\n⚠️  pysubgroup not installed - will fall back to univariate analysis")
           print("   To enable Subgroup Discovery: pip install pysubgroup")
           return False
