"""
Configuration for H2 Analysis: Hybrid Models and Orthogonality Testing
Updated to match actual file structure and add debugging
"""
import os
import json
import pickle


class ConfigH2:
    def __init__(self):
        # Get the root directory - we're in Phase 6, go up to notebooks
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming this file is in /notebooks/Phase 6 - H2 Analysis/
        self.NOTEBOOKS_DIR = os.path.abspath(os.path.join(current_dir, '..'))
        self.ROOT_DIR = os.path.abspath(os.path.join(self.NOTEBOOKS_DIR, '..'))
        
        print(f"📁 Current directory: {current_dir}")
        print(f"📁 Notebooks directory: {self.NOTEBOOKS_DIR}")
        print(f"📁 Root directory: {self.ROOT_DIR}")
        
        # --- Paths (all relative to notebooks directory) ---
        # Baseline numerical model outputs (Phase 1-2)
        self.BASELINE_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 1 and 2', 'phase_1_outputs')
        
        self.BASELINE_MODEL_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'model_1_xgboost_baseline_calibrated.pkl')

        # If results file doesn't exist in readmission_30, try parent directory
        self.BASELINE_RESULTS_PATH = os.path.join(self.BASELINE_MODEL_DIR, 'results_xgboost_baseline.pkl')

        
        # Numerical feature data paths - these appear to be in the main phase_1_outputs directory
        self.NUMERICAL_DATA_PREFIX = 'preprocessed_mort_hosp_los_3_los_7_readmission_30_intervention_vent_intervention_vaso_trends_True_window_24_gap_6_seed_42'
        self.X_TRAIN_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_train.pkl')
        self.X_VAL_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_val.pkl')
        self.X_TEST_NUM_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_X_test.pkl')
        self.Y_TRAIN_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_train.pkl')
        self.Y_VAL_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_val.pkl')
        self.Y_TEST_PATH = os.path.join(self.BASELINE_MODEL_DIR, f'{self.NUMERICAL_DATA_PREFIX}_y_test.pkl')
        
        # Champion embedding model (F2_P5 from text-embedding-005)
        self.CHAMPION_ARM = 'F2_P5'
        
        # Embedding model paths - models are in Phase 5
        self.EMBEDDING_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 5', 'text-embedding-005', 'readmission_30')
        
        # Check for model files in different formats
        model_json = os.path.join(self.EMBEDDING_MODEL_DIR, f'model_{self.CHAMPION_ARM}.json')
        model_pkl = os.path.join(self.EMBEDDING_MODEL_DIR, f'model_{self.CHAMPION_ARM}.pkl')
        
        # Use JSON format as specified to avoid GPU compatibility issues
        if os.path.exists(model_json):
            self.CHAMPION_MODEL_PATH = model_json
            self.CHAMPION_MODEL_FORMAT = 'json'
        elif os.path.exists(model_pkl):
            self.CHAMPION_MODEL_PATH = model_pkl
            self.CHAMPION_MODEL_FORMAT = 'pkl'
        else:
            # Default to JSON if neither exists yet
            self.CHAMPION_MODEL_PATH = model_json
            self.CHAMPION_MODEL_FORMAT = 'json'
        
        # Try multiple possible locations for results file
        results_pkl = os.path.join(self.EMBEDDING_MODEL_DIR, f'results_{self.CHAMPION_ARM}.pkl')
        results_json = os.path.join(self.EMBEDDING_MODEL_DIR, f'results_{self.CHAMPION_ARM}.json')
        optuna_pkl = os.path.join(self.EMBEDDING_MODEL_DIR, f'optuna_study_{self.CHAMPION_ARM}.pkl')
        
        if os.path.exists(results_pkl):
            self.CHAMPION_RESULTS_PATH = results_pkl
        elif os.path.exists(results_json):
            self.CHAMPION_RESULTS_PATH = results_json
        elif os.path.exists(optuna_pkl):
            self.CHAMPION_RESULTS_PATH = optuna_pkl
        else:
            # Default to pkl if none exist
            self.CHAMPION_RESULTS_PATH = results_pkl
        
        # Embedding data paths - embeddings are in Phase 4
        self.EMBEDDING_DATA_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 4', 'embeddings_text-embedding-005', self.CHAMPION_ARM)
        
        # Label data paths
        self.LABEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 3', 'phase_3_serialized_data')
        
        # Output directory for H2 analysis (local to current directory)
        self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results_readmission_30')
        
        # --- Experiment Settings ---
        self.TARGET_VARIABLE = 'readmission_30'
        self.SEED = 42
        
        # Hyperparameter tuning settings
        self.N_OPTUNA_TRIALS = 20
        self.OPTUNA_TIMEOUT = 3600
        
        # Statistical testing settings
        self.CORRELATION_THRESHOLD = 0.4  # H2 threshold for "weakly correlated"
        self.N_BOOTSTRAP = 1000
        self.CONFIDENCE_LEVEL = 0.95
        
        # =============================================================================
        # SUBGROUP DISCOVERY SETTINGS
        # =============================================================================
        self.USE_SUBGROUP_DISCOVERY = True
        
        # Subgroup Discovery Algorithm Parameters
        self.SUBGROUP_MIN_SUPPORT = 0.05
        self.SUBGROUP_MAX_DEPTH = 3
        self.SUBGROUP_TOP_K = 10
        
        # Quality thresholds for H2b hypothesis evaluation
        self.SUBGROUP_MIN_QUALITY = 0.1
        self.SUBGROUP_MIN_COVERAGE_PCT = 5
        self.SUBGROUP_MIN_LIFT = 1.5
        
        # Number of analyses that must yield meaningful patterns to support H2b
        self.SUBGROUP_MIN_MEANINGFUL_ANALYSES = 2
        
        # Debugging/testing
        self.DRY_RUN = False
        self.DRY_RUN_SAMPLES = 1000
        
        # Create output directory
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
    def load_model_from_json(self, json_path):
        """Load model configuration from JSON file"""
        try:
            with open(json_path, 'r') as f:
                model_config = json.load(f)
            print(f"✅ Successfully loaded model config from: {json_path}")
            return model_config
        except Exception as e:
            print(f"❌ Error loading JSON model: {e}")
            return None
    
    def debug_file_structure(self):
        """Debug method to show actual file structure"""
        print("\n" + "="*80)
        print("🔍 DEBUGGING FILE STRUCTURE")
        print("="*80)
        
        # Check Phase 1-2 outputs
        print("\n📂 Phase 1-2 Output Structure:")
        if os.path.exists(self.BASELINE_MODEL_DIR):
            for item in sorted(os.listdir(self.BASELINE_MODEL_DIR)):
                item_path = os.path.join(self.BASELINE_MODEL_DIR, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    # Show first few files in subdirectory
                    try:
                        sub_items = os.listdir(item_path)[:5]
                        for sub_item in sub_items:
                            print(f"     - {sub_item}")
                    except:
                        pass
                else:
                    if 'readmission_30' in item or 'calibrated' in item:
                        print(f"  📄 {item} ✓")
        
        # Check Phase 4 embeddings
        print("\n📂 Phase 4 Embeddings Structure:")
        phase4_dir = os.path.join(self.NOTEBOOKS_DIR, 'Phase 4')
        if os.path.exists(phase4_dir):
            for item in os.listdir(phase4_dir):
                if 'embedding' in item.lower():
                    print(f"  📁 {item}/")
                    embed_path = os.path.join(phase4_dir, item)
                    if os.path.isdir(embed_path):
                        for model_dir in os.listdir(embed_path)[:3]:
                            print(f"     📁 {model_dir}/")
        
        # Check Phase 5 models
        print("\n📂 Phase 5 Model Structure:")
        phase5_dir = os.path.join(self.NOTEBOOKS_DIR, 'Phase 5')
        if os.path.exists(phase5_dir):
            for item in os.listdir(phase5_dir):
                if 'embedding' in item.lower():
                    print(f"  📁 {item}/")
                    embed_path = os.path.join(phase5_dir, item)
                    if os.path.isdir(embed_path):
                        for sub_dir in os.listdir(embed_path):
                            sub_path = os.path.join(embed_path, sub_dir)
                            if os.path.isdir(sub_path):
                                print(f"     📁 {sub_dir}/")
                                # Show model files
                                for file in os.listdir(sub_path)[:5]:
                                    if 'model' in file or 'optuna' in file or 'results' in file:
                                        print(f"        - {file}")
        
        print("\n" + "="*80)
    
    def validate_paths(self):
        """Validate that all required files exist"""
        print("\n" + "="*80)
        print("🔍 VALIDATING FILE PATHS")
        print("="*80)
        
        # First run debug to understand structure
        self.debug_file_structure()
        
        # Define critical vs optional files
        critical_files = {
            'Baseline Model': self.BASELINE_MODEL_PATH,
            'Champion Model (JSON)': self.CHAMPION_MODEL_PATH,
            'X_train': self.X_TRAIN_NUM_PATH,
            'X_val': self.X_VAL_NUM_PATH,
            'X_test': self.X_TEST_NUM_PATH,
            'y_train': self.Y_TRAIN_PATH,
            'y_val': self.Y_VAL_PATH,
            'y_test': self.Y_TEST_PATH,
        }
        
        optional_files = {
            'Baseline Results': self.BASELINE_RESULTS_PATH,
            'Champion Results': self.CHAMPION_RESULTS_PATH,
        }
        
        missing_critical = []
        missing_optional = []
        
        print("\n📋 Critical Files:")
        for name, path in critical_files.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"  ✅ {name}: {os.path.basename(path)} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ {name}: NOT FOUND")
                print(f"     Expected at: {path}")
                missing_critical.append(name)
        
        print("\n📋 Optional Files:")
        for name, path in optional_files.items():
            if os.path.exists(path):
                print(f"  ✅ {name}: {os.path.basename(path)}")
            else:
                print(f"  ⚠️  {name}: NOT FOUND (will generate if needed)")
                missing_optional.append(name)
        
        # Check embedding directories
        print("\n📁 Embedding Directories:")
        for split in ['train', 'val', 'test']:
            embed_dir = os.path.join(self.EMBEDDING_DATA_DIR, split)
            if os.path.isdir(embed_dir):
                num_files = len([f for f in os.listdir(embed_dir) if f.endswith('.npy')])
                print(f"  ✅ {split}: {num_files} embeddings")
            else:
                print(f"  ❌ {split}: NOT FOUND at {embed_dir}")
                missing_critical.append(f"Embedding dir: {split}")
        
        # Check label files
        print("\n📋 Label Files:")
        for split in ['train', 'val', 'test']:
            label_path = os.path.join(self.LABEL_DIR, f'{split}_labels.csv')
            if os.path.exists(label_path):
                print(f"  ✅ {split}_labels.csv")
            else:
                print(f"  ❌ {split}_labels.csv NOT FOUND")
                missing_critical.append(f"Label file: {split}")
        
        # Summary
        print("\n" + "="*80)
        if missing_critical:
            print(f"❌ CRITICAL: {len(missing_critical)} required files/directories missing:")
            for item in missing_critical:
                print(f"   - {item}")
            print("\nThe analysis cannot proceed without these files.")
            return False
        else:
            print("✅ All critical files found!")
            if missing_optional:
                print(f"⚠️  {len(missing_optional)} optional files missing (can be generated)")
            return True
    
    def check_data_consistency(self):
        """Check data consistency across files"""
        print("\n" + "="*80)
        print("🔍 CHECKING DATA CONSISTENCY")
        print("="*80)
        
        try:
            # Load a sample of each dataset to check shapes
            import pickle
            
            if os.path.exists(self.X_TRAIN_NUM_PATH):
                with open(self.X_TRAIN_NUM_PATH, 'rb') as f:
                    X_train = pickle.load(f)
                print(f"X_train shape: {X_train.shape}")
                
            if os.path.exists(self.Y_TRAIN_PATH):
                with open(self.Y_TRAIN_PATH, 'rb') as f:
                    y_train = pickle.load(f)
                print(f"y_train shape: {y_train.shape}")
                
                # Check target variable distribution
                if self.TARGET_VARIABLE in y_train.columns:
                    print(f"\n{self.TARGET_VARIABLE} distribution:")
                    print(y_train[self.TARGET_VARIABLE].value_counts())
                else:
                    print(f"⚠️  WARNING: {self.TARGET_VARIABLE} not found in y_train columns!")
                    print(f"Available columns: {y_train.columns.tolist()}")
            
            # Check model JSON structure
            if os.path.exists(self.CHAMPION_MODEL_PATH):
                model_config = self.load_model_from_json(self.CHAMPION_MODEL_PATH)
                if model_config:
                    print(f"\n📊 Model configuration loaded:")
                    print(f"   Model type: {model_config.get('model_type', 'Unknown')}")
                    if 'params' in model_config:
                        print(f"   Parameters: {len(model_config['params'])} params defined")
            
        except Exception as e:
            print(f"❌ Error during consistency check: {e}")
            import traceback
            traceback.print_exc()
    
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
            return True
            
        except ImportError:
            print("\n⚠️  pysubgroup not installed - will fall back to univariate analysis")
            print("   To enable: pip install pysubgroup")
            return False
    
    def run_all_checks(self):
        """Run all validation checks"""
        print("\n" + "="*80)
        print("🚀 RUNNING COMPLETE CONFIGURATION CHECK")
        print("="*80)
        
        # Check paths exist
        paths_valid = self.validate_paths()
        
        if paths_valid:
            # Check data consistency
            self.check_data_consistency()
            
            # Check subgroup discovery
            self.validate_subgroup_discovery()
            
            print("\n" + "="*80)
            print("✅ Configuration check complete!")
            print("="*80)
            return True
        else:
            print("\n" + "="*80)
            print("❌ Configuration check failed - please fix missing files")
            print("="*80)
            return False


# Test the configuration when this file is run directly
if __name__ == "__main__":
    config = ConfigH2()
    config.run_all_checks()