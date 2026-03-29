"""
Minimal configuration for H2a Analysis (30-day readmission).
Only essential paths and settings required by the runner are defined.
"""
import os


class ConfigH2:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.NOTEBOOKS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.ROOT_DIR = os.path.abspath(os.path.join(self.NOTEBOOKS_DIR, '..'))
        self.TASK_NAME = 'Readmission'

        # Numerical features and labels (Phase 1-2 outputs)
        num_dir = os.path.join(self.NOTEBOOKS_DIR, 'Phase 1 and 2', 'phase_1_outputs')
        # Fixed filenames (overwritten by preprocessing)
        self.X_TRAIN_NUM_PATH = os.path.join(num_dir, 'X_train.pkl')
        self.X_TEST_NUM_PATH = os.path.join(num_dir, 'X_test.pkl')
        self.Y_TRAIN_PATH = os.path.join(num_dir, 'y_train.pkl')
        self.Y_TEST_PATH = os.path.join(num_dir, 'y_test.pkl')
        self.ICUSTAY_IDS_TRAIN_PATH = os.path.join(num_dir, 'icustay_ids_train.pkl')
        self.ICUSTAY_IDS_TEST_PATH = os.path.join(num_dir, 'icustay_ids_test.pkl')
        # Validation split (for threshold selection)
        self.X_VAL_NUM_PATH = os.path.join(num_dir, 'X_val.pkl')
        self.Y_VAL_PATH = os.path.join(num_dir, 'y_val.pkl')
        self.ICUSTAY_IDS_VAL_PATH = os.path.join(num_dir, 'icustay_ids_val.pkl')

        # Baseline (numeric) model
        self.BASELINE_MODEL_PATH = os.path.join(num_dir, 'readmission_30', 'model_1_xgboost_baseline_calibrated.pkl')

        # Embedding model and data (text-embedding-005, prefer calibrated models)
        self.CHAMPION_ARM = 'F1_P0'
        # Prefer calibrated models from embedding_model_results
        self.EMBEDDING_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 5', 'embedding_model_results', 'text-embedding-005', 'readmission_30')
        self._LEGACY_MODEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 5', 'text-embedding-005', 'readmission_30')
        # Resolution order: calibrated.pkl (results dir) -> json (legacy dir) -> pkl (legacy dir)
        candidate_paths = [
            os.path.join(self.EMBEDDING_MODEL_DIR, f'model_{self.CHAMPION_ARM}_calibrated.pkl'),
            os.path.join(self._LEGACY_MODEL_DIR, f'model_{self.CHAMPION_ARM}.json'),
            os.path.join(self._LEGACY_MODEL_DIR, f'model_{self.CHAMPION_ARM}.pkl'),
        ]
        self.CHAMPION_MODEL_PATH = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
        # Phase 4 embeddings directory
        self.EMBEDDING_DATA_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 4', 'embeddings_text-embedding-005', self.CHAMPION_ARM)

        # Labels
        self.LABEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 3', 'phase_3_serialized_data')

        # Outputs
        self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results_readmission_30')
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Core settings required by runner
        self.TARGET_VARIABLE = 'readmission_30'
        self.N_BOOTSTRAP = 1000
        # Thresholding
        # Strategy: 'youden' (tune) or 'fixed' (use THRESHOLD)
        self.THRESHOLD_STRATEGY = 'youden'
        # Objective when tuning: 'youden' | 'f1' | 'fbeta' | 'accuracy' | 'balanced_accuracy'
        self.THRESHOLD_OBJECTIVE = 'youden'
        # Beta for fbeta (only used if THRESHOLD_OBJECTIVE='fbeta')
        self.THRESHOLD_BETA = 1.0
        self.THRESHOLD = 0.5