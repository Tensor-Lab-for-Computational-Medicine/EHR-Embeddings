"""
Minimal configuration for H2a Analysis (mortality in hospital).
Only essential paths and settings required by the runner are defined.
"""
import os


class ConfigH2:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.NOTEBOOKS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.ROOT_DIR = os.path.abspath(os.path.join(self.NOTEBOOKS_DIR, '..'))
        self.TASK_NAME = 'Mortality'

        # Numerical features and labels (Phase 1-2 outputs)
        num_dir = os.path.join(self.NOTEBOOKS_DIR, 'Phase_1-2', 'phase_1_outputs')
        # Fixed filenames (overwritten by preprocessing)
        self.X_TRAIN_NUM_PATH = os.path.join(num_dir, 'X_train.pkl')
        self.X_TEST_NUM_PATH = os.path.join(num_dir, 'X_test.pkl')
        self.Y_TRAIN_PATH = os.path.join(num_dir, 'y_train.pkl')
        self.Y_TEST_PATH = os.path.join(num_dir, 'y_test.pkl')
        self.ICUSTAY_IDS_TRAIN_PATH = os.path.join(num_dir, 'icustay_ids_train.pkl')
        self.ICUSTAY_IDS_TEST_PATH = os.path.join(num_dir, 'icustay_ids_test.pkl')
        # Validation (for threshold selection)
        self.X_VAL_NUM_PATH = os.path.join(num_dir, 'X_val.pkl')
        self.Y_VAL_PATH = os.path.join(num_dir, 'y_val.pkl')
        self.ICUSTAY_IDS_VAL_PATH = os.path.join(num_dir, 'icustay_ids_val.pkl')

        # Baseline (numeric) model
        self.BASELINE_MODEL_PATH = os.path.join(self.NOTEBOOKS_DIR, 'Phase_1-2', 'phase_1_outputs', 'mort_hosp', 'model_1_xgboost_baseline_calibrated.pkl')

        # Embedding model and data
        self.CHAMPION_ARM = 'F3_P5'  # text-embedding-004
        self.CHAMPION_MODEL_PATH = os.path.join(self.NOTEBOOKS_DIR, 'Phase_5', 'embedding_model_results', 'text-embedding-004', 'mort_hosp', f'model_{self.CHAMPION_ARM}_calibrated.pkl')
        self.EMBEDDING_DATA_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase_4', 'phase_4_embeddings', self.CHAMPION_ARM)

        # Labels
        self.LABEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase_3', 'phase_3_serialized_data', 'mort_hosp')

        # Outputs
        self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results', 'mort_hosp')
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Core settings required by runner
        self.TARGET_VARIABLE = 'mort_hosp'
        self.ID_COLUMN = 'icustay_id'
        self.N_BOOTSTRAP = 1000
        # Thresholding
        # Strategy: 'youden' (tune) or 'fixed' (use THRESHOLD)
        self.THRESHOLD_STRATEGY = 'youden'
        # Objective when tuning: 'youden' | 'f1' | 'fbeta' | 'accuracy' | 'balanced_accuracy'
        self.THRESHOLD_OBJECTIVE = 'youden'
        # Beta for fbeta (only used if THRESHOLD_OBJECTIVE='fbeta')
        self.THRESHOLD_BETA = 1.0
        self.THRESHOLD = 0.5
