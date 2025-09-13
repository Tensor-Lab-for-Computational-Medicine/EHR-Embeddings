"""
Minimal configuration for H2a Analysis (Length-of-Stay > 3 days).
Based on champion model from Figure 5: F3_P1 (text-embedding-004).
"""
import os


class ConfigH2:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.NOTEBOOKS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.ROOT_DIR = os.path.abspath(os.path.join(self.NOTEBOOKS_DIR, '..'))

        # Numerical features and labels (Phase 1-2 outputs)
        num_dir = os.path.join(self.NOTEBOOKS_DIR, 'Phase 1 and 2', 'phase_1_outputs')
        # Fixed filenames (overwritten by preprocessing)
        self.X_TEST_NUM_PATH = os.path.join(num_dir, 'X_test.pkl')
        self.Y_TEST_PATH = os.path.join(num_dir, 'y_test.pkl')
        self.ICUSTAY_IDS_TEST_PATH = os.path.join(num_dir, 'icustay_ids_test.pkl')
        # Validation split (for threshold selection)
        self.X_VAL_NUM_PATH = os.path.join(num_dir, 'X_val.pkl')
        self.Y_VAL_PATH = os.path.join(num_dir, 'y_val.pkl')
        self.ICUSTAY_IDS_VAL_PATH = os.path.join(num_dir, 'icustay_ids_val.pkl')

        # Baseline (numeric) model
        self.BASELINE_MODEL_PATH = os.path.join(num_dir, 'los_3', 'model_1_xgboost_baseline_calibrated.pkl')

        # Champion embedding model and data (text-embedding-004)
        self.CHAMPION_ARM = 'F3_P1'
        self.CHAMPION_MODEL_PATH = os.path.join(self.NOTEBOOKS_DIR, 'Phase 5', 'embedding_model_results', 'text-embedding-004', 'los_3', f'model_{self.CHAMPION_ARM}_calibrated.pkl')
        # Phase 4 embeddings directory
        self.EMBEDDING_DATA_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 4', 'phase_4_embeddings', self.CHAMPION_ARM)

        # Labels
        self.LABEL_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 3', 'phase_3_serialized_data')

        # Outputs
        self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results_los_3')
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Core settings required by runner
        self.TARGET_VARIABLE = 'los_3'
        self.ID_COLUMN = 'icustay_id'
        self.N_BOOTSTRAP = 1000
        # Thresholding
        # Use 'youden' (default), 'f1', 'prevalence', or 'fixed' (uses THRESHOLD)
        self.THRESHOLD_STRATEGY = 'youden'
        self.THRESHOLD = 0.5


