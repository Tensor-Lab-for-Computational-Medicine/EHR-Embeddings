import os


class ConfigH2:
   def __init__(self):
       current_dir = os.path.dirname(os.path.abspath(__file__))
       self.NOTEBOOKS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))
       # Outputs
       self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results')
       self.H2A_OUTPUT_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase_6', 'h2a', 'h2_results')
       # Minimal inputs used by h2b
       self.X_TEST_NUM_PATH = os.path.join(self.NOTEBOOKS_DIR, 'Phase_1-2', 'phase_1_outputs', 'X_test.pkl')
       self.SCALER_PATH = os.path.join(self.NOTEBOOKS_DIR, 'Phase_1-2', 'phase_1_outputs', 'scaler.pkl')
       self.PHENOTYPE_RULES_CSV = os.path.join(self.NOTEBOOKS_DIR, 'Phase_6', 'feature_engineering', 'feature_rules.csv')
       self.X_TEST_PHENOS_PATH = os.path.join(current_dir, '..', 'feature_engineering', 'artifacts', 'X_test_phenotypes.pkl')
       self.X_TRAINVAL_PHENOS_PATH = os.path.join(current_dir, '..', 'feature_engineering', 'artifacts', 'X_trainval_phenotypes.pkl')
       # Settings
       self.TARGET_VARIABLE = 'mort_hosp'
       # Subgroup discovery
       self.SUBGROUP_MIN_SUPPORT = 0.01
       self.SUBGROUP_MAX_DEPTH = 3
       self.SUBGROUP_MAX_CANDIDATES = 200
       self.SUBGROUP_MIN_QUALITY = 0.0
       self.SUBGROUP_MIN_LIFT = 0.0
       os.makedirs(self.OUTPUT_DIR, exist_ok=True)
