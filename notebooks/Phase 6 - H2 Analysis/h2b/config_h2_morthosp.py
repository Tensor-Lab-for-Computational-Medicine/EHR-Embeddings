import os


class ConfigH2:
   def __init__(self):
       current_dir = os.path.dirname(os.path.abspath(__file__))
       self.NOTEBOOKS_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))
       # Outputs
       self.OUTPUT_DIR = os.path.join(current_dir, 'h2_results')
       self.H2A_OUTPUT_DIR = os.path.join(self.NOTEBOOKS_DIR, 'Phase 6 - H2 Analysis', 'h2a', 'h2_results')
       # Minimal inputs used by h2b
       self.X_TEST_NUM_PATH = os.path.join(self.NOTEBOOKS_DIR, 'Phase 1 and 2', 'phase_1_outputs', 'X_test.pkl')
       # Settings
       self.TARGET_VARIABLE = 'mort_hosp'
       # Subgroup discovery
       self.SUBGROUP_MIN_SUPPORT = 0.05
       self.SUBGROUP_MAX_DEPTH = 3
       self.SUBGROUP_TOP_K = 10
       self.SUBGROUP_MIN_QUALITY = 0.1
       self.SUBGROUP_MIN_LIFT = 1.5
       self.SUBGROUP_MIN_MEANINGFUL_ANALYSES = 2
       os.makedirs(self.OUTPUT_DIR, exist_ok=True)
