import os
import sys

# Change to the notebook directory
os.chdir(r"d:\Projects\EHR Embeddings")

from notebooks.Phase_1_and_2.xgboost_analysis import main

if __name__ == '__main__':
    main({
        'TARGET_VARIABLE': 'readmission_30',
        # Don't spend forever tuning if we just want a baseline
        'N_OPTUNA_TRIALS': 2,
        'REUSE_EXISTING_STUDY': True
    })
