import os
import pickle
import logging
import sys
import pandas as pd
import numpy as np

# Add current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xgboost_analysis import Config, load_preprocessed_data, _clean_features_and_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def patch_specific_numeric(target, model_type):
    logging.info(f"Patching Numeric Champion: {target} | {model_type}")
    
    config = Config({'TARGET_VARIABLE': target})
    
    # Identify files based on model type
    if model_type == 'XGBoost':
        results_filename = 'results_xgboost_baseline.pkl'
        model_filename = 'model_1_xgboost_baseline_calibrated.pkl'
        backup_model_filename = 'model_1_xgboost_baseline.pkl'
    elif model_type == 'ElasticNet':
        results_filename = 'results_elastic_net_baseline.pkl'
        model_filename = 'model_2_elastic_net_baseline.pkl'
        backup_model_filename = None
    else:
        logging.warning(f"Unknown model type: {model_type}")
        return

    results_path = os.path.join(config.OUTPUT_DIR, results_filename)
    
    if not os.path.exists(results_path):
        logging.warning(f"Results file not found: {results_path}")
        return

    # Check if patch needed
    with open(results_path, 'rb') as f:
        res = pickle.load(f)
        
    if 'y_true' in res and 'y_pred_proba' in res:
        logging.info("-> Already patched.")
        return

    # Load Data (Only test set needed, but load_preprocessed_data loads all)
    # We accept the overhead here as loading pickles is fast compared to embedding data
    try:
        _, _, X_test, _, _, y_test, _, _ = load_preprocessed_data(config)
        X_test_clean, y_test_clean = _clean_features_and_labels(X_test, y_test, expected_binary=True)
    except Exception as e:
        logging.warning(f"Could not load data for {target}: {e}")
        return

    # Load Model
    model_path = os.path.join(config.OUTPUT_DIR, model_filename)
    if not os.path.exists(model_path) and backup_model_filename:
        model_path = os.path.join(config.OUTPUT_DIR, backup_model_filename)
    
    if not os.path.exists(model_path):
        logging.warning(f"Model file not found: {model_path}")
        return
        
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        y_pred = model.predict_proba(X_test_clean)[:, 1]
        
        # Update results
        res['y_true'] = y_test_clean.values
        res['y_pred_proba'] = y_pred
        
        with open(results_path, 'wb') as f:
            pickle.dump(res, f)
        logging.info("-> Successfully patched!")
        
    except Exception as e:
        logging.error(f"Error patching {target} ({model_type}): {e}")

def main():
    # List of numeric champions to patch
    # (Target, ModelType)
    champions = [
        ('readmission_30',    'ElasticNet'),
        ('mort_hosp',         'XGBoost'),
        ('los_3',             'XGBoost'),
        ('los_7',             'ElasticNet'),
        ('intervention_vent', 'XGBoost'),
        ('intervention_vaso', 'ElasticNet'),
    ]

    for target, model_type in champions:
        patch_specific_numeric(target, model_type)

if __name__ == "__main__":
    main()
