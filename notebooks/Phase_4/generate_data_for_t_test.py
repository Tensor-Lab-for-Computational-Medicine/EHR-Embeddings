import os
import pickle
import logging
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import importlib.util

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_config_from_file(filepath):
    spec = importlib.util.spec_from_file_location("dynamic_config", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config

def load_test_data_only(config, exp_arm):
    y_test_path = os.path.join(config.LABEL_DIR, f'{config.TARGET_VARIABLE}_test_labels.csv')
    if not os.path.exists(y_test_path):
        logging.warning(f"Label file missing: {y_test_path}")
        return None, None
        
    y_test_df = pd.read_csv(y_test_path, header=0, index_col=0)
    
    # Try standard 'test' subdir
    embedding_dir = os.path.join(config.EMBEDDING_DIR, exp_arm, 'test')
    if not os.path.exists(embedding_dir):
        # Fallback: maybe it's flat?
        embedding_dir = os.path.join(config.EMBEDDING_DIR, exp_arm)
    
    if not os.path.exists(embedding_dir):
        logging.warning(f"Embedding dir missing: {embedding_dir}")
        return None, None

    embedding_vectors = []
    valid_indices = []
    
    for icustay_id in tqdm(y_test_df.index, desc=f"Loading {exp_arm}", leave=False):
        filepath = os.path.join(embedding_dir, f"{icustay_id}.npy")
        if os.path.exists(filepath):
            embedding_vectors.append(np.load(filepath))
            valid_indices.append(icustay_id)
            
    if not embedding_vectors:
        return None, None
        
    X_test = np.vstack(embedding_vectors)
    y_test_df = y_test_df.loc[valid_indices]
    
    return X_test, y_test_df

def patch_specific_champion(ConfigClass, config_filename, target, arm):
    logging.info(f"Patching Champion: {target} | {arm} | {config_filename}")
    
    try:
        # Instantiate config specifically for this target
        try:
            config = ConfigClass(target_variable=target)
        except TypeError:
            config = ConfigClass()
            config.TARGET_VARIABLE = target
            
            # FIX: Force update OUTPUT_DIR if it was hardcoded in class
            # Assume structure is .../model_name/default_target
            # We want .../model_name/new_target
            if hasattr(config, 'OUTPUT_DIR'):
                # Check if OUTPUT_DIR ends with a known target variable
                # If so, replace it. If not, append target?
                # Safer: use os.path.dirname which strips the last component (e.g. mort_hosp)
                # and join with new target
                base_output = os.path.dirname(config.OUTPUT_DIR)
                config.OUTPUT_DIR = os.path.join(base_output, target)
            
            if hasattr(config, 'BASE_OUTPUT_DIR'):
                config.OUTPUT_DIR = os.path.join(config.BASE_OUTPUT_DIR, target)
        
        output_dir = getattr(config, 'OUTPUT_DIR', None)
        if not output_dir or not os.path.exists(output_dir):
            logging.warning(f"Output dir not found for {target}: {output_dir}")
            return

        # Find result file for this specific arm
        filename = f"results_{arm}.pkl"
        results_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(results_path):
            logging.warning(f"Result file not found: {results_path}")
            return
            
        model_cal_path = os.path.join(output_dir, f'model_{arm}_calibrated.pkl')
        model_path = os.path.join(output_dir, f'model_{arm}.pkl')
        active_model_path = model_cal_path if os.path.exists(model_cal_path) else model_path
        
        if not os.path.exists(active_model_path):
            logging.warning(f"Model file not found: {active_model_path}")
            return
            
        # Check if patch needed (FORCE PATCH NOW DEBUGGING)
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        # if 'y_true' in results and 'y_pred_proba' in results:
        #     logging.info(f"-> Already patched: {results_path}")
        #     return

        logging.info(f"-> Patching file: {results_path}")

        # Load Data
        X_test, y_test_df = load_test_data_only(config, arm)
        if X_test is None:
            logging.warning("-> Failed to load test data.")
            return

        y_test = y_test_df[config.TARGET_VARIABLE]
        
        # Filter prevalent cases
        if target in ['intervention_vent', 'intervention_vaso']:
            valid_mask = y_test.notna()
            y_test = y_test[valid_mask]
            X_test = X_test[valid_mask]
            
        # Predict
        with open(active_model_path, 'rb') as f:
            model = pickle.load(f)
            
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results['y_true'] = y_test.values
        results['y_pred_proba'] = y_pred_proba
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
            
        logging.info("-> Successfully patched!")

    except Exception as e:
        logging.error(f"Error patching {arm}: {e}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of champions to patch
    # Format: (Target, Arm, ConfigFileName)
    # Extracted from your PDF table
    champions = [
        ('readmission_30',    'F1_P0', 'config_embedding_analysis_text_embedding_005.py'),
        ('mort_hosp',         'F3_P5', 'config_embedding_analysis_text_embedding_004.py'),
        ('los_3',             'F3_P1', 'config_embedding_analysis_text_embedding_004.py'),
        ('los_7',             'F3_P2', 'config_embedding_analysis_text_embedding_005.py'),
        ('intervention_vent', 'F3_P0', 'config_embedding_analysis_text_embedding_004.py'),
        ('intervention_vaso', 'F3_P2', 'config_embedding_analysis_text_embedding_004.py'),
    ]

    for target, arm, config_file in champions:
        full_path = os.path.join(current_dir, config_file)
        if not os.path.exists(full_path):
            logging.warning(f"Config file not found: {config_file}")
            continue
            
        try:
            ConfigClass = load_config_from_file(full_path)
            patch_specific_champion(ConfigClass, config_file, target, arm)
        except Exception as e:
            logging.error(f"Failed to load config {config_file}: {e}")

if __name__ == "__main__":
    main()
