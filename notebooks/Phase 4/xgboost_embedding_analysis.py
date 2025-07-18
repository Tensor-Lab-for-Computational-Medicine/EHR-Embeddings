# xgboost_embedding_analysis.py
"""
Trains and evaluates XGBoost models on text embedding data for all 18
experimental conditions (F1/F2/F3 x P0-P5).
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import logging
import time
import os
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from tqdm import tqdm
from config_embedding_analysis_text_embedding_004 import Config

# =============================================================================
# DATA HANDLING
# =============================================================================

def load_embedding_data(config, exp_arm):
    """Loads embedding (.npy) and label (.csv) data for a given experimental arm."""
    logging.info(f"Loading data for experimental arm: {exp_arm}...")
    
    # Load the label files specific to the target variable, using the first row as the header.
    y_train_df = pd.read_csv(os.path.join(config.LABEL_DIR, f'{config.TARGET_VARIABLE}_train_labels.csv'), header=0, index_col=0)
    y_val_df = pd.read_csv(os.path.join(config.LABEL_DIR, f'{config.TARGET_VARIABLE}_val_labels.csv'), header=0, index_col=0)
    y_test_df = pd.read_csv(os.path.join(config.LABEL_DIR, f'{config.TARGET_VARIABLE}_test_labels.csv'), header=0, index_col=0)

    if config.DRY_RUN:
        logging.warning(f"DRY RUN: Subsetting data to {config.DRY_RUN_SUBSET_SIZE} stratified samples per split.")
        # Stratified sampling for the specific target variable
        y_train_series = y_train_df[config.TARGET_VARIABLE]
        y_train_df = y_train_df.loc[y_train_series.groupby(y_train_series).head(config.DRY_RUN_SUBSET_SIZE // 2).sort_index().index]

        y_val_series = y_val_df[config.TARGET_VARIABLE]
        y_val_df = y_val_df.loc[y_val_series.groupby(y_val_series).head(config.DRY_RUN_SUBSET_SIZE // 2).sort_index().index]

        y_test_series = y_test_df[config.TARGET_VARIABLE]
        y_test_df = y_test_df.loc[y_test_series.groupby(y_test_series).head(config.DRY_RUN_SUBSET_SIZE // 2).sort_index().index]
    
    X_data = {}
    for split, y_df_split in [('train', y_train_df), ('val', y_val_df), ('test', y_test_df)]:
        logging.info(f"Loading {split} embeddings for {len(y_df_split)} samples...")
        embedding_vectors = []
        embedding_dir = os.path.join(config.EMBEDDING_DIR, exp_arm, split)
        
        for icustay_id in tqdm(y_df_split.index, desc=f"Loading {split} splits"):
            filepath = os.path.join(embedding_dir, f"{icustay_id}.npy")
            try:
                embedding_vectors.append(np.load(filepath))
            except FileNotFoundError:
                logging.error(f"Embedding file not found for icustay_id {icustay_id} in {split} set. Exiting.")
                raise
        
        X_data[f'X_{split}'] = np.vstack(embedding_vectors)

    logging.info(f"Data shapes: X_train={X_data['X_train'].shape}, X_val={X_data['X_val'].shape}, X_test={X_data['X_test'].shape}")
    
    # Return the full dataframes for y
    return X_data['X_train'], X_data['X_val'], X_data['X_test'], y_train_df, y_val_df, y_test_df

# =============================================================================
# EVALUATION (Reused from original script)
# =============================================================================

def bootstrap_metric(y_true, y_pred_proba, metric_func, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    np.random.seed(random_state)
    scores = [
        metric_func(y_true[indices], y_pred_proba[indices])
        for indices in [np.random.choice(len(y_true), len(y_true), replace=True) for _ in range(n_bootstrap)]
        if len(np.unique(y_true[indices])) > 1
    ]
    alpha = 1 - confidence_level
    ci_lower = np.percentile(scores, (alpha/2) * 100)
    ci_upper = np.percentile(scores, (1 - alpha/2) * 100)
    return {'point_estimate': metric_func(y_true, y_pred_proba), 'ci_lower': ci_lower, 'ci_upper': ci_upper}

def evaluate_with_uncertainty(y_true, y_pred_proba, y_pred=None):
    if y_pred is None: y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        'auroc': bootstrap_metric(y_true, y_pred_proba, roc_auc_score),
        'auprc': bootstrap_metric(y_true, y_pred_proba, average_precision_score),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }

# =============================================================================
# MODEL TRAINING (Adapted for multiple arms)
# =============================================================================

def tune_xgboost(X_train, y_train, X_val, y_val, config, exp_arm):
    """Tune XGBoost hyperparameters using Optuna for a specific experimental arm."""
    study_path = os.path.join(config.OUTPUT_DIR, f'optuna_study_{exp_arm}.pkl')
    study = None  # Initialize study to None

    if os.path.exists(study_path) and config.REUSE_EXISTING_STUDY:
        logging.info(f"Loading existing Optuna study: {study_path}")
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        
        if len(study.trials) >= config.N_OPTUNA_TRIALS:
            logging.info(f"Study already has {len(study.trials)} trials (>= {config.N_OPTUNA_TRIALS}). Reusing best params.")
            return study.best_params
        else:
            logging.info(f"Study has {len(study.trials)} trials, but {config.N_OPTUNA_TRIALS} are required. Continuing study.")

    if study is None:
        logging.info(f"Creating new Optuna study for {exp_arm}...")
        study = optuna.create_study(direction='maximize')
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'n_estimators': 500,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'scale_pos_weight': scale_pos_weight, 'random_state': config.SEED, 'n_jobs': -1,
            'tree_method': 'gpu_hist' if config.USE_GPU else 'auto',
            'predictor': 'gpu_predictor' if config.USE_GPU else 'auto'
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', early_stopping_rounds=30, verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    n_trials_to_run = config.N_OPTUNA_TRIALS - len(study.trials)
    if n_trials_to_run > 0:
        study.optimize(objective, n_trials=n_trials_to_run, timeout=config.OPTUNA_TIMEOUT)
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        logging.info(f"Study saved to {study_path}")
    
    return study.best_params

def train_final_model(X_train, X_val, y_train, y_val, best_params, config):
    logging.info("Training final model on combined train+validation data...")
    X_full = np.vstack([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    final_params = best_params.copy()
    final_params['scale_pos_weight'] = (y_full == 0).sum() / (y_full == 1).sum() if (y_full == 1).sum() > 0 else 1
    if config.USE_GPU:
        final_params['tree_method'] = 'gpu_hist'
        final_params['predictor'] = 'gpu_predictor'
    model = xgb.XGBClassifier(**final_params, random_state=config.SEED, n_jobs=-1)
    model.fit(X_full, y_full, verbose=False)
    return model

def save_results(model, results, best_params, config, exp_arm):
    """Save model and results for a specific experimental arm."""
    model_path = os.path.join(config.OUTPUT_DIR, f'model_{exp_arm}.pkl')
    with open(model_path, 'wb') as f: pickle.dump(model, f)
    
    results_dict = {
        'experimental_arm': exp_arm,
        'model_name': f'XGBoost on Embedding ({exp_arm})',
        **results
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, f'results_{exp_arm}.pkl')
    with open(results_path, 'wb') as f: pickle.dump(results_dict, f)
    
    logging.info(f"Model for {exp_arm} saved to: {model_path}")
    logging.info(f"Results for {exp_arm} saved to: {results_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the analysis across all experimental arms."""
    config = Config()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'embedding_analysis_log.txt'), mode='w'),
                                  logging.StreamHandler()])
    
    start_time = time.time()
    
    # The script now runs for ONE target variable at a time, defined in the config.
    # To run for other targets, you must change `TARGET_VARIABLE` in the config file.
    logging.info(f"Starting analysis for TARGET_VARIABLE: {config.TARGET_VARIABLE}")

    experimental_arms = [f"{rep}_{p}" for rep in config.REPRESENTATIONS for p in config.PROMPTS]
    if config.DRY_RUN:
        logging.warning("DRY RUN ENABLED: Processing only the first experimental arm on a subset of data.")
        experimental_arms = [experimental_arms[0]]

    for arm in experimental_arms:
        logging.info(f"\n{'='*80}\nSTARTING ANALYSIS FOR ARM: {arm}\n{'='*80}")
        
        try:
            # Load data, y_... are now dataframes
            X_train, X_val, X_test, y_train_df, y_val_df, y_test_df = load_embedding_data(config, arm)
            
            # Select the specific target variable for this run
            y_train = y_train_df[config.TARGET_VARIABLE]
            y_val = y_val_df[config.TARGET_VARIABLE]
            y_test = y_test_df[config.TARGET_VARIABLE]

            # Handle prevalent cases for intervention targets by dropping NaNs
            if config.TARGET_VARIABLE in ['intervention_vent', 'intervention_vaso']:
                logging.info(f"Handling prevalent cases for target: {config.TARGET_VARIABLE}")
                
                # Align indices before dropping NaNs
                X_train_df = pd.DataFrame(X_train, index=y_train.index)
                X_val_df = pd.DataFrame(X_val, index=y_val.index)
                X_test_df = pd.DataFrame(X_test, index=y_test.index)

                # Train set
                train_original_count = len(y_train)
                train_valid_indices = y_train.dropna().index
                X_train_df = X_train_df.loc[train_valid_indices]
                y_train = y_train.loc[train_valid_indices]
                X_train = X_train_df.values
                logging.info(f"Train set: Dropped {train_original_count - len(y_train)} prevalent cases. New size: {len(y_train)}")
                
                # Validation set
                val_original_count = len(y_val)
                val_valid_indices = y_val.dropna().index
                X_val_df = X_val_df.loc[val_valid_indices]
                y_val = y_val.loc[val_valid_indices]
                X_val = X_val_df.values
                logging.info(f"Validation set: Dropped {val_original_count - len(y_val)} prevalent cases. New size: {len(y_val)}")
                
                # Test set
                test_original_count = len(y_test)
                test_valid_indices = y_test.dropna().index
                X_test_df = X_test_df.loc[test_valid_indices]
                y_test = y_test.loc[test_valid_indices]
                X_test = X_test_df.values
                logging.info(f"Test set: Dropped {test_original_count - len(y_test)} prevalent cases. New size: {len(y_test)}")

            best_params = tune_xgboost(X_train, y_train, X_val, y_val, config, arm)
            
            model = train_final_model(X_train, X_val, y_train, y_val, best_params, config)
            
            logging.info(f"--- FINAL EVALUATION ON TEST SET FOR {arm} ---")
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            results = evaluate_with_uncertainty(y_test.values, y_pred_proba)
            
            auroc = results['auroc']
            logging.info(f"Test AUROC for {arm}: {auroc['point_estimate']:.4f} (95% CI: {auroc['ci_lower']:.4f}-{auroc['ci_upper']:.4f})")
            
            save_results(model, results, best_params, config, arm)
        except Exception as e:
            logging.error(f"!!! An error occurred while processing arm {arm}: {e}")
            logging.error(f"Skipping arm {arm} and continuing to the next one.")
            continue

    logging.info(f"\nFull analysis completed in {(time.time() - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()