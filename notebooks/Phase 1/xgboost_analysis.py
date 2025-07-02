# xgboost_analysis.py

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

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

# Default configuration values (can be overridden by passing config dict)
DEFAULT_CONFIG = {
    'OUTPUT_DIR': 'phase_1_outputs',
    'DRY_RUN': True,
    'DRY_RUN_PATIENTS': 1000,
    'CALCULATE_TRENDS': True,
    'WINDOW_SIZE': 24,
    'GAP_TIME': 6,
    'TARGET_VARIABLE': 'mort_hosp',
    'SEED': 42,
    'N_OPTUNA_TRIALS': 15,
    'OPTUNA_TIMEOUT': 1800
}

# Global configuration variables (will be set by main function)
OUTPUT_DIR = None
DRY_RUN = None
DRY_RUN_PATIENTS = None
CALCULATE_TRENDS = None
WINDOW_SIZE = None
GAP_TIME = None
TARGET_VARIABLE = None
SEED = None
N_OPTUNA_TRIALS = None
OPTUNA_TIMEOUT = None

def set_config(config_dict=None):
    """Set global configuration from dictionary or use defaults."""
    global OUTPUT_DIR, DRY_RUN, DRY_RUN_PATIENTS, CALCULATE_TRENDS
    global WINDOW_SIZE, GAP_TIME, TARGET_VARIABLE, SEED, N_OPTUNA_TRIALS, OPTUNA_TIMEOUT
    
    # Use provided config or defaults
    config = config_dict if config_dict else DEFAULT_CONFIG
    
    # Set global variables
    OUTPUT_DIR = config.get('OUTPUT_DIR', DEFAULT_CONFIG['OUTPUT_DIR'])
    DRY_RUN = config.get('DRY_RUN', DEFAULT_CONFIG['DRY_RUN'])
    DRY_RUN_PATIENTS = config.get('DRY_RUN_PATIENTS', DEFAULT_CONFIG['DRY_RUN_PATIENTS'])
    CALCULATE_TRENDS = config.get('CALCULATE_TRENDS', DEFAULT_CONFIG['CALCULATE_TRENDS'])
    WINDOW_SIZE = config.get('WINDOW_SIZE', DEFAULT_CONFIG['WINDOW_SIZE'])
    GAP_TIME = config.get('GAP_TIME', DEFAULT_CONFIG['GAP_TIME'])
    TARGET_VARIABLE = config.get('TARGET_VARIABLE', DEFAULT_CONFIG['TARGET_VARIABLE'])
    SEED = config.get('SEED', DEFAULT_CONFIG['SEED'])
    N_OPTUNA_TRIALS = config.get('N_OPTUNA_TRIALS', DEFAULT_CONFIG['N_OPTUNA_TRIALS'])
    OPTUNA_TIMEOUT = config.get('OPTUNA_TIMEOUT', DEFAULT_CONFIG['OPTUNA_TIMEOUT'])
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'xgboost_analysis_log.txt'), mode='w'),
        logging.StreamHandler()
    ]
)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def get_cache_filename_prefix():
    """Generate the same prefix used in preprocessing to load cached data."""
    prefix = f"preprocessed_{TARGET_VARIABLE}"
    if DRY_RUN:
        prefix += f"_dryrun_{DRY_RUN_PATIENTS}"
    prefix += f"_trends_{CALCULATE_TRENDS}"
    prefix += f"_window_{WINDOW_SIZE}_gap_{GAP_TIME}"
    prefix += f"_seed_{SEED}"
    return prefix

def load_preprocessed_data():
    """Load preprocessed data from cache files."""
    prefix = get_cache_filename_prefix()
    
    cache_files = {
        'X_train': os.path.join(OUTPUT_DIR, f'{prefix}_X_train.pkl'),
        'X_val': os.path.join(OUTPUT_DIR, f'{prefix}_X_val.pkl'),
        'X_test': os.path.join(OUTPUT_DIR, f'{prefix}_X_test.pkl'),
        'y_train': os.path.join(OUTPUT_DIR, f'{prefix}_y_train.pkl'),
        'y_val': os.path.join(OUTPUT_DIR, f'{prefix}_y_val.pkl'),
        'y_test': os.path.join(OUTPUT_DIR, f'{prefix}_y_test.pkl'),
        'scaler': os.path.join(OUTPUT_DIR, f'{prefix}_scaler.pkl'),
        'imputation_values': os.path.join(OUTPUT_DIR, f'{prefix}_imputation_values.pkl')
    }
    
    # Check if all cache files exist
    missing_files = [f for f in cache_files.values() if not os.path.exists(f)]
    if missing_files:
        logging.error(f"Missing preprocessed data files:")
        for f in missing_files:
            logging.error(f"  - {f}")
        raise FileNotFoundError("Preprocessed data not found. Please run data_preprocessing.py first.")
    
    logging.info("Loading preprocessed data from cache...")
    try:
        with open(cache_files['X_train'], 'rb') as f:
            X_train = pickle.load(f)
        with open(cache_files['X_val'], 'rb') as f:
            X_val = pickle.load(f)
        with open(cache_files['X_test'], 'rb') as f:
            X_test = pickle.load(f)
        with open(cache_files['y_train'], 'rb') as f:
            y_train = pickle.load(f)
        with open(cache_files['y_val'], 'rb') as f:
            y_val = pickle.load(f)
        with open(cache_files['y_test'], 'rb') as f:
            y_test = pickle.load(f)
        with open(cache_files['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        with open(cache_files['imputation_values'], 'rb') as f:
            imputation_values = pickle.load(f)
        
        logging.info(f"✓ Successfully loaded preprocessed data with prefix: {prefix}")
        logging.info(f"Data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputation_values
    
    except Exception as e:
        logging.error(f"Failed to load preprocessed data: {e}")
        raise

# =============================================================================
# DATA CLEANING FOR XGBOOST
# =============================================================================

def clean_data_for_xgboost(X_train, X_val, X_test):
    """Data cleaning for XGBoost that preserves NaN values for missingness learning."""
    logging.info("Performing additional data cleaning for XGBoost...")
    
    # Check for problematic values
    logging.info(f"Before cleaning - NaN counts:")
    logging.info(f"  X_train: {X_train.isna().sum().sum()}")
    logging.info(f"  X_val: {X_val.isna().sum().sum()}")
    logging.info(f"  X_test: {X_test.isna().sum().sum()}")
    
    logging.info(f"Before cleaning - Inf counts:")
    logging.info(f"  X_train: {np.isinf(X_train).sum().sum()}")
    logging.info(f"  X_val: {np.isinf(X_val).sum().sum()}")
    logging.info(f"  X_test: {np.isinf(X_test).sum().sum()}")
    
    # Replace infinite values with NaN first, then handle all NaNs consistently
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # XGBoost can handle NaN values natively - preserve them for better performance
    # Only impute NaNs in columns that are >95% missing (extremely sparse features)
    for col in X_train.columns:
        if X_train[col].isna().mean() > 0.95:
            median_val = X_train[col].median()
            if pd.isna(median_val):  # If median is also NaN, use 0
                median_val = 0.0
            X_train[col] = X_train[col].fillna(median_val)
            X_val[col] = X_val[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
            logging.info(f"Imputed extremely sparse feature: {col} (>95% missing)")
    
    # Log NaN preservation for XGBoost learning
    remaining_nans = X_train.isna().sum().sum()
    if remaining_nans > 0:
        logging.info(f"✓ Preserved {remaining_nans} NaN values for XGBoost missingness learning")
    else:
        logging.info(f"✓ No NaN values remaining after sparse feature imputation")
    
    # Final verification
    logging.info(f"After cleaning - NaN counts:")
    logging.info(f"  X_train: {X_train.isna().sum().sum()}")
    logging.info(f"  X_val: {X_val.isna().sum().sum()}")
    logging.info(f"  X_test: {X_test.isna().sum().sum()}")
    
    logging.info(f"After cleaning - Inf counts:")
    logging.info(f"  X_train: {np.isinf(X_train).sum().sum()}")
    logging.info(f"  X_val: {np.isinf(X_val).sum().sum()}")
    logging.info(f"  X_test: {np.isinf(X_test).sum().sum()}")
    
    # Check for extremely large values that might cause overflow
    max_abs_vals = X_train.abs().max()
    large_value_cols = max_abs_vals[max_abs_vals > 1e10].index.tolist()
    if large_value_cols:
        logging.warning(f"Found {len(large_value_cols)} columns with very large values (>1e10)")
        logging.warning(f"Large value columns: {large_value_cols[:5]}...")
        
        # Clip extremely large values
        for col in large_value_cols:
            p99 = X_train[col].quantile(0.99)
            p1 = X_train[col].quantile(0.01)
            X_train[col] = X_train[col].clip(p1, p99)
            X_val[col] = X_val[col].clip(p1, p99)
            X_test[col] = X_test[col].clip(p1, p99)
        
        logging.info(f"✓ Clipped extreme values using 1st-99th percentile bounds")
    
    return X_train, X_val, X_test

# =============================================================================
# UNCERTAINTY QUANTIFICATION FUNCTIONS
# =============================================================================

def calculate_bootstrap_ci(y_true, y_pred_proba, metric_func, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """
    Calculate bootstrap confidence intervals for a given metric.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities 
        metric_func: Function to calculate metric (e.g., roc_auc_score)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random state for reproducibility
    
    Returns:
        dict: Contains 'point_estimate', 'ci_lower', 'ci_upper', 'std', 'bootstrap_scores'
    """
    np.random.seed(random_state)
    bootstrap_scores = []
    
    n_samples = len(y_true)
    
    # Perform bootstrap resampling
    for i in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
        # Stratified bootstrap to maintain class balance
        pos_indices = np.where(y_true == 1)[0]
        neg_indices = np.where(y_true == 0)[0]
        
        # Sample with replacement maintaining original class distribution
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        boot_pos_indices = np.random.choice(pos_indices, size=n_pos, replace=True)
        boot_neg_indices = np.random.choice(neg_indices, size=n_neg, replace=True)
        boot_indices = np.concatenate([boot_pos_indices, boot_neg_indices])
        
        # Calculate metric on bootstrap sample
        try:
            score = metric_func(y_true[boot_indices], y_pred_proba[boot_indices])
            bootstrap_scores.append(score)
        except Exception as e:
            # Skip if bootstrap sample has issues
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    
    # Point estimate on original data
    point_estimate = metric_func(y_true, y_pred_proba)
    
    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': np.std(bootstrap_scores),
        'bootstrap_scores': bootstrap_scores,
        'n_bootstrap': len(bootstrap_scores)
    }

def evaluate_with_uncertainty(y_true, y_pred_proba, y_pred=None, n_bootstrap=1000):
    """
    Comprehensive evaluation with uncertainty quantification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        y_pred: Predicted binary labels (optional, will be calculated if None)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        dict: Comprehensive results with confidence intervals
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    logging.info(f"Calculating bootstrap confidence intervals with {n_bootstrap} samples...")
    
    # Calculate AUROC with CI
    auroc_results = calculate_bootstrap_ci(y_true, y_pred_proba, roc_auc_score, n_bootstrap)
    
    # Calculate AUPRC with CI  
    auprc_results = calculate_bootstrap_ci(y_true, y_pred_proba, average_precision_score, n_bootstrap)
    
    # Basic metrics without CI (for completeness)
    basic_metrics = {
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    return {
        'auroc': auroc_results,
        'auprc': auprc_results,
        'basic_metrics': basic_metrics
    }

# =============================================================================
# MODEL TUNING AND TRAINING
# =============================================================================

def tune_xgboost_with_optuna(X_train, y_train, X_val, y_val):
    """
    Uses Optuna to find the best hyperparameters for the XGBoost model.
    """
    logging.info(f"Starting Optuna hyperparameter search with {N_OPTUNA_TRIALS} trials...")
    
    # Calculate scale_pos_weight for handling class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': scale_pos_weight,
            'random_state': SEED,
            'n_jobs': -1
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  early_stopping_rounds=30,
                  verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        return auc

    study = optuna.create_study(direction='maximize', study_name='XGBoost_Mortality_Prediction')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
    
    logging.info(f"✓ Optuna study complete. Best AUROC on validation set: {study.best_value:.4f}")
    logging.info("Best hyperparameters found:")
    for key, value in study.best_params.items():
        logging.info(f"  {key}: {value}")
        
    return study.best_params

# =============================================================================
# MAIN EXECUTION SCRIPT
# =============================================================================

def main(config_dict=None):
    """Main function to run the XGBoost analysis."""
    # Set configuration first
    set_config(config_dict)
    
    start_time = time.time()
    
    # --- 1. Load Preprocessed Data ---
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputation_values = load_preprocessed_data()
    
    # --- 2. Additional Data Cleaning for XGBoost ---
    X_train, X_val, X_test = clean_data_for_xgboost(X_train, X_val, X_test)
    
    # --- 3. Tune Hyperparameters ---
    best_params = tune_xgboost_with_optuna(X_train, y_train, X_val, y_val)
    
    # --- 4. Train Final Model & Evaluate ---
    logging.info("Training final model on combined train+validation data...")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    final_scale_pos_weight = (y_train_full == 0).sum() / (y_train_full == 1).sum()
    final_params = best_params.copy()
    final_params['scale_pos_weight'] = final_scale_pos_weight
    final_params['random_state'] = SEED
    final_params['n_jobs'] = -1
    
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train_full, y_train_full, verbose=False)
    
    # --- 5. Final Evaluation on Test Set ---
    logging.info("--- FINAL EVALUATION ON TEST SET ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate comprehensive metrics with uncertainty quantification
    evaluation_results = evaluate_with_uncertainty(
        y_test.values, y_pred_proba, y_pred, n_bootstrap=1000
    )
    
    # Extract results
    auroc_results = evaluation_results['auroc']
    auprc_results = evaluation_results['auprc']
    
    # Log results with confidence intervals
    logging.info(f"Test Set AUROC: {auroc_results['point_estimate']:.4f} "
                f"(95% CI: {auroc_results['ci_lower']:.4f}-{auroc_results['ci_upper']:.4f})")
    logging.info(f"Test Set AUPRC: {auprc_results['point_estimate']:.4f} "
                f"(95% CI: {auprc_results['ci_lower']:.4f}-{auprc_results['ci_upper']:.4f})")
    
    # Log additional uncertainty statistics
    logging.info(f"AUROC Bootstrap Std: {auroc_results['std']:.4f}")
    logging.info(f"AUPRC Bootstrap Std: {auprc_results['std']:.4f}")
    logging.info(f"Bootstrap samples used: {auroc_results['n_bootstrap']}")
    
    logging.info("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    logging.info("\n" + report)
    
    logging.info("Confusion Matrix:")
    cm = evaluation_results['basic_metrics']['confusion_matrix']
    print(cm)
    logging.info("\n" + str(cm))
    
    # --- 6. Save Artifacts ---
    logging.info("Saving all XGBoost analysis artifacts...")
    
    # Save final model
    model_path = os.path.join(OUTPUT_DIR, 'model_1_xgboost_baseline.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"✓ Model saved to: {model_path}")
    
    # Save results with uncertainty quantification
    results = {
        'model_name': 'Model 1 (XGBoost Baseline)',
        'target_variable': TARGET_VARIABLE,
        'test_auroc': auroc_results['point_estimate'],
        'test_auroc_ci_lower': auroc_results['ci_lower'],
        'test_auroc_ci_upper': auroc_results['ci_upper'],
        'test_auroc_std': auroc_results['std'],
        'test_auprc': auprc_results['point_estimate'],
        'test_auprc_ci_lower': auprc_results['ci_lower'],
        'test_auprc_ci_upper': auprc_results['ci_upper'],
        'test_auprc_std': auprc_results['std'],
        'bootstrap_samples': auroc_results['n_bootstrap'],
        'evaluation_results_full': evaluation_results,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm.tolist(),
        'best_hyperparameters': best_params
    }
    results_path = os.path.join(OUTPUT_DIR, 'results_xgboost_baseline.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"✓ Results dictionary saved to: {results_path}")
    
    total_time = time.time() - start_time
    logging.info(f"--- XGBoost analysis finished successfully in {total_time/60:.2f} minutes. ---")

if __name__ == "__main__":
    main() 