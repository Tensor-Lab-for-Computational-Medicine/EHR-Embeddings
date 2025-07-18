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
# CONFIGURATION
# =============================================================================

class Config:
    def __init__(self, config_dict=None):
        config = config_dict if config_dict else {}
        
        # Explicitly define all attributes to avoid linter errors
        self.TARGET_VARIABLE = config.get('TARGET_VARIABLE', 'mort_hosp')  # This is the variable we are predicting
        self.INPUT_DIR = config.get('INPUT_DIR', 'notebooks\Phase 1 and 2\phase_1_outputs')  # Where to load data from
        self.OUTPUT_DIR = os.path.join(self.INPUT_DIR, self.TARGET_VARIABLE)  # Where to save results

        self.DRY_RUN = config.get('DRY_RUN', False)
        self.DRY_RUN_PATIENTS = config.get('DRY_RUN_PATIENTS', 1000)
        self.CALCULATE_TRENDS = config.get('CALCULATE_TRENDS', True)
        self.WINDOW_SIZE = config.get('WINDOW_SIZE', 24)
        self.GAP_TIME = config.get('GAP_TIME', 6)
        self.TARGET_VARIABLES = config.get('TARGET_VARIABLES', ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso'])
        self.SEED = config.get('SEED', 42)
        self.N_OPTUNA_TRIALS = config.get('N_OPTUNA_TRIALS', 30)
        self.OPTUNA_TIMEOUT = config.get('OPTUNA_TIMEOUT', 1800)
        self.REUSE_EXISTING_STUDY = config.get('REUSE_EXISTING_STUDY', True)
        
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATA HANDLING
# =============================================================================

def get_cache_prefix(config):
    """Generate cache filename prefix."""
    prefix = f"preprocessed_{config.TARGET_VARIABLE}"
    if config.DRY_RUN:
        prefix += f"_dryrun_{config.DRY_RUN_PATIENTS}"
    prefix += f"_trends_{config.CALCULATE_TRENDS}_window_{config.WINDOW_SIZE}_gap_{config.GAP_TIME}_seed_{config.SEED}"
    return prefix

def load_preprocessed_data(config):
    """Load preprocessed data from cache files."""
    prefix = get_cache_prefix(config)
    cache_files = {
        key: os.path.join(config.OUTPUT_DIR, f'{prefix}_{key}.pkl')
        for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'scaler', 'imputation_values']
    }
    
    missing_files = [f for f in cache_files.values() if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing preprocessed data files: {missing_files}")
    
    logging.info(f"Loading preprocessed data with prefix: {prefix}")
    data = {}
    for key, filepath in cache_files.items():
        with open(filepath, 'rb') as f:
            data[key] = pickle.load(f)
    
    logging.info(f"Data shapes: X_train={data['X_train'].shape}, X_val={data['X_val'].shape}, X_test={data['X_test'].shape}")
    return data['X_train'], data['X_val'], data['X_test'], data['y_train'], data['y_val'], data['y_test'], data['scaler'], data['imputation_values']

# =============================================================================
# EVALUATION WITH UNCERTAINTY QUANTIFICATION
# =============================================================================

def bootstrap_metric(y_true, y_pred_proba, metric_func, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """Calculate bootstrap confidence intervals for a metric."""
    np.random.seed(random_state)
    scores = []
    n_samples = len(y_true)
    
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    for _ in range(n_bootstrap):
        # Stratified bootstrap sampling
        boot_pos = np.random.choice(pos_indices, size=len(pos_indices), replace=True)
        boot_neg = np.random.choice(neg_indices, size=len(neg_indices), replace=True)
        boot_indices = np.concatenate([boot_pos, boot_neg])
        
        try:
            score = metric_func(y_true[boot_indices], y_pred_proba[boot_indices])
            scores.append(score)
        except:
            continue
    
    scores = np.array(scores)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(scores, (alpha/2) * 100)
    ci_upper = np.percentile(scores, (1 - alpha/2) * 100)
    
    return {
        'point_estimate': metric_func(y_true, y_pred_proba),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': np.std(scores),
        'n_bootstrap': len(scores)
    }

def evaluate_with_uncertainty(y_true, y_pred_proba, y_pred=None, n_bootstrap=1000):
    """Comprehensive evaluation with uncertainty quantification."""
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    logging.info(f"Calculating bootstrap CIs with {n_bootstrap} samples...")
    
    return {
        'auroc': bootstrap_metric(y_true, y_pred_proba, roc_auc_score, n_bootstrap),
        'auprc': bootstrap_metric(y_true, y_pred_proba, average_precision_score, n_bootstrap),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }

# =============================================================================
# MODEL TRAINING
# =============================================================================

def tune_xgboost(X_train, y_train, X_val, y_val, config):
    """Tune XGBoost hyperparameters using Optuna."""
    study_path = os.path.join(config.OUTPUT_DIR, f'{get_cache_prefix(config)}_optuna_study.pkl')
    
    if os.path.exists(study_path) and config.REUSE_EXISTING_STUDY:
        logging.info(f"Loading existing Optuna study from {study_path}")
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        logging.info(f"Loaded study with {len(study.trials)} completed trials, best AUROC: {study.best_value:.4f}")
        logging.info("Reusing existing study results without additional optimization")
        return study.best_params
    
    # Create new study only if no existing study or REUSE_EXISTING_STUDY is False
    logging.info(f"Creating new Optuna study with {config.N_OPTUNA_TRIALS} trials...")
    study = optuna.create_study(direction='maximize')
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': scale_pos_weight,
            'random_state': config.SEED,
            'n_jobs': -1
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
                  early_stopping_rounds=30, verbose=False)
        
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS, timeout=config.OPTUNA_TIMEOUT)
    
    # Save the study
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    logging.info(f"Study saved to {study_path}")
    
    logging.info(f"Best validation AUROC: {study.best_value:.4f}")
    return study.best_params

def train_final_model(X_train, X_val, y_train, y_val, best_params, config):
    """Train final model on combined train+validation data."""
    logging.info("Training final model on combined train+validation data...")
    
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    
    # Reset index after concatenation
    X_full = X_full.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)
    
    final_params = best_params.copy()
    final_params.update({
        'scale_pos_weight': (y_full == 0).sum() / (y_full == 1).sum(),
        'random_state': config.SEED,
        'n_jobs': -1
    })
    
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_full, y_full, verbose=False)
    return model

def save_results(model, results, best_params, config):
    """Save model and results."""
    # Save model
    model_path = os.path.join(config.OUTPUT_DIR, 'model_1_xgboost_baseline.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save results
    results_dict = {
        'model_name': 'Model 1 (XGBoost Baseline)',
        'target_variable': config.TARGET_VARIABLE,
        'test_auroc': results['auroc']['point_estimate'],
        'test_auroc_ci_lower': results['auroc']['ci_lower'],
        'test_auroc_ci_upper': results['auroc']['ci_upper'],
        'test_auprc': results['auprc']['point_estimate'],
        'test_auprc_ci_lower': results['auprc']['ci_lower'],
        'test_auprc_ci_upper': results['auprc']['ci_upper'],
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'best_hyperparameters': best_params,
        'full_evaluation': results
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, 'results_xgboost_baseline.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    logging.info(f"Model saved to: {model_path}")
    logging.info(f"Results saved to: {results_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(config_dict=None):
    """Main function to run XGBoost analysis."""
    config = Config(config_dict)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'xgboost_analysis_log.txt'), mode='w'),
            logging.StreamHandler()
        ]
    )
    
    start_time = time.time()
    
    # Load preprocessed data (no additional cleaning needed)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputation_values = load_preprocessed_data(config)
    logging.info("Using preprocessed data directly - no additional cleaning required")
    
    # Tune hyperparameters
    best_params = tune_xgboost(X_train, y_train, X_val, y_val, config)
    
    # Train final model
    model = train_final_model(X_train, X_val, y_train, y_val, best_params, config)
    
    # Evaluate on test set
    logging.info("--- FINAL EVALUATION ON TEST SET ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    results = evaluate_with_uncertainty(y_test.values, y_pred_proba, y_pred, n_bootstrap=1000)
    
    # Log results
    auroc = results['auroc']
    auprc = results['auprc']
    logging.info(f"Test AUROC: {auroc['point_estimate']:.4f} (95% CI: {auroc['ci_lower']:.4f}-{auroc['ci_upper']:.4f})")
    logging.info(f"Test AUPRC: {auprc['point_estimate']:.4f} (95% CI: {auprc['ci_lower']:.4f}-{auprc['ci_upper']:.4f})")
    
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
    
    # Save all artifacts
    save_results(model, results, best_params, config)
    
    logging.info(f"Analysis completed in {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main() 