# elastic_net_analysis.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
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
        self.OUTPUT_DIR = config.get('OUTPUT_DIR', 'phase_1_outputs')
        self.DRY_RUN = config.get('DRY_RUN', True)
        self.DRY_RUN_PATIENTS = config.get('DRY_RUN_PATIENTS', 1000)
        self.CALCULATE_TRENDS = config.get('CALCULATE_TRENDS', True)
        self.WINDOW_SIZE = config.get('WINDOW_SIZE', 24)
        self.GAP_TIME = config.get('GAP_TIME', 6)
        self.TARGET_VARIABLE = config.get('TARGET_VARIABLE', 'mort_hosp')
        self.SEED = config.get('SEED', 42)
        # Reduced trials and timeout for faster optimization
        self.N_OPTUNA_TRIALS = config.get('N_OPTUNA_TRIALS', 10)  # Reduced from 15
        self.OPTUNA_TIMEOUT = config.get('OPTUNA_TIMEOUT', 600)   # Reduced from 1800 (10 min vs 30 min)
        
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

def clean_data_for_elastic_net(X_train, X_val, X_test):
    """Optimized data cleaning for Elastic Net (data should already be scaled)."""
    logging.info("Performing optimized data cleaning for Elastic Net...")
    
    # Log data types - should all be numeric after preprocessing
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']
    non_numeric_cols = [col for col in X_train.columns if str(X_train[col].dtype) not in numeric_types]
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Found non-numeric columns after preprocessing: {non_numeric_cols}. "
                        "All categorical encoding should be done in preprocessing.")
    
    logging.info(f"✓ All {len(X_train.columns)} columns are numeric")
    
    # Replace infinite values with NaN (fast vectorized operation)
    logging.info("Replacing infinite values...")
    for df in [X_train, X_val, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Use more efficient float32 instead of float64 for speed
    logging.info("Converting to float32 for efficiency...")
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32) 
    X_test = X_test.astype(np.float32)
    
    # Count missing values
    missing_counts = X_train.isna().sum()
    total_missing = missing_counts.sum()
    logging.info(f"Total missing values before imputation: {total_missing}")
    
    # Fast median imputation with fit_transform
    logging.info("Performing median imputation...")
    imputer = SimpleImputer(strategy='median', copy=False)  # copy=False for memory efficiency
    
    # Fit and transform in one step for training, then transform others
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
        dtype=np.float32
    )
    
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
        dtype=np.float32
    )
    
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
        dtype=np.float32
    )
    
    # Verify no missing values remain
    remaining_missing = X_train_imputed.isna().sum().sum()
    if remaining_missing > 0:
        raise ValueError(f"Imputation failed: {remaining_missing} missing values remain")
    
    logging.info(f"✓ Successfully imputed {total_missing} missing values")
    logging.info(f"✓ Data prepared for Elastic Net (already scaled, no missing values)")
    
    return X_train_imputed, X_val_imputed, X_test_imputed, imputer

# =============================================================================
# EVALUATION WITH UNCERTAINTY QUANTIFICATION
# =============================================================================

def bootstrap_metric(y_true, y_pred_proba, metric_func, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """Calculate bootstrap confidence intervals for a metric with optimized sampling."""
    np.random.seed(random_state)
    scores = []
    n_samples = len(y_true)
    
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    # Pre-allocate arrays for efficiency
    scores = np.zeros(n_bootstrap)
    valid_samples = 0
    
    for i in range(n_bootstrap):
        # Stratified bootstrap sampling
        boot_pos = np.random.choice(pos_indices, size=len(pos_indices), replace=True)
        boot_neg = np.random.choice(neg_indices, size=len(neg_indices), replace=True)
        boot_indices = np.concatenate([boot_pos, boot_neg])
        
        try:
            score = metric_func(y_true[boot_indices], y_pred_proba[boot_indices])
            scores[valid_samples] = score
            valid_samples += 1
        except:
            continue
    
    # Use only valid samples
    scores = scores[:valid_samples]
    alpha = 1 - confidence_level
    ci_lower = np.percentile(scores, (alpha/2) * 100)
    ci_upper = np.percentile(scores, (1 - alpha/2) * 100)
    
    return {
        'point_estimate': metric_func(y_true, y_pred_proba),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': np.std(scores),
        'n_bootstrap': valid_samples
    }

def evaluate_with_uncertainty(y_true, y_pred_proba, y_pred=None, n_bootstrap=1000):
    """Optimized evaluation with uncertainty quantification (reduced bootstrap samples)."""
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    logging.info(f"Calculating bootstrap CIs with {n_bootstrap} samples (optimized)...")
    
    return {
        'auroc': bootstrap_metric(y_true, y_pred_proba, roc_auc_score, n_bootstrap),
        'auprc': bootstrap_metric(y_true, y_pred_proba, average_precision_score, n_bootstrap),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }

# =============================================================================
# MODEL TRAINING
# =============================================================================

def tune_elastic_net(X_train, y_train, X_val, y_val, config):
    """Optimized Elastic Net hyperparameter tuning with wider parameter exploration."""
    logging.info(f"Starting optimized Optuna search with {config.N_OPTUNA_TRIALS} trials...")
    
    def objective(trial):
        # Wider hyperparameter ranges for meaningful exploration
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)  # Even wider range: 0.0001 to 100
        l1_ratio = trial.suggest_float('l1_ratio', 0.01, 0.99)  # Full range but avoid exact 0/1
        # Vary max_iter more to see convergence differences
        max_iter = trial.suggest_int('max_iter', 1000, 5000, step=500)
        # Add tolerance as a hyperparameter
        tol = trial.suggest_float('tol', 1e-5, 1e-2, log=True)
        
        # CRITICAL FIX: Remove fixed random seed to allow different initializations
        # Use trial number to create different but reproducible seeds
        trial_seed = config.SEED + trial.number
        
        model = LogisticRegression(
            penalty='elasticnet',
            C=C,
            l1_ratio=l1_ratio,
            solver='saga',
            class_weight='balanced',
            random_state=trial_seed,  # Different seed per trial
            max_iter=max_iter,
            n_jobs=1,  # Single job for stability with saga solver
            warm_start=False,
            tol=tol,  # Variable tolerance
            fit_intercept=True
        )
        
        # Ensure proper dtypes
        y_train_clean = y_train.astype(int)
        y_val_clean = y_val.astype(int)
        
        try:
            # Use fit with error handling
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model.fit(X_train, y_train_clean)
            
            # Check if model converged
            converged = True
            if hasattr(model, 'n_iter_'):
                actual_iter = model.n_iter_
                if actual_iter >= max_iter:
                    logging.warning(f"Trial {trial.number} reached max_iter ({max_iter}) with C={C:.4f}, l1_ratio={l1_ratio:.3f}")
                    converged = False
                else:
                    logging.info(f"Trial {trial.number} converged in {actual_iter} iterations, C={C:.4f}, l1_ratio={l1_ratio:.3f}, tol={tol:.2e}")
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val_clean, y_pred_proba)
            
            # Add convergence penalty for non-converged models
            if not converged:
                score *= 0.95  # Small penalty for non-convergence
            
            # Log more detailed information
            n_features_used = np.sum(np.abs(model.coef_[0]) > 1e-6)
            logging.info(f"Trial {trial.number} AUROC: {score:.6f} [C={C:.4f}, l1_ratio={l1_ratio:.3f}, features_used={n_features_used}, converged={converged}]")
            
            return score
            
        except Exception as e:
            logging.warning(f"Trial {trial.number} failed with C={C:.4f}, l1_ratio={l1_ratio:.3f}: {e}")
            return 0.0  # Return poor score for failed trials

    # Create study with better settings and different sampler
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=config.SEED),  # Use TPE sampler for better exploration
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    )
    
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS, timeout=config.OPTUNA_TIMEOUT)
    
    logging.info(f"Best validation AUROC: {study.best_value:.6f}")
    logging.info(f"Best parameters: {study.best_params}")
    
    # Log parameter exploration statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 1:
        C_values = [t.params['C'] for t in completed_trials]
        l1_ratios = [t.params['l1_ratio'] for t in completed_trials]
        scores = [t.value for t in completed_trials]
        
        logging.info(f"Parameter exploration summary:")
        logging.info(f"  C range explored: {min(C_values):.6f} - {max(C_values):.6f}")
        logging.info(f"  l1_ratio range explored: {min(l1_ratios):.3f} - {max(l1_ratios):.3f}")
        logging.info(f"  AUROC range: {min(scores):.6f} - {max(scores):.6f}")
        logging.info(f"  AUROC std: {np.std(scores):.6f}")
        
        # Check if we're seeing meaningful variation
        if np.std(scores) < 1e-5:
            logging.warning("Very low variation in AUROC scores - consider checking data preprocessing or model setup")
    
    # Add convergence check for best params
    if study.best_value < 0.6:
        logging.warning("Best model has poor performance - consider adjusting hyperparameter ranges")
    
    return study.best_params

def train_final_model(X_train, X_val, y_train, y_val, best_params, config):
    """Train final model with aggressive convergence settings."""
    logging.info("Training final model on combined train+validation data...")
    
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    
    # Reset index and ensure proper dtypes after concat
    X_full = X_full.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)
    
    # Convert dtypes efficiently
    X_full = X_full.astype(np.float32)
    y_full = y_full.astype(int)
    
    final_params = best_params.copy()
    final_params.update({
        'penalty': 'elasticnet',
        'solver': 'saga',
        'class_weight': 'balanced',
        'random_state': config.SEED,  # Use fixed seed for final model
        'n_jobs': 1,  # Single job for stability
        'warm_start': False,
        'fit_intercept': True
    })
    
    # Use the tolerance found during optimization, but ensure high max_iter for final model
    final_tol = final_params.get('tol', 1e-4)
    final_max_iter = max(final_params.get('max_iter', 5000), 5000)  # Ensure at least 5000 iterations
    final_params['max_iter'] = final_max_iter
    final_params['tol'] = final_tol
    
    logging.info(f"Using final parameters: C={final_params['C']:.6f}, l1_ratio={final_params['l1_ratio']:.3f}")
    logging.info(f"Using tol={final_tol:.2e}, max_iter={final_max_iter}")
    
    model = LogisticRegression(**final_params)
    
    start_time = time.time()
    
    # Train with warning suppression
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X_full, y_full)
    
    training_time = time.time() - start_time
    
    # Check convergence
    if hasattr(model, 'n_iter_'):
        if model.n_iter_ >= final_max_iter:
            logging.warning(f"Final model did not converge (used {model.n_iter_} iterations)")
        else:
            logging.info(f"Final model converged in {model.n_iter_} iterations ({training_time:.2f}s)")
    
    # Log feature usage
    n_features_used = np.sum(np.abs(model.coef_[0]) > 1e-6)
    total_features = len(model.coef_[0])
    logging.info(f"Final model uses {n_features_used}/{total_features} features ({100*n_features_used/total_features:.1f}%)")
    
    return model

def save_results(model, imputer, scaler, results, best_params, config):
    """Save model, imputer, preprocessing scaler and results."""
    # Save model
    model_path = os.path.join(config.OUTPUT_DIR, 'model_2_elastic_net.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save imputer
    imputer_path = os.path.join(config.OUTPUT_DIR, 'imputer_elastic_net.pkl')
    with open(imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    
    # Save preprocessing scaler
    scaler_path = os.path.join(config.OUTPUT_DIR, 'scaler_elastic_net.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save results
    results_dict = {
        'model_name': 'Model 2 (Elastic Net)',
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
    
    results_path = os.path.join(config.OUTPUT_DIR, 'results_elastic_net.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    logging.info(f"Model saved to: {model_path}")
    logging.info(f"Imputer saved to: {imputer_path}")
    logging.info(f"Preprocessing scaler saved to: {scaler_path}")
    logging.info(f"Results saved to: {results_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(config_dict=None):
    """Main function to run Elastic Net analysis."""
    config = Config(config_dict)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'elastic_net_analysis_log.txt'), mode='w'),
            logging.StreamHandler()
        ]
    )
    
    start_time = time.time()
    
    # Load and clean data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputation_values = load_preprocessed_data(config)
    X_train, X_val, X_test, imputer = clean_data_for_elastic_net(X_train, X_val, X_test)
    
    # Tune hyperparameters
    best_params = tune_elastic_net(X_train, y_train, X_val, y_val, config)
    
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
    save_results(model, imputer, scaler, results, best_params, config)
    
    logging.info(f"Analysis completed in {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main() 