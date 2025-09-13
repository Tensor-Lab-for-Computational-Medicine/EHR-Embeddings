# xgboost_analysis.py

# Configure NumExpr to use maximum cores available before any imports
import os
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
print(f"Setting NUMEXPR_MAX_THREADS to {os.cpu_count()} cores")

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import logging
import time
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    def __init__(self, config_dict=None):
        config = config_dict if config_dict else {}
        
        # Explicitly define all attributes to avoid linter errors
        self.TARGET_VARIABLE = config.get('TARGET_VARIABLE', 'intervention_vaso')  # This is the variable we are predicting
        self.INPUT_DIR = config.get('INPUT_DIR', 'notebooks\Phase 1 and 2\phase_1_outputs')  # Where to load data from
        self.OUTPUT_DIR = os.path.join(self.INPUT_DIR, self.TARGET_VARIABLE)  # Where to save results

        self.DRY_RUN = config.get('DRY_RUN', False)
        self.DRY_RUN_PATIENTS = config.get('DRY_RUN_PATIENTS', 1000)
        self.CALCULATE_TRENDS = config.get('CALCULATE_TRENDS', True)
        self.WINDOW_SIZE = config.get('WINDOW_SIZE', 24)
        self.GAP_TIME = config.get('GAP_TIME', 6)
        self.TARGET_VARIABLES = config.get('TARGET_VARIABLES', ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso'])
        self.SEED = config.get('SEED', 42)
        self.N_OPTUNA_TRIALS = config.get('N_OPTUNA_TRIALS', 15)
        self.OPTUNA_TIMEOUT = config.get('OPTUNA_TIMEOUT', 1800)
        self.REUSE_EXISTING_STUDY = config.get('REUSE_EXISTING_STUDY', True)
        self.CALIBRATION_ENABLED = config.get('CALIBRATION_ENABLED', True)
        self.CALIBRATION_METHOD = config.get('CALIBRATION_METHOD', 'isotonic')  # 'isotonic' or 'sigmoid'
        self.VAL_CAL_FRACTION = config.get('VAL_CAL_FRACTION', 0.5)  # Fraction of original val used for calibration
        
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATA HANDLING
# =============================================================================

def get_cache_prefix(config):
    """Generate cache filename prefix."""
    prefix = f"preprocessed_{'_'.join(config.TARGET_VARIABLES)}"
    if config.DRY_RUN:
        prefix += f"_dryrun_{config.DRY_RUN_PATIENTS}"
    prefix += f"_trends_{config.CALCULATE_TRENDS}_window_{config.WINDOW_SIZE}_gap_{config.GAP_TIME}_seed_{config.SEED}"
    return prefix

def load_preprocessed_data(config):
    """Load preprocessed data using fixed filenames; fallback to prefixed if needed."""
    keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'scaler', 'imputation_values']
    fixed_files = {key: os.path.join(config.INPUT_DIR, f'{key}.pkl') for key in keys}
    use_fixed = all(os.path.exists(path) for path in fixed_files.values())
    
    if use_fixed:
        cache_files = fixed_files
        logging.info("Loading preprocessed data from fixed filenames (override mode)")
    else:
        prefix = get_cache_prefix(config)
        cache_files = {key: os.path.join(config.INPUT_DIR, f'{prefix}_{key}.pkl') for key in keys}
        missing_files = [f for f in cache_files.values() if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing preprocessed data files: {missing_files}")
        logging.info(f"Loading preprocessed data with prefix: {prefix}")
    
    data = {}
    for key, filepath in cache_files.items():
        with open(filepath, 'rb') as f:
            data[key] = pickle.load(f)
    
    # Extract only the target variable we're analyzing from the multi-target y data
    y_train = data['y_train'][config.TARGET_VARIABLE]
    y_val = data['y_val'][config.TARGET_VARIABLE]
    y_test = data['y_test'][config.TARGET_VARIABLE]
    
    logging.info(f"Data shapes: X_train={data['X_train'].shape}, X_val={data['X_val'].shape}, X_test={data['X_test'].shape}")
    logging.info(f"Target variable: {config.TARGET_VARIABLE}, y_train shape: {y_train.shape}")
    return data['X_train'], data['X_val'], data['X_test'], y_train, y_val, y_test, data['scaler'], data['imputation_values']

# =============================================================================
# VALIDATION SPLITTING FOR TUNING AND CALIBRATION
# =============================================================================

def _clean_features_and_labels(X: pd.DataFrame, y: pd.Series, expected_binary: bool = True):
    """Clean labels and align feature rows.
    - Aligns X and y on common index (if X is a DataFrame)
    - Drops rows where y is NaN/Inf
    - If expected_binary, keeps only labels in {0, 1} (coercing bools to ints)
    - Returns copies with reset indices and integer labels
    """
    y_series = pd.Series(y)
    # Align by index if possible
    if isinstance(X, pd.DataFrame):
        common_index = X.index.intersection(y_series.index)
        X_aligned = X.loc[common_index]
        y_aligned = y_series.loc[common_index]
    else:
        X_aligned = X
        y_aligned = y_series
    # Coerce to numeric to handle True/False etc.
    y_numeric = pd.to_numeric(y_aligned, errors='coerce')
    # Build mask AFTER alignment
    finite_mask = np.isfinite(y_numeric.values)
    mask = finite_mask
    if expected_binary:
        binary_mask = (y_numeric.values == 0) | (y_numeric.values == 1)
        mask = mask & binary_mask
    # Filter
    if isinstance(X_aligned, pd.DataFrame):
        X_clean = X_aligned.loc[mask]
        y_clean = y_numeric.loc[mask]
        # Optionally drop rows with NaNs in features
        row_mask = ~X_clean.isnull().any(axis=1)
        if row_mask.sum() < len(X_clean):
            X_clean = X_clean.loc[row_mask]
            y_clean = y_clean.loc[X_clean.index]
        X_clean = X_clean.reset_index(drop=True)
        y_clean = y_clean.reset_index(drop=True)
    else:
        # Assume numpy arrays of same length
        X_clean = X_aligned[mask]
        y_clean = pd.Series(y_numeric.values[mask]).reset_index(drop=True)
    # Cast labels to int
    y_clean = y_clean.astype(int)
    return X_clean, y_clean

def split_validation_set(X_val, y_val, config):
    """Split the original validation set into tuning and calibration subsets (stratified)."""
    logging.info(
        f"Splitting validation set: {1 - config.VAL_CAL_FRACTION:.2f} for tuning, {config.VAL_CAL_FRACTION:.2f} for calibration"
    )
    X_val_tune, X_val_cal, y_val_tune, y_val_cal = train_test_split(
        X_val,
        y_val,
        test_size=config.VAL_CAL_FRACTION,
        random_state=config.SEED,
        stratify=y_val
    )
    logging.info(
        f"val_tune shape: {X_val_tune.shape}, val_cal shape: {X_val_cal.shape}"
    )
    return X_val_tune, X_val_cal, y_val_tune, y_val_cal

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
        'brier': bootstrap_metric(y_true, y_pred_proba, brier_score_loss, n_bootstrap),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }

# =============================================================================
# CALIBRATION / ALIGNMENT METRICS & PLOTS (concise)
# =============================================================================

def _safe_logit(probabilities, eps=1e-6):
    p = np.clip(probabilities, eps, 1 - eps)
    return np.log(p / (1 - p))

def compute_calibration_metrics(y_true, y_proba, n_bins=15):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba).astype(float)

    # ECE/MCE
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(p, edges[1:-1], right=False)
    ece = 0.0; mce = 0.0; total = len(p); per_bin = []
    for b in range(n_bins):
        m = bins == b
        cnt = int(np.sum(m))
        if cnt == 0:
            per_bin.append({'bin': b, 'count': 0, 'accuracy': None, 'confidence': None}); continue
        acc = float(np.mean(y[m])); conf = float(np.mean(p[m])); gap = abs(acc - conf)
        ece += (cnt / total) * gap; mce = max(mce, gap)
        per_bin.append({'bin': b, 'count': cnt, 'accuracy': acc, 'confidence': conf})

    # Brier, slope, intercept
    brier = float(brier_score_loss(y, p))
    X = _safe_logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(X, y)
    slope = float(lr.coef_.ravel()[0]); intercept = float(lr.intercept_.ravel()[0])

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy='uniform')
    return {
        'ece': ece, 'mce': mce, 'brier': brier, 'slope': slope, 'intercept': intercept,
        'reliability_curve': {
            'fraction_of_positives': frac_pos.tolist(),
            'mean_predicted_value': mean_pred.tolist()
        },
        'per_bin': per_bin
    }

def plot_reliability_comparison(y_true, p_before, p_after, output_dir, title='Reliability Diagram', n_bins=15, zoom=True):
    # Quantile binning ensures populated bins when probabilities cluster near 0
    frac_b, mean_b = calibration_curve(y_true, p_before, n_bins=n_bins, strategy='quantile')
    frac_a, mean_a = calibration_curve(y_true, p_after, n_bins=n_bins, strategy='quantile')

    x_max = float(max(mean_b.max() if len(mean_b) else 0.0, mean_a.max() if len(mean_a) else 0.0))
    y_max = float(max(frac_b.max() if len(frac_b) else 0.0, frac_a.max() if len(frac_a) else 0.0))
    if zoom:
        x_lim = max(0.1, x_max * 1.1) if x_max > 0 else 0.1
        y_lim = max(0.1, y_max * 1.1) if y_max > 0 else 0.1
    else:
        x_lim = 1.0; y_lim = 1.0

    plt.figure(figsize=(6, 6))
    plt.plot([0, x_lim], [0, x_lim], '--', color='gray', label='Perfect calibration')
    plt.plot(mean_b, frac_b, marker='o', label='Before')
    plt.plot(mean_a, frac_a, marker='o', label='After')
    plt.xlim(0, x_lim); plt.ylim(0, y_lim)
    plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives')
    plt.title(title); plt.legend(loc='best'); plt.tight_layout()
    path = os.path.join(output_dir, 'calibration_reliability.png')
    plt.savefig(path, dpi=150); plt.close(); return path

def evaluate_and_calibrate(base_model, X_test, y_test_values, X_val_cal, y_val_cal, config):
    model, used_method = calibrate_model(base_model, X_val_cal, y_val_cal, config) if config.CALIBRATION_ENABLED else (base_model, None)
    p_before = base_model.predict_proba(X_test)[:, 1]
    p_after = model.predict_proba(X_test)[:, 1]
    results = evaluate_with_uncertainty(y_test_values, p_after, n_bootstrap=1000)
    cal_before = compute_calibration_metrics(y_test_values, p_before)
    cal_after = compute_calibration_metrics(y_test_values, p_after)
    imp = {
        'ece_delta': cal_before['ece'] - cal_after['ece'],
        'mce_delta': cal_before['mce'] - cal_after['mce'],
        'brier_delta': cal_before['brier'] - cal_after['brier'],
        'slope_delta_toward_1': abs(1 - cal_before['slope']) - abs(1 - cal_after['slope']),
        'intercept_delta_toward_0': abs(cal_before['intercept']) - abs(cal_after['intercept'])
    }
    plot_path = plot_reliability_comparison(y_test_values, p_before, p_after, config.OUTPUT_DIR, title='Reliability Diagram - Model 1')
    logging.info(
        f"Calibration: ECE {cal_before['ece']:.4f}->{cal_after['ece']:.4f} (delta {imp['ece_delta']:.4f}); "
        f"MCE {cal_before['mce']:.4f}->{cal_after['mce']:.4f} (delta {imp['mce_delta']:.4f}); "
        f"Brier {cal_before['brier']:.4f}->{cal_after['brier']:.4f} (delta {imp['brier_delta']:.4f}); "
        f"Slope {cal_before['slope']:.3f}->{cal_after['slope']:.3f}; Intercept {cal_before['intercept']:.3f}->{cal_after['intercept']:.3f}"
    )
    results['calibration_alignment'] = {'before': cal_before, 'after': cal_after, 'improvement': imp, 'reliability_plot_path': plot_path, 'used_method': used_method}
    return model, results

# =============================================================================
# MODEL TRAINING
# =============================================================================

def tune_xgboost(X_train, y_train, X_val, y_val, config):
    """Tune XGBoost hyperparameters using Optuna."""
    # Include validation split info in study path to avoid mixing studies across different strategies
    study_path = os.path.join(
        config.OUTPUT_DIR,
        f"{get_cache_prefix(config)}_optuna_study_valtune_{len(X_val)}_calfrac_{'nocal' if not getattr(config, 'CALIBRATION_ENABLED', True) else config.VAL_CAL_FRACTION}.pkl"
    )
    
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

def train_final_model(X_train, X_val_tune, y_train, y_val_tune, best_params, config):
    """Train final model on combined train+val_tune data."""
    logging.info("Training final model on combined train+val_tune data...")
    
    X_full = pd.concat([X_train, X_val_tune])
    y_full = pd.concat([y_train, y_val_tune])
    
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

def calibrate_model(fitted_model, X_val_cal, y_val_cal, config):
    """Calibrate a pre-fitted classifier using a held-out calibration set.
    Tries methods and applies only if ECE improves on an internal holdout; otherwise skips.
    Returns (model, used_method or None).
    """
    if not config.CALIBRATION_ENABLED or X_val_cal is None:
        logging.info("Calibration disabled or no cal split. Skipping calibration.")
        return fitted_model, None

    # Split calibration set to prevent overfitting when choosing method
    X_c_train, X_c_eval, y_c_train, y_c_eval = train_test_split(
        X_val_cal, y_val_cal, test_size=0.3, random_state=config.SEED, stratify=y_val_cal
    )

    base_eval_probs = fitted_model.predict_proba(X_c_eval)[:, 1]
    base_ece = compute_calibration_metrics(y_c_eval.values, base_eval_probs)['ece']

    methods = ['isotonic', 'sigmoid'] if str(config.CALIBRATION_METHOD).lower() == 'auto' else [config.CALIBRATION_METHOD]
    best_method = None
    best_ece = base_ece

    for method in methods:
        try:
            calibrator = CalibratedClassifierCV(estimator=fitted_model, method=method, cv='prefit')
        except TypeError:
            calibrator = CalibratedClassifierCV(base_estimator=fitted_model, method=method, cv='prefit')
        calibrator.fit(X_c_train, y_c_train)
        eval_probs = calibrator.predict_proba(X_c_eval)[:, 1]
        ece = compute_calibration_metrics(y_c_eval.values, eval_probs)['ece']
        if ece < best_ece - 1e-6:  # require strict improvement
            best_ece = ece
            best_method = method

    if best_method is None:
        logging.info(f"Skipping calibration (base ECE={base_ece:.4f} is better than any method tried).")
        return fitted_model, None

    # Fit selected method on full calibration split
    try:
        final_calibrator = CalibratedClassifierCV(estimator=fitted_model, method=best_method, cv='prefit')
    except TypeError:
        final_calibrator = CalibratedClassifierCV(base_estimator=fitted_model, method=best_method, cv='prefit')
    final_calibrator.fit(X_val_cal, y_val_cal)
    logging.info(f"Calibration done using method='{best_method}'.")
    return final_calibrator, best_method

def save_results(model, results, best_params, config, is_calibrated=False):
    """Save model and results."""
    # Save model
    model_filename = 'model_1_xgboost_baseline_calibrated.pkl' if is_calibrated else 'model_1_xgboost_baseline.pkl'
    model_path = os.path.join(config.OUTPUT_DIR, model_filename)
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
        'test_brier': results['brier']['point_estimate'],
        'test_brier_ci_lower': results['brier']['ci_lower'],
        'test_brier_ci_upper': results['brier']['ci_upper'],
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'best_hyperparameters': best_params,
        'full_evaluation': results,
        'calibration_enabled': config.CALIBRATION_ENABLED,
        'calibration_method': config.CALIBRATION_METHOD if config.CALIBRATION_ENABLED else None,
        'val_cal_fraction': config.VAL_CAL_FRACTION
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
    
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputation_values = load_preprocessed_data(config)
    # Clean labels and align features for each split
    X_train, y_train = _clean_features_and_labels(X_train, y_train, expected_binary=True)
    X_val, y_val = _clean_features_and_labels(X_val, y_val, expected_binary=True)
    X_test, y_test = _clean_features_and_labels(X_test, y_test, expected_binary=True)
    logging.info("Cleaned labels and aligned features for train/val/test; proceeding to splits")
    
    # Split validation into tuning and calibration subsets only if calibration is enabled
    if config.CALIBRATION_ENABLED:
        X_val_tune, X_val_cal, y_val_tune, y_val_cal = split_validation_set(X_val, y_val, config)
    else:
        X_val_tune, y_val_tune = X_val, y_val
        X_val_cal, y_val_cal = None, None
    
    # Tune hyperparameters on val_tune (or full val if calibration disabled)
    best_params = tune_xgboost(X_train, y_train, X_val_tune, y_val_tune, config)
    
    # Train final model on train + val_tune
    base_model = train_final_model(X_train, X_val_tune, y_train, y_val_tune, best_params, config)
    
    # Calibrate and evaluate concisely (includes alignment metrics and plot)
    logging.info("--- FINAL EVALUATION ON TEST SET ---")
    model, results = evaluate_and_calibrate(base_model, X_test, y_test.values, X_val_cal, y_val_cal, config)
    
    # Log results
    auroc = results['auroc']
    auprc = results['auprc']
    brier = results['brier']
    logging.info(f"Test AUROC: {auroc['point_estimate']:.4f} (95% CI: {auroc['ci_lower']:.4f}-{auroc['ci_upper']:.4f})")
    logging.info(f"Test AUPRC: {auprc['point_estimate']:.4f} (95% CI: {auprc['ci_lower']:.4f}-{auprc['ci_upper']:.4f})")
    logging.info(f"Test Brier: {brier['point_estimate']:.4f} (95% CI: {brier['ci_lower']:.4f}-{brier['ci_upper']:.4f}); Calibration used: {results['calibration_alignment'].get('used_method')}")
    
    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
    
    # Save all artifacts
    save_results(model, results, best_params, config, is_calibrated=config.CALIBRATION_ENABLED)
    
    logging.info(f"Analysis completed in {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main() 