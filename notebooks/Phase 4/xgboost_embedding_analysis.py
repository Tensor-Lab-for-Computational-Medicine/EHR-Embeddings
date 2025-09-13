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
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from config_embedding_analysis_text_embedding_large import Config

# =============================================================================
# DATA HANDLING
# =============================================================================

def load_embedding_data(config, exp_arm):
    """Loads embedding (.npy) and label (.csv) data for a given experimental arm."""
    logging.info(f"Loading data for arm: {exp_arm}...")
    
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
        logging.info(f"Loading {split} embeddings: n={len(y_df_split)}")
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

    logging.info(f"Shapes: train={X_data['X_train'].shape}, val={X_data['X_val'].shape}, test={X_data['X_test'].shape}")
    
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
        'brier': {'point_estimate': brier_score_loss(y_true, y_pred_proba)},
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }

# =============================================================================
# CALIBRATION / ALIGNMENT METRICS & PLOTS
# =============================================================================

def _safe_logit(probabilities, eps=1e-6):
    probs = np.clip(probabilities, eps, 1 - eps)
    return np.log(probs / (1 - probs))

def compute_calibration_metrics(y_true, y_pred_proba, n_bins=15):
    """Compute calibration alignment metrics: ECE, MCE, Brier, slope, intercept, and reliability curve."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_pred_proba).astype(float)

    # Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1], right=False)
    ece = 0.0
    mce = 0.0
    total_count = len(y_prob)
    per_bin = []
    for b in range(n_bins):
        mask = bin_indices == b
        count_b = np.sum(mask)
        if count_b == 0:
            per_bin.append({'bin': b, 'count': 0, 'accuracy': None, 'confidence': None})
            continue
        acc_b = np.mean(y_true[mask])
        conf_b = np.mean(y_prob[mask])
        gap_b = abs(acc_b - conf_b)
        ece += (count_b / total_count) * gap_b
        mce = max(mce, gap_b)
        per_bin.append({'bin': b, 'count': int(count_b), 'accuracy': float(acc_b), 'confidence': float(conf_b)})

    # Brier score
    brier = float(brier_score_loss(y_true, y_prob))

    # Calibration slope/intercept: logistic regression of outcome on logit(prob)
    X = _safe_logit(y_prob).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    lr.fit(X, y_true)
    slope = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_.ravel()[0])

    # Reliability curve (for plotting/inspection)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'slope': slope,
        'intercept': intercept,
        'reliability_curve': {
            'fraction_of_positives': frac_pos.tolist(),
            'mean_predicted_value': mean_pred.tolist()
        },
        'per_bin': per_bin
    }

def plot_reliability_comparison(y_true, probs_before, probs_after, arm, output_dir, n_bins=15, zoom=True):
    """Save a reliability diagram comparing pre- vs post-calibration probabilities."""
    # Use quantile binning so bins have data when probabilities cluster near 0
    frac_b, mean_b = calibration_curve(y_true, probs_before, n_bins=n_bins, strategy='quantile')
    if probs_after is not None:
        frac_a, mean_a = calibration_curve(y_true, probs_after, n_bins=n_bins, strategy='quantile')
    else:
        frac_a, mean_a = None, None

    plt.figure(figsize=(6, 6))
    # Compute zoomed limits
    x_max = float(max(mean_b.max() if len(mean_b) else 0.0, (mean_a.max() if (mean_a is not None and len(mean_a)) else 0.0)))
    y_max = float(max(frac_b.max() if len(frac_b) else 0.0, (frac_a.max() if (frac_a is not None and len(frac_a)) else 0.0)))
    if zoom:
        x_lim = max(0.1, x_max * 1.1) if x_max > 0 else 0.1
        y_lim = max(0.1, y_max * 1.1) if y_max > 0 else 0.1
    else:
        x_lim = 1.0; y_lim = 1.0

    plt.plot([0, x_lim], [0, x_lim], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.plot(mean_b, frac_b, marker='o', label='Before calibration')
    if mean_a is not None:
        plt.plot(mean_a, frac_a, marker='o', label='After calibration')
    plt.xlim(0, x_lim); plt.ylim(0, y_lim)
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Reliability Diagram - {arm}')
    plt.legend(loc='best')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'calibration_reliability_{arm}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path

# =============================================================================
# MODEL TRAINING (Adapted for multiple arms)
# =============================================================================

def _filter_prevalent_cases_if_needed(config, X_train, X_val, X_test, y_train, y_val, y_test):
    """For certain intervention targets, drop prevalent cases (NaNs) and keep arrays aligned."""
    if config.TARGET_VARIABLE not in ['intervention_vent', 'intervention_vaso']:
        return X_train, X_val, X_test, y_train, y_val, y_test

    logging.info(f"Filtering prevalent cases for target: {config.TARGET_VARIABLE}")

    X_train_df = pd.DataFrame(X_train, index=y_train.index)
    X_val_df = pd.DataFrame(X_val, index=y_val.index)
    X_test_df = pd.DataFrame(X_test, index=y_test.index)

    train_original_count = len(y_train)
    train_valid_indices = y_train.dropna().index
    X_train_df = X_train_df.loc[train_valid_indices]
    y_train = y_train.loc[train_valid_indices]
    X_train = X_train_df.values
    logging.info(f"Train: dropped {train_original_count - len(y_train)}; n={len(y_train)}")

    val_original_count = len(y_val)
    val_valid_indices = y_val.dropna().index
    X_val_df = X_val_df.loc[val_valid_indices]
    y_val = y_val.loc[val_valid_indices]
    X_val = X_val_df.values
    logging.info(f"Val: dropped {val_original_count - len(y_val)}; n={len(y_val)}")

    test_original_count = len(y_test)
    test_valid_indices = y_test.dropna().index
    X_test_df = X_test_df.loc[test_valid_indices]
    y_test = y_test.loc[test_valid_indices]
    X_test = X_test_df.values
    logging.info(f"Test: dropped {test_original_count - len(y_test)}; n={len(y_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def tune_xgboost(X_train, y_train, X_val, y_val, config, exp_arm, val_cal_fraction):
    """Tune XGBoost hyperparameters using Optuna for a specific experimental arm."""
    # Include validation tuning size and calibration fraction in the study path to avoid cross-run mixing
    study_path = os.path.join(config.OUTPUT_DIR, f'optuna_study_{exp_arm}_valtune_{len(X_val)}_calfrac_{val_cal_fraction}.pkl')
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

def train_final_model(X_train, X_val_tune, y_train, y_val_tune, best_params, config):
    logging.info("Training final model on combined train+val_tune data...")
    X_full = np.vstack([X_train, X_val_tune])
    y_full = pd.concat([y_train, y_val_tune])
    final_params = best_params.copy()
    final_params['scale_pos_weight'] = (y_full == 0).sum() / (y_full == 1).sum() if (y_full == 1).sum() > 0 else 1
    if config.USE_GPU:
        final_params['tree_method'] = 'gpu_hist'
        final_params['predictor'] = 'gpu_predictor'
    model = xgb.XGBClassifier(**final_params, random_state=config.SEED, n_jobs=-1)
    model.fit(X_full, y_full, verbose=False)
    return model

def split_validation_set(X_val, y_val, seed, val_cal_fraction):
    """Split validation set into tuning and calibration subsets (stratified)."""
    logging.info(f"Split val: tune={(1 - val_cal_fraction):.2f}, cal={val_cal_fraction:.2f}")
    X_val_tune, X_val_cal, y_val_tune, y_val_cal = train_test_split(
        X_val,
        y_val,
        test_size=val_cal_fraction,
        random_state=seed,
        stratify=y_val
    )
    logging.info(f"val_tune={X_val_tune.shape}, val_cal={X_val_cal.shape}")
    return X_val_tune, X_val_cal, y_val_tune, y_val_cal

def calibrate_model(fitted_model, X_val_cal, y_val_cal, method, enabled):
    """Calibrate a pre-fitted classifier using held-out calibration data."""
    if not enabled:
        logging.info("Calibration disabled. Skipping calibration step.")
        return fitted_model
    logging.info(f"Calibrating model using method='{method}' on val_cal...")
    try:
        # Newer scikit-learn uses 'estimator'
        calibrator = CalibratedClassifierCV(estimator=fitted_model, method=method, cv='prefit')
    except TypeError:
        # Older scikit-learn uses 'base_estimator'
        calibrator = CalibratedClassifierCV(base_estimator=fitted_model, method=method, cv='prefit')
    calibrator.fit(X_val_cal, y_val_cal)
    logging.info("Calibration done.")
    return calibrator

def evaluate_and_calibrate(arm, base_model, X_test, y_test_values, X_val_cal, y_val_cal, calibration_method, calibration_enabled, output_dir):
    """Run calibration (optional), compute metrics before/after, produce plot, and compose results dict."""
    model = calibrate_model(base_model, X_val_cal, y_val_cal, calibration_method, calibration_enabled) if calibration_enabled else base_model

    y_pred_proba_before = base_model.predict_proba(X_test)[:, 1]
    y_pred_proba_after = model.predict_proba(X_test)[:, 1]

    results = evaluate_with_uncertainty(y_test_values, y_pred_proba_after)

    cal_before = compute_calibration_metrics(y_test_values, y_pred_proba_before)
    cal_after = compute_calibration_metrics(y_test_values, y_pred_proba_after)
    ece_improvement = cal_before['ece'] - cal_after['ece']
    mce_improvement = cal_before['mce'] - cal_after['mce']
    brier_improvement = cal_before['brier'] - cal_after['brier']
    slope_delta_to_one = abs(1.0 - cal_before['slope']) - abs(1.0 - cal_after['slope'])
    intercept_delta_to_zero = abs(cal_before['intercept']) - abs(cal_after['intercept'])

    plot_path = plot_reliability_comparison(y_test_values, y_pred_proba_before, y_pred_proba_after, arm, output_dir)

    logging.info(
        f"Calibration alignment for {arm}: "
        f"ECE {cal_before['ece']:.4f}->{cal_after['ece']:.4f} (delta {ece_improvement:.4f}); "
        f"MCE {cal_before['mce']:.4f}->{cal_after['mce']:.4f} (delta {mce_improvement:.4f}); "
        f"Brier {cal_before['brier']:.4f}->{cal_after['brier']:.4f} (delta {brier_improvement:.4f}); "
        f"Slope {cal_before['slope']:.3f}->{cal_after['slope']:.3f} (delta toward 1 {slope_delta_to_one:.3f}); "
        f"Intercept {cal_before['intercept']:.3f}->{cal_after['intercept']:.3f} (delta toward 0 {intercept_delta_to_zero:.3f})"
    )

    results['calibration_alignment'] = {
        'before': cal_before,
        'after': cal_after,
        'improvement': {
            'ece_delta': ece_improvement,
            'mce_delta': mce_improvement,
            'brier_delta': brier_improvement,
            'slope_delta_toward_1': slope_delta_to_one,
            'intercept_delta_toward_0': intercept_delta_to_zero
        },
        'reliability_plot_path': plot_path
    }

    return model, results

def save_results(model, results, best_params, config, exp_arm, is_calibrated=False, calibration_method=None, val_cal_fraction=None):
    """Save model and results for a specific experimental arm."""
    # Save model in multiple formats for compatibility
    model_suffix = '_calibrated' if is_calibrated else ''
    model_path_pkl = os.path.join(config.OUTPUT_DIR, f'model_{exp_arm}{model_suffix}.pkl')
    model_path_json = os.path.join(config.OUTPUT_DIR, f'model_{exp_arm}{model_suffix}.json')
    
    # Try to save in pickle format first
    try:
        with open(model_path_pkl, 'wb') as f: 
            pickle.dump(model, f)
        logging.info(f"Model for {exp_arm} saved to: {model_path_pkl}")
    except Exception as e:
        logging.warning(f"Failed to save pickle format for {exp_arm}: {e}")
    
    # Save in XGBoost native JSON format (more compatible)
    try:
        # Only attempt XGBoost native JSON save if the model exposes save_model (i.e., not CalibratedClassifierCV)
        if hasattr(model, 'save_model'):
            model.save_model(model_path_json)
            logging.info(f"Model for {exp_arm} saved to: {model_path_json}")
        else:
            logging.info(f"Skipping JSON save for {exp_arm} (not an XGBoost estimator).")
    except Exception as e:
        logging.warning(f"Failed to save JSON format for {exp_arm}: {e}")
    
    # Also save hyperparameters separately for easy access
    params_path = os.path.join(config.OUTPUT_DIR, f'params_{exp_arm}.pkl')
    try:
        with open(params_path, 'wb') as f:
            pickle.dump(best_params, f)
        logging.info(f"Parameters for {exp_arm} saved to: {params_path}")
    except Exception as e:
        logging.warning(f"Failed to save parameters for {exp_arm}: {e}")
    
    results_dict = {
        'experimental_arm': exp_arm,
        'model_name': f'XGBoost on Embedding ({exp_arm})',
        **results,
        'calibration_enabled': is_calibrated,
        'calibration_method': calibration_method if is_calibrated else None,
        'val_cal_fraction': val_cal_fraction
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, f'results_{exp_arm}.pkl')
    with open(results_path, 'wb') as f: pickle.dump(results_dict, f)
    
    logging.info(f"Results for {exp_arm} saved to: {results_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the analysis across all experimental arms."""
    config = Config()
    # Backward-compatible calibration config fallbacks
    CALIBRATION_ENABLED = getattr(config, 'CALIBRATION_ENABLED', True)
    CALIBRATION_METHOD = getattr(config, 'CALIBRATION_METHOD', 'isotonic')  # 'isotonic' or 'sigmoid'
    VAL_CAL_FRACTION = getattr(config, 'VAL_CAL_FRACTION', 0.5)
    
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

            # Handle prevalent cases if needed
            X_train, X_val, X_test, y_train, y_val, y_test = _filter_prevalent_cases_if_needed(
                config, X_train, X_val, X_test, y_train, y_val, y_test
            )

            # Split validation into tuning and calibration subsets only if calibration is enabled
            if CALIBRATION_ENABLED:
                X_val_tune, X_val_cal, y_val_tune, y_val_cal = split_validation_set(X_val, y_val, config.SEED, VAL_CAL_FRACTION)
            else:
                X_val_tune, y_val_tune = X_val, y_val
                X_val_cal, y_val_cal = None, None

            # Tune on val_tune
            best_params = tune_xgboost(
                X_train,
                y_train,
                X_val_tune,
                y_val_tune,
                config,
                arm,
                VAL_CAL_FRACTION if CALIBRATION_ENABLED else 'nocal'
            )

            # Train on train + val_tune
            base_model = train_final_model(X_train, X_val_tune, y_train, y_val_tune, best_params, config)

            # Calibrate + evaluate
            logging.info(f"--- FINAL EVALUATION ON TEST SET FOR {arm} ---")
            model, results = evaluate_and_calibrate(
                arm,
                base_model,
                X_test,
                y_test.values,
                X_val_cal,
                y_val_cal,
                CALIBRATION_METHOD,
                CALIBRATION_ENABLED,
                config.OUTPUT_DIR,
            )

            auroc = results['auroc']
            logging.info(f"Test AUROC for {arm}: {auroc['point_estimate']:.4f} (95% CI: {auroc['ci_lower']:.4f}-{auroc['ci_upper']:.4f})")

            save_results(
                model,
                results,
                best_params,
                config,
                arm,
                is_calibrated=CALIBRATION_ENABLED,
                calibration_method=CALIBRATION_METHOD if CALIBRATION_ENABLED else None,
                val_cal_fraction=VAL_CAL_FRACTION
            )
        except Exception as e:
            logging.error(f"!!! An error occurred while processing arm {arm}: {e}")
            logging.error(f"Skipping arm {arm} and continuing to the next one.")
            continue

    logging.info(f"\nFull analysis completed in {(time.time() - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()