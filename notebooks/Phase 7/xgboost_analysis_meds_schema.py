import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from tqdm import tqdm

# --- Configuration ---
# These can be adjusted via command-line arguments
TASK_NAME = 'mort_hosp'
TARGET_VARIABLE = 'boolean_value'
MODEL_NAME = "models/text-embedding-004"
N_OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 600  # seconds
USE_GPU = False # Set to False if you don't have a CUDA-enabled GPU
DRY_RUN = False
DRY_RUN_SUBSET_SIZE = 100
SEED = 42

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate an XGBoost model on patient embeddings for mortality prediction.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/meds_cohort_split_filtered"),
        help="The root directory of the processed MEDS dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/meds_cohort_split_filtered/results/" + TASK_NAME),
        help="Directory to save models, results, and logs."
    )
    return parser.parse_args()

def load_data_for_split(embedding_dir: Path, label_path: Path, split_name: str):
    """Loads embeddings and labels for a single data split (train, val, or test)."""
    logging.info(f"Loading data for '{split_name}' split...")
    
    # Load labels
    y_df = pd.read_parquet(label_path)
    y_df = y_df.set_index('patient_id')
    
    if DRY_RUN:
        y_df = y_df.sample(n=DRY_RUN_SUBSET_SIZE, random_state=SEED)

    # Load corresponding embeddings
    embedding_vectors = []
    valid_indices = []
    for patient_id in tqdm(y_df.index, desc=f"Loading {split_name} embeddings"):
        filepath = embedding_dir / f"{patient_id}.npy"
        if filepath.exists():
            embedding_vectors.append(np.load(filepath))
            valid_indices.append(patient_id)
        else:
            logging.warning(f"Embedding not found for patient_id {patient_id} in '{split_name}'. Skipping.")

    X = np.vstack(embedding_vectors)
    y = y_df.loc[valid_indices][TARGET_VARIABLE]
    
    return X, y

def tune_xgboost(X_train, y_train, X_val, y_val, output_dir: Path):
    """Tunes XGBoost hyperparameters using Optuna."""
    study_path = output_dir / f'optuna_study_{TARGET_VARIABLE}.pkl'
    
    if study_path.exists():
        logging.info(f"Loading existing Optuna study: {study_path}")
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        logging.info(f"Study loaded with {len(study.trials)} trials. Best value: {study.best_value:.4f}")
        return study.best_params

    logging.info("Creating new Optuna study...")
    study = optuna.create_study(direction='maximize')
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'scale_pos_weight': scale_pos_weight, 'random_state': SEED, 'n_jobs': -1,
            'tree_method': 'gpu_hist' if USE_GPU else 'auto',
            'predictor': 'gpu_predictor' if USE_GPU else 'auto',
            'eval_metric': 'auc',
            'early_stopping_rounds': 50
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
    
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    logging.info(f"Study saved to {study_path}. Best params: {study.best_params}")
    
    return study.best_params

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Trains the final model on combined training and validation data."""
    logging.info("Training final model on combined train+validation data...")
    X_full = np.vstack([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    
    final_params = best_params.copy()
    final_params['n_estimators'] = 1000 # Use a high number, will be controlled by early stopping if needed
    final_params['scale_pos_weight'] = (y_full == 0).sum() / (y_full == 1).sum() if (y_full == 1).sum() > 0 else 1
    if USE_GPU:
        final_params['tree_method'] = 'gpu_hist'
        final_params['predictor'] = 'gpu_predictor'
        
    model = xgb.XGBClassifier(**final_params, random_state=SEED, n_jobs=-1)
    model.fit(X_full, y_full, verbose=False)
    return model

def bootstrap_metric(y_true, y_pred_proba, metric_func, n_bootstrap=1000):
    """Computes a metric with bootstrap confidence intervals."""
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) > 1:
            scores.append(metric_func(y_true[indices], y_pred_proba[indices]))
    
    point_estimate = metric_func(y_true, y_pred_proba)
    ci_lower = np.percentile(scores, 2.5)
    ci_upper = np.percentile(scores, 97.5)
    return {'point_estimate': point_estimate, 'ci_lower': ci_lower, 'ci_upper': ci_upper}

def main():
    """Main execution function."""
    args = get_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Setup Logging ---
    log_file = args.output_dir / 'training_log.txt'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"--- Starting Mortality Prediction Training ---")
    logging.info(f"Data Directory: {args.data_dir}")
    logging.info(f"Output Directory: {args.output_dir}")
    
    # --- Load Data ---
    model_name_safe = MODEL_NAME.replace('/', '_')
    embedding_dir_root = args.data_dir / f"embeddings_{model_name_safe}"
    label_dir_root = args.data_dir / "tasks" / TASK_NAME

    try:
        X_train, y_train = load_data_for_split(embedding_dir_root / 'train', label_dir_root / 'train' / 'labels.parquet', 'train')
        X_val, y_val = load_data_for_split(embedding_dir_root / 'val', label_dir_root / 'val' / 'labels.parquet', 'val')
        X_test, y_test = load_data_for_split(embedding_dir_root / 'test', label_dir_root / 'test' / 'labels.parquet', 'test')
    except FileNotFoundError as e:
        sys.exit(f"Fatal: Could not load data. Ensure conversion and embedding scripts have been run. Error: {e}")

    # --- Tune and Train ---
    best_params = tune_xgboost(X_train, y_train, X_val, y_val, args.output_dir)
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    # --- Evaluate ---
    logging.info(f"--- FINAL EVALUATION ON TEST SET ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    results = {
        'auroc': bootstrap_metric(y_test, y_pred_proba, roc_auc_score),
        'auprc': bootstrap_metric(y_test, y_pred_proba, average_precision_score),
        'classification_report': classification_report(y_test, y_pred_binary, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_binary).tolist(),
        'best_params': best_params,
    }

    # --- Log and Save Results ---
    auroc = results['auroc']
    auprc = results['auprc']
    logging.info(f"Test AUROC: {auroc['point_estimate']:.4f} (95% CI: {auroc['ci_lower']:.4f}-{auroc['ci_upper']:.4f})")
    logging.info(f"Test AUPRC: {auprc['point_estimate']:.4f} (95% CI: {auprc['ci_lower']:.4f}-{auprc['ci_upper']:.4f})")

    model_path = args.output_dir / f'model_{TARGET_VARIABLE}.pkl'
    results_path = args.output_dir / f'results_{TARGET_VARIABLE}.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"Final model saved to: {model_path}")
    logging.info(f"Final results saved to: {results_path}")
    logging.info("--- Analysis complete. ---")

if __name__ == "__main__":
    main()
