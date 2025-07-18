"""
Tunes and trains a Neural Network on text embedding data using KerasTuner
for all 18 experimental conditions (F1/F2/F3 x P0-P5).
This version tunes the number of layers, neurons, dropout, and learning rate.
Includes a patch for older Python versions missing math.prod().
"""
import math
import numpy as np

# =============================================================================
# MONKEY-PATCH for older Python versions (< 3.8)
# This adds the math.prod function if it's missing, which KerasTuner requires.
# This must be at the top of the script.
# =============================================================================
if not hasattr(math, 'prod'):
    from functools import reduce
    import operator
    def _prod(iterable):
        return reduce(operator.mul, iterable, 1)
    math.prod = _prod
# =============================================================================

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import keras_tuner as kt
import logging
import time
import os
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
from tqdm import tqdm
from config_embedding_analysis_text_embedding_large import Config

# =============================================================================
# DATA HANDLING (Unchanged)
# =============================================================================

def load_embedding_data(config, exp_arm):
    """Loads embedding (.npy) and label (.csv) data for a given experimental arm."""
    logging.info(f"Loading data for experimental arm: {exp_arm}...")
    
    y_train_df = pd.read_csv(os.path.join(config.LABEL_DIR, f'{config.TARGET_VARIABLE}_train_labels.csv'), header=0, index_col=0)
    y_val_df = pd.read_csv(os.path.join(config.LABEL_DIR, f'{config.TARGET_VARIABLE}_val_labels.csv'), header=0, index_col=0)
    y_test_df = pd.read_csv(os.path.join(config.LABEL_DIR, f'{config.TARGET_VARIABLE}_test_labels.csv'), header=0, index_col=0)

    if config.DRY_RUN:
        logging.warning(f"DRY RUN: Subsetting data to {config.DRY_RUN_SUBSET_SIZE} stratified samples per split.")
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
    
    return X_data['X_train'], X_data['X_val'], X_data['X_test'], y_train_df, y_val_df, y_test_df

# =============================================================================
# EVALUATION (Unchanged)
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
# MODEL TUNING AND TRAINING (MODIFIED to tune num_layers)
# =============================================================================

def find_best_model_and_train(X_train, y_train, X_val, y_val, config, exp_arm):
    """Tunes hyperparameters with KerasTuner and trains a final model."""
    
    input_dim = X_train.shape[1]

    def build_model(hp):
        """Builds a Keras model with a tunable number of layers and units."""
        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        # Tune the number of hidden layers from 1 to 3
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(
                Dense(
                    # Tune the number of units in each layer
                    units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                    activation="relu",
                )
            )
            model.add(Dropout(hp.Float(f"dropout_{i}", 0.1, 0.5, step=0.1)))

        model.add(Dense(1, activation="sigmoid")) # Add the final output layer
        
        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='binary_crossentropy',
                      metrics=[AUC(name='auroc')])
        return model

    logging.info("Starting hyperparameter tuning with KerasTuner...")
    
    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective=kt.Objective("val_auroc", direction="max"),
        max_epochs=30,
        factor=3,
        directory=os.path.join(config.OUTPUT_DIR, 'tuner'),
        project_name=f'tuner_{exp_arm}',
        overwrite=True
    )
    
    early_stopping_tuner = EarlyStopping(monitor='val_loss', patience=5)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))

    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 callbacks=[early_stopping_tuner],
                 class_weight=class_weights_dict)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"Best hyperparameters found for arm {exp_arm}:\n{best_hps.values}")

    logging.info("Training with best HPs to find optimal epochs...")
    temp_model = tuner.hypermodel.build(best_hps)
    early_stopping_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = temp_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stopping_final],
        class_weight=class_weights_dict,
        verbose=0
    )
    
    optimal_epochs = early_stopping_final.best_epoch + 1
    logging.info(f"Optimal number of epochs found: {optimal_epochs}")
    
    logging.info("Training final model on combined train+validation data...")
    X_full = np.vstack([X_train, X_val])
    y_full = pd.concat([y_train, y_val])

    full_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_full), y=y_full)
    full_class_weights_dict = dict(enumerate(full_weights))
    
    final_model = tuner.hypermodel.build(best_hps)
    final_model.fit(
        X_full, y_full,
        epochs=optimal_epochs,
        batch_size=config.BATCH_SIZE,
        class_weight=full_class_weights_dict,
        verbose=2
    )
    
    return final_model, best_hps.values

def save_results(model, results, best_hps, config, exp_arm):
    """Save model, results, and best hyperparameters."""
    model_path = os.path.join(config.OUTPUT_DIR, f'model_{exp_arm}.keras')
    model.save(model_path)
    
    results_dict = {
        'experimental_arm': exp_arm,
        'model_name': f'Tuned Neural Network on Embedding ({exp_arm})',
        'best_hyperparameters': best_hps,
        **results
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, f'results_{exp_arm}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    logging.info(f"Model for {exp_arm} saved to: {model_path}")
    logging.info(f"Results for {exp_arm} saved to: {results_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the analysis across all experimental arms."""
    config = Config()
    
    log_file = os.path.join(config.OUTPUT_DIR, 'nn_tuner_embedding_analysis_log.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    
    start_time = time.time()
    
    logging.info(f"Starting analysis for TARGET_VARIABLE: {config.TARGET_VARIABLE}")
    logging.info(f"Log file will be saved to: {log_file}")

    experimental_arms = [f"{rep}_{p}" for rep in config.REPRESENTATIONS for p in config.PROMPTS]
    if config.DRY_RUN:
        logging.warning("DRY RUN ENABLED: Processing only the first experimental arm on a subset of data.")
        experimental_arms = [experimental_arms[0]]

    for arm in experimental_arms:
        logging.info(f"\n{'='*80}\nSTARTING ANALYSIS FOR ARM: {arm}\n{'='*80}")
        
        try:
            X_train, X_val, X_test, y_train_df, y_val_df, y_test_df = load_embedding_data(config, arm)
            
            y_train = y_train_df[config.TARGET_VARIABLE]
            y_val = y_val_df[config.TARGET_VARIABLE]
            y_test = y_test_df[config.TARGET_VARIABLE]

            if config.TARGET_VARIABLE in ['intervention_vent', 'intervention_vaso']:
                logging.info(f"Handling prevalent cases for target: {config.TARGET_VARIABLE}")
                
                def handle_prevalent_cases(X, y, set_name):
                    original_count = len(y)
                    valid_indices = y.dropna().index
                    X_df = pd.DataFrame(X, index=y.index).loc[valid_indices]
                    y_series = y.loc[valid_indices]
                    logging.info(f"{set_name} set: Dropped {original_count - len(y_series)} prevalent cases. New size: {len(y_series)}")
                    return X_df.values, y_series

                X_train, y_train = handle_prevalent_cases(X_train, y_train, "Train")
                X_val, y_val = handle_prevalent_cases(X_val, y_val, "Validation")
                X_test, y_test = handle_prevalent_cases(X_test, y_test, "Test")

            model, best_hps = find_best_model_and_train(X_train, y_train, X_val, y_val, config, arm)
            
            logging.info(f"--- FINAL EVALUATION ON TEST SET FOR {arm} ---")
            y_pred_proba = model.predict(X_test).flatten()
            results = evaluate_with_uncertainty(y_test.values, y_pred_proba)
            
            auroc = results['auroc']
            logging.info(f"Test AUROC for {arm}: {auroc['point_estimate']:.4f} (95% CI: {auroc['ci_lower']:.4f}-{auroc['ci_upper']:.4f})")
            
            save_results(model, results, best_hps, config, arm)
            
        except Exception as e:
            logging.error(f"!!! An error occurred while processing arm {arm}: {e}", exc_info=True)
            logging.error(f"Skipping arm {arm} and continuing to the next one.")
            continue

    logging.info(f"\nFull analysis completed in {(time.time() - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()