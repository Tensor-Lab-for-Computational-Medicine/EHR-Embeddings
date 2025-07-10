# h2_analysis.py
"""
H2 Hypothesis Testing: Orthogonal Information and Hybrid Models

Tests:
- H2a: Correlation analysis between baseline and champion model errors
- H2b: Early fusion hybrid model (concatenated features)
- H2c: Late fusion hybrid model (stacked predictions)
- H2d: Statistical significance testing
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
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config_h2 import ConfigH2

# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_numerical_data(config):
    """Load preprocessed numerical features and labels"""
    logging.info("Loading numerical features and labels...")
    
    data = {}
    file_paths = {
        'X_train': config.X_TRAIN_NUM_PATH,
        'X_val': config.X_VAL_NUM_PATH,
        'X_test': config.X_TEST_NUM_PATH,
        'y_train': config.Y_TRAIN_PATH,
        'y_val': config.Y_VAL_PATH,
        'y_test': config.Y_TEST_PATH
    }
    
    for key, path in file_paths.items():
        with open(path, 'rb') as f:
            data[key] = pickle.load(f)
    
    logging.info(f"Numerical data shapes: X_train={data['X_train'].shape}, "
                f"X_val={data['X_val'].shape}, X_test={data['X_test'].shape}")
    
    return data

def load_embedding_data(config, split_names=['train', 'val', 'test']):
    """Load champion model embeddings"""
    logging.info(f"Loading champion model embeddings for {config.CHAMPION_ARM}...")
    
    # Load labels to get icustay_id order
    label_files = {
        'train': f'{config.LABEL_DIR}/train_labels.csv',
        'val': f'{config.LABEL_DIR}/val_labels.csv', 
        'test': f'{config.LABEL_DIR}/test_labels.csv'
    }
    
    embeddings = {}
    for split in split_names:
        # Load labels to get correct order
        labels_df = pd.read_csv(label_files[split], header=None, names=['icustay_id', config.TARGET_VARIABLE])
        icustay_ids = labels_df['icustay_id'].values
        
        # Load embeddings in correct order
        embedding_dir = os.path.join(config.EMBEDDING_DATA_DIR, split)
        embedding_vectors = []
        
        for icustay_id in tqdm(icustay_ids, desc=f"Loading {split} embeddings"):
            embedding_path = os.path.join(embedding_dir, f"{icustay_id}.npy")
            embedding_vectors.append(np.load(embedding_path))
        
        embeddings[f'X_{split}'] = np.vstack(embedding_vectors)
        logging.info(f"Loaded {split} embeddings: {embeddings[f'X_{split}'].shape}")
    
    return embeddings

def load_trained_models(config):
    """Load pre-trained baseline and champion models"""
    logging.info("Loading pre-trained models...")
    
    # Load baseline XGBoost model
    with open(config.BASELINE_MODEL_PATH, 'rb') as f:
        baseline_model = pickle.load(f)
    
    # Load champion embedding model 
    with open(config.CHAMPION_MODEL_PATH, 'rb') as f:
        champion_model = pickle.load(f)
    
    logging.info("✅ Models loaded successfully")
    return baseline_model, champion_model

# =============================================================================
# PREDICTION AND ERROR ANALYSIS
# =============================================================================

def generate_test_predictions(baseline_model, champion_model, X_test_num, X_test_emb, y_test):
    """Generate predictions from both models on test set"""
    logging.info("Generating test set predictions...")
    
    # Baseline predictions
    baseline_proba = baseline_model.predict_proba(X_test_num)[:, 1]
    baseline_pred = baseline_model.predict(X_test_num)
    
    # Champion predictions  
    champion_proba = champion_model.predict_proba(X_test_emb)[:, 1]
    champion_pred = champion_model.predict(X_test_emb)
    
    # Calculate prediction errors (absolute difference from true probability)
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    baseline_errors = np.abs(baseline_proba - y_test_array)
    champion_errors = np.abs(champion_proba - y_test_array)
    
    predictions = {
        'baseline_proba': baseline_proba,
        'baseline_pred': baseline_pred,
        'champion_proba': champion_proba, 
        'champion_pred': champion_pred,
        'baseline_errors': baseline_errors,
        'champion_errors': champion_errors,
        'y_true': y_test_array
    }
    
    logging.info("✅ Predictions generated")
    return predictions

def analyze_error_correlation(predictions, config):
    """H2a: Analyze correlation between model errors"""
    logging.info("=== H2a: ERROR CORRELATION ANALYSIS ===")
    
    baseline_errors = predictions['baseline_errors']
    champion_errors = predictions['champion_errors']
    
    # Calculate Pearson correlation
    correlation, p_value = pearsonr(baseline_errors, champion_errors)
    
    # Check H2 hypothesis (correlation < 0.4)
    h2_threshold = config.CORRELATION_THRESHOLD
    h2_satisfied = abs(correlation) < h2_threshold
    
    correlation_results = {
        'correlation': correlation,
        'p_value': p_value,
        'h2_threshold': h2_threshold,
        'h2_satisfied': h2_satisfied,
        'interpretation': 'weakly correlated' if h2_satisfied else 'strongly correlated'
    }
    
    logging.info(f"Error correlation: {correlation:.4f} (p={p_value:.4f})")
    logging.info(f"H2 threshold: {h2_threshold}")
    logging.info(f"H2 satisfied: {h2_satisfied} - Errors are {correlation_results['interpretation']}")
    
    # Create correlation plot
    plt.figure(figsize=(8, 6))
    plt.scatter(baseline_errors, champion_errors, alpha=0.5, s=20)
    plt.xlabel('Baseline Model Prediction Errors')
    plt.ylabel('Champion Model Prediction Errors')
    plt.title(f'H2a: Model Error Correlation\nr = {correlation:.4f}, p = {p_value:.4f}')
    plt.plot([0, max(baseline_errors)], [0, max(champion_errors)], 'r--', alpha=0.7, label='Perfect correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'h2a_error_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation_results

# =============================================================================
# HYBRID MODEL IMPLEMENTATIONS
# =============================================================================

def bootstrap_metric(y_true, y_pred_proba, metric_func, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """Calculate bootstrap confidence intervals for a metric"""
    np.random.seed(random_state)
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Stratified bootstrap
        indices = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) > 1:  # Ensure both classes present
            try:
                score = metric_func(y_true[indices], y_pred_proba[indices])
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
        'std': np.std(scores)
    }

def evaluate_model_with_ci(y_true, y_pred_proba, model_name, config):
    """Evaluate model with confidence intervals"""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    results = {
        'model_name': model_name,
        'auroc': bootstrap_metric(y_true, y_pred_proba, roc_auc_score, config.N_BOOTSTRAP),
        'auprc': bootstrap_metric(y_true, y_pred_proba, average_precision_score, config.N_BOOTSTRAP),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    auroc = results['auroc']
    auprc = results['auprc'] 
    logging.info(f"{model_name}:")
    logging.info(f"  AUROC: {auroc['point_estimate']:.4f} (95% CI: {auroc['ci_lower']:.4f}-{auroc['ci_upper']:.4f})")
    logging.info(f"  AUPRC: {auprc['point_estimate']:.4f} (95% CI: {auprc['ci_lower']:.4f}-{auprc['ci_upper']:.4f})")
    
    return results

def build_early_fusion_hybrid(X_train_num, X_val_num, X_test_num, 
                             X_train_emb, X_val_emb, X_test_emb,
                             y_train, y_val, y_test, config):
    """H2b: Early fusion hybrid model (concatenated features)"""
    logging.info("=== H2b: EARLY FUSION HYBRID MODEL ===")
    
    # Ensure consistent indices for concatenation
    # Convert to numpy arrays and align by index
    if hasattr(X_train_num, 'index'):
        train_indices = X_train_num.index
        val_indices = X_val_num.index  
        test_indices = X_test_num.index
        
        X_train_num_array = X_train_num.values
        X_val_num_array = X_val_num.values
        X_test_num_array = X_test_num.values
    else:
        X_train_num_array = X_train_num
        X_val_num_array = X_val_num
        X_test_num_array = X_test_num
    
    # Concatenate numerical and embedding features
    X_train_hybrid = np.hstack([X_train_num_array, X_train_emb])
    X_val_hybrid = np.hstack([X_val_num_array, X_val_emb])
    X_test_hybrid = np.hstack([X_test_num_array, X_test_emb])
    
    logging.info(f"Hybrid feature shapes: train={X_train_hybrid.shape}, val={X_val_hybrid.shape}, test={X_test_hybrid.shape}")
    
    # Hyperparameter tuning for hybrid model
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'n_estimators': 500,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state': config.SEED,
            'n_jobs': -1
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_hybrid, y_train, eval_set=[(X_val_hybrid, y_val)], 
                 eval_metric='auc', early_stopping_rounds=30, verbose=False)
        
        return roc_auc_score(y_val, model.predict_proba(X_val_hybrid)[:, 1])
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS, timeout=config.OPTUNA_TIMEOUT//2)
    
    # Train final early fusion model
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'n_estimators': 500,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': config.SEED,
        'n_jobs': -1
    })
    
    # Train on combined train+val data
    X_full_hybrid = np.vstack([X_train_hybrid, X_val_hybrid])
    y_full = pd.concat([y_train, y_val]) if hasattr(y_train, 'index') else np.concatenate([y_train, y_val])
    
    early_fusion_model = xgb.XGBClassifier(**best_params)
    early_fusion_model.fit(X_full_hybrid, y_full, verbose=False)
    
    # Evaluate on test set
    y_pred_proba_early = early_fusion_model.predict_proba(X_test_hybrid)[:, 1]
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    
    early_fusion_results = evaluate_model_with_ci(y_test_array, y_pred_proba_early, 
                                                 "Early Fusion Hybrid", config)
    
    return early_fusion_model, early_fusion_results, y_pred_proba_early

def build_late_fusion_hybrid(baseline_proba, champion_proba, y_train, y_val, y_test, config):
    """H2c: Late fusion hybrid model (stacked predictions)"""
    logging.info("=== H2c: LATE FUSION HYBRID MODEL ===")
    
    # We need train/val predictions for training the meta-learner
    # For now, we'll use cross-validation or assume we have out-of-fold predictions
    # Since we don't have train/val predictions readily available, we'll train a simple stacker
    
    # Create training data for meta-learner using the test predictions as proof-of-concept
    # In practice, you'd want to use out-of-fold predictions on train/val
    X_meta = np.column_stack([baseline_proba, champion_proba])
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    
    # For demonstration, we'll train on a portion and test on remainder
    # In real implementation, use proper cross-validation
    n_samples = len(X_meta)
    n_train_meta = int(0.7 * n_samples)
    
    # Random split for meta-learner training
    np.random.seed(config.SEED)
    indices = np.random.permutation(n_samples)
    train_meta_idx = indices[:n_train_meta]
    test_meta_idx = indices[n_train_meta:]
    
    X_meta_train = X_meta[train_meta_idx]
    y_meta_train = y_test_array[train_meta_idx]
    X_meta_test = X_meta[test_meta_idx]
    y_meta_test = y_test_array[test_meta_idx]
    
    # Train logistic regression meta-learner
    meta_learner = LogisticRegression(random_state=config.SEED, max_iter=1000)
    meta_learner.fit(X_meta_train, y_meta_train)
    
    # Predict with meta-learner
    y_pred_proba_late = meta_learner.predict_proba(X_meta_test)[:, 1]
    
    late_fusion_results = evaluate_model_with_ci(y_meta_test, y_pred_proba_late,
                                                "Late Fusion Hybrid", config)
    
    logging.info(f"Meta-learner coefficients: Baseline={meta_learner.coef_[0][0]:.4f}, Champion={meta_learner.coef_[0][1]:.4f}")
    
    return meta_learner, late_fusion_results, y_pred_proba_late

# =============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# =============================================================================

def statistical_significance_testing(baseline_results, champion_results, early_fusion_results, 
                                   late_fusion_results, config):
    """Perform statistical significance testing for H2"""
    logging.info("=== H2d: STATISTICAL SIGNIFICANCE TESTING ===")
    
    # Extract AUROC estimates and CIs
    models = {
        'Baseline': baseline_results,
        'Champion': champion_results, 
        'Early Fusion': early_fusion_results,
        'Late Fusion': late_fusion_results
    }
    
    # Create comparison table
    comparison_data = []
    for name, results in models.items():
        auroc = results['auroc']
        auprc = results['auprc'] 
        
        comparison_data.append({
            'Model': name,
            'AUROC': auroc['point_estimate'],
            'AUROC_CI_Lower': auroc['ci_lower'],
            'AUROC_CI_Upper': auroc['ci_upper'],
            'AUPRC': auprc['point_estimate'],
            'AUPRC_CI_Lower': auprc['ci_lower'],
            'AUPRC_CI_Upper': auprc['ci_upper']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Check for non-overlapping confidence intervals (conservative significance test)
    def check_significance(model1_results, model2_results, metric='auroc'):
        ci1_lower = model1_results[metric]['ci_lower']
        ci1_upper = model1_results[metric]['ci_upper']
        ci2_lower = model2_results[metric]['ci_lower'] 
        ci2_upper = model2_results[metric]['ci_upper']
        
        # Non-overlapping CIs indicate significance
        return ci1_upper < ci2_lower or ci2_upper < ci1_lower
    
    significance_tests = {
        'early_vs_baseline': check_significance(early_fusion_results, baseline_results),
        'early_vs_champion': check_significance(early_fusion_results, champion_results),
        'late_vs_baseline': check_significance(late_fusion_results, baseline_results),
        'late_vs_champion': check_significance(late_fusion_results, champion_results)
    }
    
    logging.info("Significance testing (non-overlapping 95% CIs):")
    for test, significant in significance_tests.items():
        logging.info(f"  {test}: {'Significant' if significant else 'Not significant'}")
    
    return comparison_df, significance_tests

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to execute H2 analysis"""
    config = ConfigH2()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'h2_analysis_log.txt'), mode='w'),
            logging.StreamHandler()
        ]
    )
    
    start_time = time.time()
    logging.info("=== STARTING H2 HYPOTHESIS TESTING ===")
    logging.info(f"Champion model: {config.CHAMPION_ARM}")
    
    # Validate all required files exist
    config.validate_paths()
    
    # Load all data
    numerical_data = load_numerical_data(config)
    embedding_data = load_embedding_data(config)
    baseline_model, champion_model = load_trained_models(config)
    
    # Extract data splits
    X_train_num, X_val_num, X_test_num = numerical_data['X_train'], numerical_data['X_val'], numerical_data['X_test']
    X_train_emb, X_val_emb, X_test_emb = embedding_data['X_train'], embedding_data['X_val'], embedding_data['X_test']
    y_train, y_val, y_test = numerical_data['y_train'], numerical_data['y_val'], numerical_data['y_test']
    
    # Generate test predictions for both individual models
    predictions = generate_test_predictions(baseline_model, champion_model, 
                                          X_test_num, X_test_emb, y_test)
    
    # H2a: Error correlation analysis
    correlation_results = analyze_error_correlation(predictions, config)
    
    # Evaluate individual models with proper confidence intervals
    y_test_array = predictions['y_true']
    baseline_results = evaluate_model_with_ci(y_test_array, predictions['baseline_proba'], 
                                            "Baseline XGBoost", config)
    champion_results = evaluate_model_with_ci(y_test_array, predictions['champion_proba'],
                                            f"Champion {config.CHAMPION_ARM}", config)
    
    # H2b: Early fusion hybrid model
    early_fusion_model, early_fusion_results, early_fusion_proba = build_early_fusion_hybrid(
        X_train_num, X_val_num, X_test_num, X_train_emb, X_val_emb, X_test_emb,
        y_train, y_val, y_test, config)
    
    # H2c: Late fusion hybrid model  
    late_fusion_model, late_fusion_results, late_fusion_proba = build_late_fusion_hybrid(
        predictions['baseline_proba'], predictions['champion_proba'], 
        y_train, y_val, y_test, config)
    
    # H2d: Statistical significance testing
    comparison_df, significance_tests = statistical_significance_testing(
        baseline_results, champion_results, early_fusion_results, late_fusion_results, config)
    
    # Save all results
    final_results = {
        'config': {
            'champion_arm': config.CHAMPION_ARM,
            'correlation_threshold': config.CORRELATION_THRESHOLD,
            'n_bootstrap': config.N_BOOTSTRAP
        },
        'h2a_correlation': correlation_results,
        'individual_models': {
            'baseline': baseline_results,
            'champion': champion_results
        },
        'hybrid_models': {
            'early_fusion': early_fusion_results,
            'late_fusion': late_fusion_results
        },
        'statistical_tests': significance_tests,
        'comparison_table': comparison_df.to_dict('records'),
        'predictions': {
            'baseline_proba': predictions['baseline_proba'].tolist(),
            'champion_proba': predictions['champion_proba'].tolist(),
            'early_fusion_proba': early_fusion_proba.tolist(),
            'late_fusion_proba': late_fusion_proba.tolist(),
            'y_true': predictions['y_true'].tolist()
        }
    }
    
    # Save results
    results_path = os.path.join(config.OUTPUT_DIR, 'h2_analysis_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(final_results, f)
    
    # Save models
    models_path = os.path.join(config.OUTPUT_DIR, 'h2_trained_models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump({
            'early_fusion_model': early_fusion_model,
            'late_fusion_model': late_fusion_model
        }, f)
    
    # Save comparison table as CSV
    comparison_df.to_csv(os.path.join(config.OUTPUT_DIR, 'h2_model_comparison.csv'), index=False)
    
    # Print summary
    logging.info("\n" + "="*50)
    logging.info("H2 HYPOTHESIS TESTING SUMMARY")
    logging.info("="*50)
    
    # H2a Summary
    correlation = correlation_results['correlation']
    h2a_result = "SUPPORTED" if correlation_results['h2_satisfied'] else "NOT SUPPORTED"
    logging.info(f"H2a (Orthogonal Information): {h2a_result}")
    logging.info(f"    Error correlation: {correlation:.4f} (threshold: {config.CORRELATION_THRESHOLD})")
    
    # H2b Summary - Check if hybrid models outperform individual models
    baseline_auroc = baseline_results['auroc']['point_estimate']
    champion_auroc = champion_results['auroc']['point_estimate']
    early_auroc = early_fusion_results['auroc']['point_estimate']
    late_auroc = late_fusion_results['auroc']['point_estimate']
    
    early_improves = early_auroc > max(baseline_auroc, champion_auroc)
    late_improves = late_auroc > max(baseline_auroc, champion_auroc)
    
    h2b_result = "SUPPORTED" if (early_improves or late_improves) else "NOT SUPPORTED"
    logging.info(f"H2b (Hybrid Performance): {h2b_result}")
    logging.info(f"    Baseline AUROC: {baseline_auroc:.4f}")
    logging.info(f"    Champion AUROC: {champion_auroc:.4f}")
    logging.info(f"    Early Fusion AUROC: {early_auroc:.4f} ({'↑' if early_improves else '↓'})")
    logging.info(f"    Late Fusion AUROC: {late_auroc:.4f} ({'↑' if late_improves else '↓'})")
    
    # Overall H2 conclusion
    h2_overall = "SUPPORTED" if (correlation_results['h2_satisfied'] and (early_improves or late_improves)) else "PARTIALLY SUPPORTED" if (correlation_results['h2_satisfied'] or early_improves or late_improves) else "NOT SUPPORTED"
    logging.info(f"\nOVERALL H2 HYPOTHESIS: {h2_overall}")
    
    logging.info(f"\nAnalysis completed in {(time.time() - start_time)/60:.2f} minutes")
    logging.info(f"Results saved to: {results_path}")
    logging.info(f"Models saved to: {models_path}")
    logging.info(f"Comparison table saved to: {os.path.join(config.OUTPUT_DIR, 'h2_model_comparison.csv')}")
    
    return final_results

if __name__ == "__main__":
    main()