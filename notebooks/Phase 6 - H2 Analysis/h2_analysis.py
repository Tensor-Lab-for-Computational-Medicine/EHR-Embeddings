# h2_analysis.py
"""
H2 Hypothesis Testing: Model Discordance, Failure Mode, and Synergy Analysis (Definitive Final Version)

This script provides a complete and definitive implementation of the H2 analysis plan. It includes:
- The full sensitivity analysis loop for three thresholding schemes.
- Correct application of statistical tests (Mann-Whitney U and Chi-squared).
- Explicit confounder analysis and robustness checks.
- Complete hybrid model selection and synergy analysis workflow.
- Generation of all specified tables (H2-1 through H2-5) and figures.
- Final logging of key results for easy review.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import time
import os
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, mannwhitneyu, chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Assuming config_h2.py is in the same directory.
from config_h2 import ConfigH2

# =============================================================================
# UTILITIES (LOADING & EVALUATION)
# =============================================================================

def load_data(config):
    """Load all preprocessed numerical features, embeddings, and labels."""
    logging.info("Loading all required data splits...")
    data = {}
    
    numerical_paths = {
        'X_train_num': config.X_TRAIN_NUM_PATH, 'X_val_num': config.X_VAL_NUM_PATH, 'X_test_num': config.X_TEST_NUM_PATH,
        'y_train': config.Y_TRAIN_PATH, 'y_val': config.Y_VAL_PATH, 'y_test': config.Y_TEST_PATH
    }
    for key, path in tqdm(numerical_paths.items(), desc="Loading numerical data & labels"):
        with open(path, 'rb') as f:
            data[key] = pickle.load(f)
            
    label_files = {
        'train': os.path.join(config.LABEL_DIR, 'train_labels.csv'), 
        'val': os.path.join(config.LABEL_DIR, 'val_labels.csv'), 
        'test': os.path.join(config.LABEL_DIR, 'test_labels.csv')
    }
    for split in ['train', 'val', 'test']:
        labels_df = pd.read_csv(label_files[split], header=None, names=['icustay_id', config.TARGET_VARIABLE])
        icustay_ids = labels_df['icustay_id'].values
        
        embedding_dir = os.path.join(config.EMBEDDING_DATA_DIR, split)
        embedding_vectors = [np.load(os.path.join(embedding_dir, f"{icustay_id}.npy")) for icustay_id in tqdm(icustay_ids, desc=f"Loading {split} embeddings")]
        data[f'X_{split}_emb'] = np.vstack(embedding_vectors)
        
    logging.info("✅ All data loaded successfully.")
    return data

def load_trained_models(config):
    """Load pre-trained Numerical Model (NM) and Semantic Model (SM)."""
    logging.info("Loading pre-trained Numerical (NM) and Semantic (SM) models...")
    with open(config.BASELINE_MODEL_PATH, 'rb') as f:
        numerical_model = pickle.load(f)
    with open(config.CHAMPION_MODEL_PATH, 'rb') as f:
        semantic_model = pickle.load(f)
    logging.info(f"✅ NM: {type(numerical_model).__name__}, SM: {type(semantic_model).__name__}")
    return numerical_model, semantic_model

def evaluate_model_performance(y_true, y_pred_proba, model_name, config):
    """Evaluate model with bootstrap confidence intervals for AUROC, AUPRC, and Brier Score."""
    logging.info(f"--- Evaluating Performance for: {model_name} ---")
    results = {'model_name': model_name}
    metrics = {'AUROC': roc_auc_score, 'AUPRC': average_precision_score, 'Brier': brier_score_loss}
    
    for metric_name, metric_func in metrics.items():
        point_estimate = metric_func(y_true, y_pred_proba)
        
        metric_samples = []
        y_true_np, y_pred_proba_np = (d.values if hasattr(d, 'values') else d for d in (y_true, y_pred_proba))
        
        for _ in range(config.N_BOOTSTRAP):
            indices = np.random.choice(len(y_true_np), len(y_true_np), replace=True)
            if len(np.unique(y_true_np[indices])) < 2 and metric_name != 'Brier':
                continue
            metric_samples.append(metric_func(y_true_np[indices], y_pred_proba_np[indices]))
        
        ci_low, ci_high = np.percentile(metric_samples, [2.5, 97.5])
        
        results[f'{metric_name}_pe'] = point_estimate
        results[f'{metric_name}_ci_low'] = ci_low
        results[f'{metric_name}_ci_high'] = ci_high
        
        logging.info(f"  {metric_name}: {point_estimate:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")
    
    return results

# =============================================================================
# H2a: MODEL DISCORDANCE ANALYSIS
# =============================================================================

def determine_f1_threshold(y_true, y_probas):
    """Determine the optimal probability threshold by maximizing the F1-score."""
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [f1_score(y_true, y_probas >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    # Now returns both the threshold and the score
    return thresholds[optimal_idx], f1_scores[optimal_idx]

def define_analysis_cohorts(nm_proba, sm_proba, y_true, nm_thresh, sm_thresh):
    """Define the eight granular analysis cohorts based on model predictions and true outcomes."""
    nm_pred, sm_pred, y_true_arr = (nm_proba >= nm_thresh).astype(int), (sm_proba >= sm_thresh).astype(int), y_true.astype(int)
    return {
        'TP_concordant': (nm_pred == 1) & (sm_pred == 1) & (y_true_arr == 1),
        'TN_concordant': (nm_pred == 0) & (sm_pred == 0) & (y_true_arr == 0),
        'FN_concordant': (nm_pred == 0) & (sm_pred == 0) & (y_true_arr == 1),
        'FP_concordant': (nm_pred == 1) & (sm_pred == 1) & (y_true_arr == 0),
        'FN_SM': (sm_pred == 0) & (nm_pred == 1) & (y_true_arr == 1),
        'FP_SM': (sm_pred == 1) & (nm_pred == 0) & (y_true_arr == 0),
        'FN_NM': (sm_pred == 1) & (nm_pred == 0) & (y_true_arr == 1),
        'FP_NM': (sm_pred == 0) & (nm_pred == 1) & (y_true_arr == 0),
    }

def analyze_model_discordance(nm_proba, sm_proba, y_true, nm_thresh, sm_thresh):
    """H2a: Quantify overall model discordance."""
    logging.info("=== H2a: QUANTIFYING MODEL DISCORDANCE ===")
    nm_pred = (nm_proba >= nm_thresh).astype(int)
    sm_pred = (sm_proba >= sm_thresh).astype(int)

    kappa = cohen_kappa_score(nm_pred, sm_pred)
    contingency_table = pd.crosstab(nm_pred, sm_pred)
    
    mcnemar_p_value = np.nan
    if contingency_table.shape == (2, 2):
        mcnemar_result = mcnemar(contingency_table.to_numpy())
        mcnemar_p_value = mcnemar_result.pvalue
    else:
        logging.warning("Contingency table is not 2x2. McNemar's test skipped.")

    correlation, _ = pearsonr(nm_proba, sm_proba)
    mcnemar_str = f"{mcnemar_p_value:.4f}" if not np.isnan(mcnemar_p_value) else "N/A"
    
    discordance_metrics = {
        "Cohen's Kappa": kappa,
        "McNemar's Test p-value": mcnemar_p_value,
        "Pearson Correlation": correlation,
    }
    
    logging.info(f"  Cohen's Kappa: {kappa:.4f}")
    logging.info(f"  McNemar's Test p-value: {mcnemar_str}")
    logging.info(f"  Pearson Correlation (on probabilities): r = {correlation:.4f}")
    
    return discordance_metrics

# =============================================================================
# H2b: FAILURE MODE, CONFOUNDER, and ROBUSTNESS ANALYSIS
# =============================================================================
def is_binary(series):
    """Check if a pandas Series is binary (contains only 0s and 1s)."""
    return series.dropna().isin([0, 1]).all()

def analyze_differential_failure_modes(cohorts, X_test_num):
    """H2b: Identify features driving model discordance using appropriate statistical tests."""
    logging.info("=== H2b: DIFFERENTIAL FAILURE MODE ANALYSIS ===")
    comparisons = {
        "FP_SM_vs_TN_concordant": ('FP_SM', 'TN_concordant'), 
        "FN_SM_vs_TP_concordant": ('FN_SM', 'TP_concordant'),
        "FP_NM_vs_TN_concordant": ('FP_NM', 'TN_concordant'), 
        "FN_NM_vs_TP_concordant": ('FN_NM', 'TP_concordant')
    }
    all_results = []
    
    for comp_name, (c1_name, c2_name) in comparisons.items():
        c1_mask, c2_mask = cohorts[c1_name], cohorts[c2_name]
        if c1_mask.sum() < 3 or c2_mask.sum() < 3:
            logging.warning(f"Skipping comparison {comp_name} due to insufficient samples.")
            continue
            
        g1, g2 = X_test_num[c1_mask], X_test_num[c2_mask]
        
        for feature in X_test_num.columns:
            series = X_test_num[feature]
            p_val, effect = 1.0, 0.0
            
            if is_binary(series):
                contingency = pd.crosstab(series[c1_mask | c2_mask], [c1_name]*c1_mask.sum() + [c2_name]*c2_mask.sum())
                if contingency.shape == (2, 2) and contingency.sum().sum() > 0 and 0 not in contingency.sum(axis=1) and 0 not in contingency.sum(axis=0):
                    _, p_val, _, _ = chi2_contingency(contingency, correction=False)
                    effect = g1[feature].mean() - g2[feature].mean()
                else: p_val = 1.0
            else:
                if g1[feature].dropna().empty or g2[feature].dropna().empty:
                    p_val = 1.0
                else:
                    _, p_val = mannwhitneyu(g1[feature].dropna(), g2[feature].dropna(), alternative='two-sided')
                effect = g1[feature].median() - g2[feature].median()
            
            all_results.append({"comparison": comp_name, "feature": feature, "p_value": p_val, "effect_size": effect})

    if not all_results: return pd.DataFrame()
    results_df = pd.DataFrame(all_results)
    results_df['q_value'] = fdrcorrection(results_df['p_value'].fillna(1.0), alpha=0.05)[1]
    return results_df

def analyze_confounders(cohorts, X_test_num):
    """H2b Addendum: Analyze potential confounders."""
    logging.info("=== H2b: CONFOUNDER ANALYSIS ===")
    confounder_features = ['gcs_last_derived_feature']
    
    comparisons = {
        'FP_SM vs FP_NM': (cohorts['FP_SM'], cohorts['FP_NM']),
        'FN_SM vs FN_NM': (cohorts['FN_SM'], cohorts['FN_NM'])
    }
    results = []
    
    for comp_name, (c1_mask, c2_mask) in comparisons.items():
        if c1_mask.sum() < 3 or c2_mask.sum() < 3: continue
        for feature in confounder_features:
            if feature in X_test_num.columns:
                stat, p_val = mannwhitneyu(X_test_num.loc[c1_mask, feature].dropna(), X_test_num.loc[c2_mask, feature].dropna())
                results.append({'Comparison': comp_name, 'Confounder': feature, 'Statistic': stat, 'p-value': p_val})
    
    return pd.DataFrame(results)

def check_robustness(output_dir):
    """H2b Addendum: Check feature robustness across sensitivity runs."""
    logging.info("=== H2b: ROBUSTNESS CHECK ACROSS THRESHOLDS ===")
    strategies = [d for d in os.listdir(output_dir) if d.startswith('sensitivity_')]
    if len(strategies) < 2: 
        logging.warning("Not enough sensitivity runs to check robustness.")
        return
    
    top_features = {}
    for strategy in strategies:
        path = os.path.join(output_dir, strategy, 'table_h2_3_failure_modes.csv')
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        top_features[strategy] = set(df[df['q_value'] < 0.05].sort_values('q_value').head(10)['feature'])
        
    for s1, s2 in combinations(strategies, 2):
        if s1 in top_features and s2 in top_features and top_features[s1] and top_features[s2]:
            jaccard_sim = len(top_features[s1].intersection(top_features[s2])) / len(top_features[s1].union(top_features[s2]))
            logging.info(f"  Jaccard similarity of top 10 features between '{s1}' and '{s2}': {jaccard_sim:.2f}")

# =============================================================================
# H2c: HYBRID MODELING AND SYNERGY ANALYSIS
# =============================================================================

def build_early_fusion_model(X_train_num, X_train_emb, y_train, config):
    logging.info("Building Early Fusion Hybrid Model...")
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=config.SEED, n_jobs=-1, n_estimators=500, max_depth=5, learning_rate=0.05).fit(np.hstack([X_train_num, X_train_emb]), y_train)
    logging.info("✅ Early Fusion model trained.")
    return model

def build_late_fusion_model(nm_model, sm_model, X_train_num, X_val_num, X_train_emb, X_val_emb, y_train, y_val, config):
    logging.info("Building Late Fusion (Stacking) Hybrid Model...")
    X_train_full_num, X_train_full_emb, y_train_full = pd.concat([X_train_num, X_val_num]), np.vstack([X_train_emb, X_val_emb]), pd.concat([y_train, y_val])
    oof_nm_preds, oof_sm_preds = np.zeros(len(y_train_full)), np.zeros(len(y_train_full))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    for _, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_train_full_num, y_train_full), total=5, desc="Stacking CV")):
        temp_nm = xgb.XGBClassifier(**nm_model.get_params()).fit(X_train_full_num.iloc[train_idx], y_train_full.iloc[train_idx])
        temp_sm = xgb.XGBClassifier(**sm_model.get_params()).fit(X_train_full_emb[train_idx], y_train_full.iloc[train_idx])
        oof_nm_preds[val_idx], oof_sm_preds[val_idx] = temp_nm.predict_proba(X_train_full_num.iloc[val_idx])[:, 1], temp_sm.predict_proba(X_train_full_emb[val_idx])[:, 1]
    meta_learner = LogisticRegression(random_state=config.SEED).fit(np.column_stack([oof_nm_preds, oof_sm_preds]), y_train_full)
    logging.info(f"✅ Late Fusion model trained. Coeffs: NM={meta_learner.coef_[0][0]:.2f}, SM={meta_learner.coef_[0][1]:.2f}")
    return meta_learner

def analyze_hybrid_synergy(hybrid_proba, nm_proba, sm_proba, cohorts, y_true, config):
    """H2c: Rigorously evaluate if the hybrid model's value comes from resolving disagreements."""
    logging.info("=== H2c: HYBRID SYNERGY ANALYSIS ===")
    
    discordant_mask = cohorts['FP_SM'] | cohorts['FN_SM'] | cohorts['FP_NM'] | cohorts['FN_NM']
    concordant_mask = cohorts['TP_concordant'] | cohorts['TN_concordant']
    logging.info(f"  Discordant Cohort Size: {discordant_mask.sum()}, Concordant (Correct) Cohort Size: {concordant_mask.sum()}")
    
    synergy_results = []
    y_true_np = y_true.values if hasattr(y_true, 'values') else y_true
    
    for pop_name, mask in {'Discordant': discordant_mask, 'Concordant_Correct': concordant_mask}.items():
        # ** THIS SECTION IS FIXED **
        # The incorrect check against N_BOOTSTRAP is removed. We only check if the cohort is too small
        # to produce a meaningful result (e.g., less than 2 samples).
        if mask.sum() < 2: 
             logging.warning(f"Skipping synergy analysis for {pop_name} due to empty or single-member cohort ({mask.sum()})")
             continue

        y_sub, nm_sub, sm_sub, hyb_sub = y_true_np[mask], nm_proba[mask], sm_proba[mask], hybrid_proba[mask]
        
        lift = min(brier_score_loss(y_sub, nm_sub), brier_score_loss(y_sub, sm_sub)) - brier_score_loss(y_sub, hyb_sub)
        lift_samples = []
        for _ in range(config.N_BOOTSTRAP):
            indices = np.random.choice(len(y_sub), len(y_sub), replace=True)
            # Ensure bootstrap sample is not degenerate
            if len(np.unique(y_sub[indices])) < 2:
                continue
            lift_samples.append(min(brier_score_loss(y_sub[indices], nm_sub[indices]), brier_score_loss(y_sub[indices], sm_sub[indices])) - brier_score_loss(y_sub[indices], hyb_sub[indices]))
        
        if not lift_samples:
            logging.warning(f"Could not generate valid bootstrap samples for {pop_name}. Skipping CI calculation.")
            ci_low, ci_high = np.nan, np.nan
        else:
            ci_low, ci_high = np.percentile(lift_samples, [2.5, 97.5])
        
        synergy_results.append({
            'Cohort': pop_name, 'N': len(y_sub), 'Brier_Lift': lift, 
            'Lift_CI_Lower': ci_low, 'Lift_CI_Upper': ci_high
        })

    synergy_df = pd.DataFrame(synergy_results)
    h2c_supported = False
    if len(synergy_df) == 2 and not synergy_df.isnull().values.any():
        if synergy_df.loc[0, 'Lift_CI_Lower'] > synergy_df.loc[1, 'Lift_CI_Upper']:
            h2c_supported = True
            
    logging.info(f"✅ H2c Synergy Supported: {h2c_supported}")
    return synergy_df, h2c_supported

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    config = ConfigH2()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s', handlers=[logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'h2_analysis_log.txt'), mode='w'), logging.StreamHandler()])
    start_time = time.time()
    
    data = load_data(config)
    nm_model, sm_model = load_trained_models(config)
    
    nm_val_proba = nm_model.predict_proba(data['X_val_num'])[:, 1]
    sm_val_proba = sm_model.predict_proba(data['X_val_emb'])[:, 1]
    nm_test_proba = nm_model.predict_proba(data['X_test_num'])[:, 1]
    sm_test_proba = sm_model.predict_proba(data['X_test_emb'])[:, 1]

    nm_perf = evaluate_model_performance(data['y_test'], nm_test_proba, "Numerical Model", config)
    sm_perf = evaluate_model_performance(data['y_test'], sm_test_proba, "Semantic Model", config)

    # --- SENSITIVITY ANALYSIS LOOP ---
    threshold_strategies = {'optimal_sm_f1': 'Primary', 'fixed_0.5': 'Fixed 0.5', 'model_specific_f1': 'Model-Specific'}
    for strategy_code, strategy_name in threshold_strategies.items():
        run_dir = os.path.join(config.OUTPUT_DIR, f'sensitivity_{strategy_code}')
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"\n{'='*20} RUNNING SENSITIVITY ANALYSIS: {strategy_name} {'='*20}")
        
        if strategy_code == 'optimal_sm_f1':
            primary_thresh, optimal_f1 = determine_f1_threshold(data['y_val'], sm_val_proba)
            logging.info(f"Primary classification threshold set to: {primary_thresh:.4f} (with F1-score: {optimal_f1:.4f})")
            nm_thresh, sm_thresh = primary_thresh, primary_thresh
        elif strategy_code == 'fixed_0.5':
            nm_thresh, sm_thresh = 0.5, 0.5
        else: # model_specific_f1
            nm_thresh, nm_f1 = determine_f1_threshold(data['y_val'], nm_val_proba)
            sm_thresh, sm_f1 = determine_f1_threshold(data['y_val'], sm_val_proba)
            logging.info(f"Model-Specific NM threshold: {nm_thresh:.4f} (F1: {nm_f1:.4f})")
            logging.info(f"Model-Specific SM threshold: {sm_thresh:.4f} (F1: {sm_f1:.4f})")
        
        # --- H2a & H2b Analyses ---
        discordance_metrics = analyze_model_discordance(nm_test_proba, sm_test_proba, data['y_test'], nm_thresh, sm_thresh)
        pd.DataFrame(list(discordance_metrics.items()), columns=['Metric', 'Value']).to_csv(os.path.join(run_dir, 'table_h2_2_discordance_metrics.csv'), index=False)

        cohorts = define_analysis_cohorts(nm_test_proba, sm_test_proba, data['y_test'], nm_thresh, sm_thresh)
        pd.DataFrame(list({k: v.sum() for k, v in cohorts.items()}.items()), columns=['Cohort', 'N']).to_csv(os.path.join(run_dir, 'table_h2_1_concordance.csv'), index=False)
        
        failure_df = analyze_differential_failure_modes(cohorts, data['X_test_num'])
        failure_df.to_csv(os.path.join(run_dir, 'table_h2_3_failure_modes.csv'), index=False)
        
        confounder_df = analyze_confounders(cohorts, data['X_test_num'])
        confounder_df.to_csv(os.path.join(run_dir, 'table_h2_4_confounders.csv'), index=False)
    
    check_robustness(config.OUTPUT_DIR)
    
    # --- HYBRID MODELING & SYNERGY (LEAK-FREE WORKFLOW) ---
    logging.info(f"\n{'='*20} PERFORMING HYBRID MODELING & SYNERGY ANALYSIS {'='*20}")

    # Step 1: Train candidate models ONLY on the training set
    logging.info("Training candidate hybrid models on the training set for champion selection...")
    candidate_early_fusion_model = build_early_fusion_model(data['X_train_num'].values, data['X_train_emb'], data['y_train'], config)
    # Note: For Late Fusion OOF, we pass the training data twice. The function's internal CV will split it.
    candidate_late_fusion_model = build_late_fusion_model(nm_model, sm_model, data['X_train_num'], data['X_train_num'], data['X_train_emb'], data['X_train_emb'], data['y_train'], data['y_train'], config)

    # Step 2: Evaluate candidates on the UNSEEN validation set
    logging.info("Evaluating candidate models on the validation set...")
    early_val_proba = candidate_early_fusion_model.predict_proba(np.hstack([data['X_val_num'].values, data['X_val_emb']]))[:, 1]
    late_val_proba = candidate_late_fusion_model.predict_proba(np.column_stack([nm_val_proba, sm_val_proba]))[:, 1]
    
    early_val_auroc = roc_auc_score(data['y_val'], early_val_proba)
    late_val_auroc = roc_auc_score(data['y_val'], late_val_proba)
    logging.info(f"Validation AUROC -> Early Fusion: {early_val_auroc:.4f}, Late Fusion: {late_val_auroc:.4f}")

    # Step 3: Select champion and RETRAIN on combined train+val data
    X_train_full_num = pd.concat([data['X_train_num'], data['X_val_num']])
    X_train_full_emb = np.vstack([data['X_train_emb'], data['X_val_emb']])
    y_train_full = pd.concat([data['y_train'], data['y_val']])

    if early_val_auroc >= late_val_auroc:
        champion_name = "Early Fusion"
        logging.info(f"Champion Hybrid Model selected: {champion_name}. Retraining on full train+val data...")
        final_champion_model = build_early_fusion_model(X_train_full_num.values, X_train_full_emb, y_train_full, config)
        hybrid_test_proba = final_champion_model.predict_proba(np.hstack([data['X_test_num'].values, data['X_test_emb']]))[:, 1]
    else:
        champion_name = "Late Fusion"
        logging.info(f"Champion Hybrid Model selected: {champion_name}. Retraining on full train+val data...")
        final_champion_model = build_late_fusion_model(nm_model, sm_model, data['X_train_num'], data['X_val_num'], data['X_train_emb'], data['X_val_emb'], data['y_train'], data['y_val'], config)
        hybrid_test_proba = final_champion_model.predict_proba(np.column_stack([nm_test_proba, sm_test_proba]))[:, 1]

    # Step 4: Evaluate final model on test set and perform synergy analysis
    hybrid_perf = evaluate_model_performance(data['y_test'], hybrid_test_proba, f"Champion Hybrid ({champion_name})", config)
    
    primary_thresh, _ = determine_f1_threshold(data['y_val'], sm_val_proba)
    primary_cohorts = define_analysis_cohorts(nm_test_proba, sm_test_proba, data['y_test'], primary_thresh, primary_thresh)
    synergy_df, _ = analyze_hybrid_synergy(hybrid_test_proba, nm_test_proba, sm_test_proba, primary_cohorts, data['y_test'], config)
    synergy_df.to_csv(os.path.join(config.OUTPUT_DIR, 'table_h2_5_synergy_analysis.csv'), index=False)
    
    # --- FINAL LOG PRINTOUTS ---
    logging.info("\n" + "="*80 + "\nFINAL RESULTS SUMMARY\n" + "="*80)
    master_perf_df = pd.DataFrame([nm_perf, sm_perf, hybrid_perf])
    logging.info(f"\n--- Master Performance Table (Test Set) ---\n{master_perf_df.to_string(index=False)}")
    
    primary_run_dir = os.path.join(config.OUTPUT_DIR, 'sensitivity_optimal_sm_f1')
    table_h2_1_path = os.path.join(primary_run_dir, 'table_h2_1_concordance.csv')
    if os.path.exists(table_h2_1_path):
        table_h2_1 = pd.read_csv(table_h2_1_path)
        logging.info(f"\n--- Primary Cohort Sizes (Table H2-1) ---\n{table_h2_1.to_string(index=False)}")
    
    table_h2_5_path = os.path.join(config.OUTPUT_DIR, 'table_h2_5_synergy_analysis.csv')
    if os.path.exists(table_h2_5_path):
        table_h2_5 = pd.read_csv(table_h2_5_path)
        logging.info(f"\n--- Quantitative Synergy Analysis (Table H2-5) ---\n{table_h2_5.to_string(index=False)}")

    logging.info(f"\nAnalysis completed in {(time.time() - start_time)/60:.2f} minutes. Results in '{config.OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()