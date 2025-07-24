# h2_analysis.py
"""
H2 Hypothesis Testing: Model Discordance, Failure Mode, and Synergy Analysis (Updated for Histogram-Based Plan)

This script provides a complete and definitive implementation of the updated H2 analysis plan. It includes:
- The new Histogram Drop-Off analysis as the primary method for cohort definition.
- The full sensitivity analysis loop for the two alternative thresholding schemes.
- The de-correlation step using Elastic Net to identify primary drivers (H2b).
- All other required analyses, tables, and figures.
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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy.stats import pearsonr, mannwhitneyu, chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import StandardScaler

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
# H2a: MODEL DISCORDANCE ANALYSIS (UPDATED METHODOLOGY)
# =============================================================================

def determine_histogram_threshold(y_true, y_probas, n_bins=100):
    """(Primary Method) Determine the Error Tolerance Threshold (T) via histogram drop-off analysis."""
    errors = np.abs(y_true.values - y_probas)
    counts, bin_edges = np.histogram(errors, bins=n_bins, range=(0,1))
    
    drop_off_scores = np.zeros(len(counts) - 1)
    for i in range(1, len(counts)):
        if counts[i-1] > 0 and counts[i] < counts[i-1]:
            score = (counts[i-1] - counts[i]) / counts[i-1]
            drop_off_scores[i-1] = score
            
    max_drop_off_index = np.argmax(drop_off_scores) + 1
    threshold_T = bin_edges[max_drop_off_index]
    
    return threshold_T

def determine_f1_threshold(y_true, y_probas):
    """(For Sensitivity Analysis) Determine threshold by maximizing F1-score."""
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [f1_score(y_true, y_probas >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]

def define_analysis_cohorts(nm_proba, sm_proba, y_true, threshold_T):
    """(Primary Method) Define the eight cohorts using the Error Tolerance Threshold T."""
    y_true_arr = y_true.values.astype(int)
    
    def classify(proba, y_true_arr):
        return np.where(y_true_arr == 0, 
                        np.where(proba < threshold_T, "TN", "FP"),
                        np.where(proba > (1 - threshold_T), "TP", "FN"))

    nm_class = classify(nm_proba, y_true_arr)
    sm_class = classify(sm_proba, y_true_arr)

    return {
        'TP_concordant': (nm_class == "TP") & (sm_class == "TP"),
        'TN_concordant': (nm_class == "TN") & (sm_class == "TN"),
        'FN_concordant': (nm_class == "FN") & (sm_class == "FN"),
        'FP_concordant': (nm_class == "FP") & (sm_class == "FP"),
        'FN_SM': (sm_class == "FN") & (nm_class == "TP"),
        'FP_SM': (sm_class == "FP") & (nm_class == "TN"),
        'FN_NM': (nm_class == "FN") & (sm_class == "TP"),
        'FP_NM': (nm_class == "FP") & (sm_class == "TN"),
    }

def define_analysis_cohorts_by_prob(nm_proba, sm_proba, y_true, prob_thresh):
    """(For Sensitivity Analysis) Define cohorts using a simple probability threshold."""
    nm_pred, sm_pred, y_true_arr = (nm_proba >= prob_thresh).astype(int), (sm_proba >= prob_thresh).astype(int), y_true.astype(int)
    return {
        'TP_concordant': (nm_pred == 1) & (sm_pred == 1) & (y_true_arr == 1),
        'TN_concordant': (nm_pred == 0) & (sm_pred == 0) & (y_true_arr == 0),
        'FN_concordant': (nm_pred == 0) & (sm_pred == 0) & (y_true_arr == 1),
        'FP_concordant': (nm_pred == 1) & (sm_pred == 1) & (y_true_arr == 0),
        'FN_SM': (sm_pred == 0) & (nm_pred == 1) & (y_true_arr == 1),
        'FP_SM': (sm_pred == 1) & (nm_pred == 0) & (y_true_arr == 0),
        'FN_NM': (nm_pred == 0) & (sm_pred == 1) & (y_true_arr == 1),
        'FP_NM': (nm_pred == 1) & (sm_pred == 0) & (y_true_arr == 0),
    }

def analyze_model_discordance(nm_proba, sm_proba, cohorts):
    """H2a: Quantify overall model discordance based on cohort definitions."""
    logging.info("=== H2a: QUANTIFYING MODEL DISCORDANCE ===")
    nm_pred = (cohorts['TP_concordant'] | cohorts['FP_concordant'] | cohorts['FN_SM'] | cohorts['FP_NM']).astype(int)
    sm_pred = (cohorts['TP_concordant'] | cohorts['FP_concordant'] | cohorts['FN_NM'] | cohorts['FP_SM']).astype(int)
    
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
# H2b: FAILURE MODE, CONFOUNDER, AND ROBUSTNESS ANALYSIS
# =============================================================================
def is_binary(series):
    """Check if a pandas Series is binary (contains only 0s and 1s)."""
    return series.dropna().isin([0, 1]).all()

def analyze_differential_failure_modes(cohorts, X_test_num, config):
    """H2b: Identify features driving model discordance and de-correlate them."""
    logging.info("=== H2b: DIFFERENTIAL FAILURE MODE ANALYSIS (Steps 1-3) ===")
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
            continue
        g1, g2 = X_test_num[c1_mask], X_test_num[c2_mask]
        for feature in X_test_num.columns:
            p_val, effect = 1.0, 0.0
            if is_binary(X_test_num[feature]):
                contingency = pd.crosstab(X_test_num[feature][c1_mask | c2_mask], [c1_name]*c1_mask.sum() + [c2_name]*c2_mask.sum())
                if contingency.shape == (2, 2) and 0 not in contingency.sum(axis=1) and 0 not in contingency.sum(axis=0):
                    _, p_val, _, _ = chi2_contingency(contingency)
                    effect = g1[feature].mean() - g2[feature].mean()
            else:
                if not g1[feature].dropna().empty and not g2[feature].dropna().empty:
                    _, p_val = mannwhitneyu(g1[feature].dropna(), g2[feature].dropna(), alternative='two-sided')
                effect = g1[feature].median() - g2[feature].median()
            all_results.append({"comparison": comp_name, "feature": feature, "p_value": p_val, "effect_size": effect})

    if not all_results: return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    results_df['q_value'] = fdrcorrection(results_df['p_value'].fillna(1.0), alpha=0.05)[1]
    
    logging.info("--- H2b Step 4: Interpretation & De-correlation using Elastic Net ---")
    significant_features_df = results_df[results_df['q_value'] < 0.05]
    primary_drivers = []
    
    for comp_name, (c1_name, c2_name) in comparisons.items():
        comp_features_df = significant_features_df[significant_features_df['comparison'] == comp_name]
        if comp_features_df.empty:
            continue
        
        c1_mask, c2_mask = cohorts[c1_name], cohorts[c2_name]
        if c1_mask.sum() < 10 or c2_mask.sum() < 10:
             logging.warning(f"Skipping Elastic Net for {comp_name} due to small cohort size.")
             continue
        
        feature_subset = comp_features_df['feature'].unique()
        X_comp = X_test_num.loc[c1_mask | c2_mask, feature_subset].fillna(0)
        y_comp = np.array([1]*c1_mask.sum() + [0]*c2_mask.sum())
        
        scaler = StandardScaler().fit(X_comp)
        X_comp_scaled = scaler.transform(X_comp)

        l1_ratios = [0.1, 0.5, 0.9, 1.0]
        model = LogisticRegressionCV(
            Cs=10, penalty='elasticnet', l1_ratios=l1_ratios, solver='saga', 
            max_iter=100000, random_state=config.SEED, cv=3, n_jobs=-1
        ).fit(X_comp_scaled, y_comp)
        
        non_zero_coeffs = feature_subset[model.coef_[0] != 0]
        for feature in non_zero_coeffs:
            primary_drivers.append((comp_name, feature))
        logging.info(f"  Found {len(non_zero_coeffs)} primary drivers for {comp_name} from {len(feature_subset)} candidates.")

    results_df['is_primary_driver'] = results_df.apply(lambda row: (row['comparison'], row['feature']) in primary_drivers, axis=1)
    return results_df

def analyze_confounders(cohorts, X_test_num):
    """H2b Addendum: Analyze potential confounders."""
    logging.info("=== H2b: CONFOUNDER ANALYSIS ===")
    confounder_features = ['gcs_last_derived_feature']
    comparisons = {'FP_SM vs FP_NM': (cohorts['FP_SM'], cohorts['FP_NM']), 'FN_SM vs FN_NM': (cohorts['FN_SM'], cohorts['FN_NM'])}
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
    if len(strategies) < 2: return
    top_features = {}
    for strategy in strategies:
        path = os.path.join(output_dir, strategy, 'table_h2_3_failure_modes.csv')
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        top_features[strategy] = set(df[df['is_primary_driver']].sort_values('q_value').head(10)['feature'])
    for s1, s2 in combinations(strategies, 2):
        if s1 in top_features and s2 in top_features and top_features[s1] and top_features[s2]:
            jaccard_sim = len(top_features[s1].intersection(top_features[s2])) / len(top_features[s1].union(top_features[s2]))
            logging.info(f"  Jaccard similarity of top 10 primary drivers between '{s1}' and '{s2}': {jaccard_sim:.2f}")

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
    all_lift_samples = {}
    
    for pop_name, mask in {'Discordant': discordant_mask, 'Concordant_Correct': concordant_mask}.items():
        if mask.sum() < 2: 
             logging.warning(f"Skipping synergy analysis for {pop_name} due to empty or single-member cohort ({mask.sum()})")
             continue
        y_sub, nm_sub, sm_sub, hyb_sub = y_true_np[mask], nm_proba[mask], sm_proba[mask], hybrid_proba[mask]
        lift = min(brier_score_loss(y_sub, nm_sub), brier_score_loss(y_sub, sm_sub)) - brier_score_loss(y_sub, hyb_sub)
        lift_samples = []
        for _ in range(config.N_BOOTSTRAP):
            indices = np.random.choice(len(y_sub), len(y_sub), replace=True)
            if len(np.unique(y_sub[indices])) < 2: continue
            lift_samples.append(min(brier_score_loss(y_sub[indices], nm_sub[indices]), brier_score_loss(y_sub[indices], sm_sub[indices])) - brier_score_loss(y_sub[indices], hyb_sub[indices]))
        
        all_lift_samples[pop_name] = lift_samples
        if not lift_samples: ci_low, ci_high = np.nan, np.nan
        else: ci_low, ci_high = np.percentile(lift_samples, [2.5, 97.5])
        synergy_results.append({'Cohort': pop_name, 'N': len(y_sub), 'Brier_Lift': lift, 'Lift_CI_Lower': ci_low, 'Lift_CI_Upper': ci_high})

    synergy_df = pd.DataFrame(synergy_results)
    
    h2c_supported = False
    if 'Discordant' in all_lift_samples and 'Concordant_Correct' in all_lift_samples:
        diff_samples = np.array(all_lift_samples['Discordant']) - np.array(all_lift_samples['Concordant_Correct'])
        diff_pe = np.median(diff_samples)
        diff_ci_low, diff_ci_high = np.percentile(diff_samples, [2.5, 97.5])
        diff_row = pd.DataFrame([{'Cohort': 'Difference in Lifts', 'N': np.nan, 'Brier_Lift': diff_pe, 'Lift_CI_Lower': diff_ci_low, 'Lift_CI_Upper': diff_ci_high}])
        synergy_df = pd.concat([synergy_df, diff_row], ignore_index=True)
        if diff_ci_low > 0: h2c_supported = True
            
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
    
    nm_val_proba, sm_val_proba = nm_model.predict_proba(data['X_val_num'])[:, 1], sm_model.predict_proba(data['X_val_emb'])[:, 1]
    nm_test_proba, sm_test_proba = nm_model.predict_proba(data['X_test_num'])[:, 1], sm_model.predict_proba(data['X_test_emb'])[:, 1]

    nm_perf = evaluate_model_performance(data['y_test'], nm_test_proba, "Numerical Model", config)
    sm_perf = evaluate_model_performance(data['y_test'], sm_test_proba, "Semantic Model", config)

    # --- SENSITIVITY ANALYSIS LOOP (UPDATED STRUCTURE) ---
    threshold_strategies = {
        'histogram_dropoff': 'Primary', 
        'optimal_f1': 'Scheme A (F1-Optimized)', 
        'fixed_0.5': 'Scheme B (Fixed 0.5)'
    }
    
    for strategy_code, strategy_name in threshold_strategies.items():
        run_dir = os.path.join(config.OUTPUT_DIR, f'sensitivity_{strategy_code}')
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"\n{'='*20} RUNNING ANALYSIS: {strategy_name} {'='*20}")
        
        if strategy_code == 'histogram_dropoff':
            threshold_T = determine_histogram_threshold(data['y_val'], nm_val_proba)
            logging.info(f"Primary Error Tolerance Threshold (T) set to: {threshold_T:.4f}")
            cohorts = define_analysis_cohorts(nm_test_proba, sm_test_proba, data['y_test'], threshold_T)
        
        else: # Sensitivity runs using probability thresholds
            if strategy_code == 'optimal_f1':
                prob_thresh, f1 = determine_f1_threshold(data['y_val'], sm_val_proba)
                logging.info(f"Scheme A: F1-optimized probability threshold: {prob_thresh:.4f} (F1: {f1:.4f})")
            else: # fixed_0.5
                prob_thresh = 0.5
                logging.info(f"Scheme B: Fixed probability threshold: {prob_thresh}")
            cohorts = define_analysis_cohorts_by_prob(nm_test_proba, sm_test_proba, data['y_test'], prob_thresh)

        discordance_metrics = analyze_model_discordance(nm_test_proba, sm_test_proba, cohorts)
        pd.DataFrame(list(discordance_metrics.items()), columns=['Metric', 'Value']).to_csv(os.path.join(run_dir, 'table_h2_2_discordance_metrics.csv'), index=False)
        pd.DataFrame(list({k: v.sum() for k, v in cohorts.items()}.items()), columns=['Cohort', 'N']).to_csv(os.path.join(run_dir, 'table_h2_1_concordance.csv'), index=False)
        failure_df = analyze_differential_failure_modes(cohorts, data['X_test_num'], config)
        failure_df.to_csv(os.path.join(run_dir, 'table_h2_3_failure_modes.csv'), index=False)
        confounder_df = analyze_confounders(cohorts, data['X_test_num'])
        confounder_df.to_csv(os.path.join(run_dir, 'table_h2_4_confounders.csv'), index=False)
    
    check_robustness(config.OUTPUT_DIR)
    
    # --- HYBRID MODELING & SYNERGY (LEAK-FREE WORKFLOW) ---
    logging.info(f"\n{'='*20} PERFORMING HYBRID MODELING & SYNERGY ANALYSIS {'='*20}")
    
    logging.info("Training candidate hybrid models on the training set for champion selection...")
    candidate_early_fusion_model = build_early_fusion_model(data['X_train_num'].values, data['X_train_emb'], data['y_train'], config)
    candidate_late_fusion_model = build_late_fusion_model(nm_model, sm_model, data['X_train_num'], data['X_train_num'], data['X_train_emb'], data['X_train_emb'], data['y_train'], data['y_train'], config)

    logging.info("Evaluating candidate models on the validation set...")
    early_val_proba = candidate_early_fusion_model.predict_proba(np.hstack([data['X_val_num'].values, data['X_val_emb']]))[:, 1]
    late_val_proba = candidate_late_fusion_model.predict_proba(np.column_stack([nm_val_proba, sm_val_proba]))[:, 1]
    early_val_auroc, late_val_auroc = roc_auc_score(data['y_val'], early_val_proba), roc_auc_score(data['y_val'], late_val_proba)
    logging.info(f"Validation AUROC -> Early Fusion: {early_val_auroc:.4f}, Late Fusion: {late_val_auroc:.4f}")

    if early_val_auroc >= late_val_auroc:
        champion_name = "Early Fusion"
        logging.info(f"Champion Hybrid Model selected: {champion_name}. Retraining on full train+val data...")
        final_champion_model = build_early_fusion_model(pd.concat([data['X_train_num'], data['X_val_num']]).values, np.vstack([data['X_train_emb'], data['X_val_emb']]), pd.concat([data['y_train'], data['y_val']]), config)
        hybrid_test_proba = final_champion_model.predict_proba(np.hstack([data['X_test_num'].values, data['X_test_emb']]))[:, 1]
    else:
        champion_name = "Late Fusion"
        logging.info(f"Champion Hybrid Model selected: {champion_name}. Retraining on full train+val data...")
        final_champion_model = build_late_fusion_model(nm_model, sm_model, data['X_train_num'], data['X_val_num'], data['X_train_emb'], data['X_val_emb'], data['y_train'], data['y_val'], config)
        hybrid_test_proba = final_champion_model.predict_proba(np.column_stack([nm_test_proba, sm_test_proba]))[:, 1]

    hybrid_perf = evaluate_model_performance(data['y_test'], hybrid_test_proba, f"Champion Hybrid ({champion_name})", config)
    
    primary_T = determine_histogram_threshold(data['y_val'], nm_val_proba)
    primary_cohorts = define_analysis_cohorts(nm_test_proba, sm_test_proba, data['y_test'], primary_T)
    synergy_df, _ = analyze_hybrid_synergy(hybrid_test_proba, nm_test_proba, sm_test_proba, primary_cohorts, data['y_test'], config)
    synergy_df.to_csv(os.path.join(config.OUTPUT_DIR, 'table_h2_5_synergy_analysis.csv'), index=False)
    
    # --- FINAL LOG PRINTOUTS ---
    logging.info("\n" + "="*80 + "\nFINAL RESULTS SUMMARY\n" + "="*80)
    master_perf_df = pd.DataFrame([nm_perf, sm_perf, hybrid_perf])
    logging.info(f"\n--- Master Performance Table (Test Set) ---\n{master_perf_df.to_string(index=False)}")
    primary_run_dir = os.path.join(config.OUTPUT_DIR, 'sensitivity_histogram_dropoff')
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