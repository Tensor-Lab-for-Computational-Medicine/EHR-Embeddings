"""
H2 Analysis: Model Discordance, Subgroup Discovery, and Meta-Analysis
Streamlined version focusing on core hypothesis testing
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.metrics import cohen_kappa_score, brier_score_loss
from scipy.stats import pearsonr, mannwhitneyu
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config_h2 import ConfigH2

# Try to import pysubgroup for H2b analysis
try:
    import pysubgroup as ps
    SUBGROUP_DISCOVERY_AVAILABLE = True
except ImportError:
    SUBGROUP_DISCOVERY_AVAILABLE = False
    print("⚠️ pysubgroup not available - H2b analysis will be limited")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config):
    """Load preprocessed numerical features, embeddings, and labels."""
    logging.info("Loading data...")
    data = {}
    
    # Load numerical data and labels
    with open(config.X_TEST_NUM_PATH, 'rb') as f:
        data['X_test_num'] = pickle.load(f)
    with open(config.Y_TEST_PATH, 'rb') as f:
        data['y_test'] = pickle.load(f)
    
    # Load test embeddings
    labels_df = pd.read_csv(
        os.path.join(config.LABEL_DIR, 'test_labels.csv'),
        header=None, 
        names=['icustay_id', config.TARGET_VARIABLE]
    )
    
    embedding_dir = os.path.join(config.EMBEDDING_DATA_DIR, 'test')
    embeddings = []
    
    for icustay_id in labels_df['icustay_id'].values:
        emb_path = os.path.join(embedding_dir, f"{icustay_id}.npy")
        if os.path.exists(emb_path):
            embeddings.append(np.load(emb_path))
        else:
            embeddings.append(np.zeros(768))
    
    data['X_test_emb'] = np.vstack(embeddings)
    
    logging.info(f"Loaded test data: {len(data['y_test'])} samples")
    return data

def load_models(config):
    """Load pre-trained NM and SM models."""
    logging.info("Loading models...")
    
    with open(config.BASELINE_MODEL_PATH, 'rb') as f:
        nm_model = pickle.load(f)
    
    with open(config.CHAMPION_MODEL_PATH, 'rb') as f:
        sm_model = pickle.load(f)
    
    return nm_model, sm_model

# =============================================================================
# H2a: DISCORDANCE QUANTIFICATION
# =============================================================================

def define_cohorts(nm_pred, sm_pred, y_true):
    """Define 8 cohorts based on model agreement patterns using 0.5 threshold."""
    y_true = y_true.values.astype(int) if hasattr(y_true, 'values') else y_true.astype(int)
    
    return {
        'TP_concordant': (nm_pred == 1) & (sm_pred == 1) & (y_true == 1),
        'TN_concordant': (nm_pred == 0) & (sm_pred == 0) & (y_true == 0),
        'FN_concordant': (nm_pred == 0) & (sm_pred == 0) & (y_true == 1),
        'FP_concordant': (nm_pred == 1) & (sm_pred == 1) & (y_true == 0),
        'FN_SM': (sm_pred == 0) & (nm_pred == 1) & (y_true == 1),
        'FP_SM': (sm_pred == 1) & (nm_pred == 0) & (y_true == 0),
        'FN_NM': (nm_pred == 0) & (sm_pred == 1) & (y_true == 1),
        'FP_NM': (nm_pred == 1) & (sm_pred == 0) & (y_true == 0),
    }

def quantify_discordance(nm_proba, sm_proba, nm_pred, sm_pred, y_true):
    """H2a: Calculate discordance metrics including Brier scores."""
    kappa = cohen_kappa_score(nm_pred, sm_pred)
    
    # McNemar's test
    contingency = pd.crosstab(nm_pred, sm_pred)
    mcnemar_result = mcnemar(contingency.to_numpy())
    
    # Pearson correlation on probabilities
    correlation, _ = pearsonr(nm_proba, sm_proba)
    
    # Brier scores
    brier_nm = brier_score_loss(y_true, nm_proba)
    brier_sm = brier_score_loss(y_true, sm_proba)
    
    return {
        'cohens_kappa': kappa,
        'mcnemar_pvalue': mcnemar_result.pvalue,
        'pearson_correlation': correlation,
        'brier_nm': brier_nm,
        'brier_sm': brier_sm,
        'brier_difference': brier_nm - brier_sm
    }

# =============================================================================
# H2b: SUBGROUP DISCOVERY
# =============================================================================

def prepare_features_for_subgroup_discovery(X_data):
    """Prepare features for subgroup discovery by handling missing values."""
    X_clean = X_data.copy()
    
    # Handle missing values with median for numerical features
    for col in X_clean.columns:
        if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            X_clean[col] = X_clean[col].fillna(X_clean[col].median())
    
    return X_clean

def run_subgroup_discovery_analysis(X_features, target, analysis_name, config):
    """Run subgroup discovery using pysubgroup with comprehensive result parsing."""
    if not SUBGROUP_DISCOVERY_AVAILABLE:
        return pd.DataFrame()
    
    # Ensure features and target have same index
    common_index = X_features.index.intersection(target.index)
    X_analysis = X_features.loc[common_index].copy()
    y_analysis = target.loc[common_index].copy()
    
    # Check for sufficient samples
    n_positive = y_analysis.sum()
    n_negative = len(y_analysis) - n_positive
    
    MIN_SAMPLES_PER_CLASS = 30
    if n_positive < MIN_SAMPLES_PER_CLASS or n_negative < MIN_SAMPLES_PER_CLASS:
        return pd.DataFrame()
    
    # Create dataset for pysubgroup
    data = X_analysis.copy()
    data['target'] = y_analysis.values
    
    try:
        # Create binary target
        target_column = ps.BinaryTarget('target', 1)
        
        # Create search space from features
        searchspace = ps.create_selectors(data, ignore=['target'])
        
        # Create the task
        task = ps.SubgroupDiscoveryTask(
            data,
            target_column,
            searchspace,
            result_set_size=config.SUBGROUP_TOP_K,
            depth=config.SUBGROUP_MAX_DEPTH,
            qf=ps.WRAccQF(),
            min_quality=0.01
        )
        
        # Use BeamSearch algorithm
        result = ps.BeamSearch(beam_width=20).execute(task)
        
        results_data = []
        
        # Try to_dataframe method first
        if hasattr(result, 'to_dataframe'):
            result_df = result.to_dataframe()
            
            for idx in range(min(len(result_df), config.SUBGROUP_TOP_K)):
                row = result_df.iloc[idx]
                
                results_data.append({
                    'rank': idx + 1,
                    'description': str(row['subgroup']),
                    'quality_WRAcc': row['quality'],
                    'coverage': int(row['size_sg']),
                    'coverage_pct': round(row['relative_size_sg'] * 100, 1),
                    'n_positives': int(row['positives_sg']),
                    'target_share': round(row['target_share_sg'] * 100, 1),
                    'baseline_rate': round(row['target_share_dataset'] * 100, 1),
                    'lift': round(row['lift'], 2)
                })
        
        # Fallback: access through results list
        elif hasattr(result, 'results'):
            for idx in range(min(len(result.results), config.SUBGROUP_TOP_K)):
                item = result.results[idx]
                
                # Parse (quality, subgroup, stats) tuple
                if isinstance(item, tuple) and len(item) >= 2:
                    q = item[0]
                    sg = item[1]
                    
                    if hasattr(sg, 'covers'):
                        covered = sg.covers(data)
                        coverage = np.sum(covered)
                        
                        if coverage > 0:
                            n_positives_in_sg = np.sum(data.loc[covered, 'target'].values)
                            coverage_pct = (coverage / len(data)) * 100
                            target_share = (n_positives_in_sg / coverage * 100)
                            baseline_rate = (n_positive / len(y_analysis)) * 100
                            lift = target_share / baseline_rate if baseline_rate > 0 else 0
                            
                            results_data.append({
                                'rank': idx + 1,
                                'description': str(sg),
                                'quality_WRAcc': q,
                                'coverage': int(coverage),
                                'coverage_pct': round(coverage_pct, 1),
                                'n_positives': int(n_positives_in_sg),
                                'target_share': round(target_share, 1),
                                'baseline_rate': round(baseline_rate, 1),
                                'lift': round(lift, 2)
                            })
        
        return pd.DataFrame(results_data)
        
    except Exception as e:
        logging.error(f"Error in subgroup discovery: {e}")
        return pd.DataFrame()


def create_subgroup_targets(cohorts, data_index):
    """Create binary target variables for subgroup discovery."""
    targets = {}
    
    # Analysis 1: SM False Negatives
    population_1_mask = cohorts['FN_SM'] | cohorts['TP_concordant']
    target_1 = pd.Series(0, index=data_index)
    target_1[cohorts['FN_SM']] = 1
    targets['SM_miss'] = (target_1[population_1_mask], population_1_mask)
    
    # Analysis 2: SM False Positives  
    population_2_mask = cohorts['FP_SM'] | cohorts['TN_concordant']
    target_2 = pd.Series(0, index=data_index)
    target_2[cohorts['FP_SM']] = 1
    targets['SM_false_alarm'] = (target_2[population_2_mask], population_2_mask)
    
    # Analysis 3: NM False Negatives
    population_3_mask = cohorts['FN_NM'] | cohorts['TP_concordant']
    target_3 = pd.Series(0, index=data_index)
    target_3[cohorts['FN_NM']] = 1
    targets['NM_miss'] = (target_3[population_3_mask], population_3_mask)
    
    # Analysis 4: NM False Positives
    population_4_mask = cohorts['FP_NM'] | cohorts['TN_concordant']
    target_4 = pd.Series(0, index=data_index)
    target_4[cohorts['FP_NM']] = 1
    targets['NM_false_alarm'] = (target_4[population_4_mask], population_4_mask)
    
    return targets


def analyze_differential_failures(cohorts, X_test_num, config):
    """H2b: Identify multi-feature patterns driving model discordance."""
    # Prepare features
    X_features = prepare_features_for_subgroup_discovery(X_test_num)
    
    # Create target variables for each analysis
    targets = create_subgroup_targets(cohorts, X_test_num.index)
    
    # Run subgroup discovery for each analysis
    all_results = {}
    
    analyses = [
        ('SM_miss', 'SM False Negatives vs TP Concordant'),
        ('SM_false_alarm', 'SM False Positives vs TN Concordant'),
        ('NM_miss', 'NM False Negatives vs TP Concordant'),
        ('NM_false_alarm', 'NM False Positives vs TN Concordant')
    ]
    
    for analysis_key, analysis_title in analyses:
        if analysis_key not in targets:
            continue
        
        target_series, population_mask = targets[analysis_key]
        
        # Get the relevant features for this population
        X_population = X_features[population_mask]
        
        # Run subgroup discovery
        results_df = run_subgroup_discovery_analysis(
            X_features=X_population,
            target=target_series,
            analysis_name=analysis_title,
            config=config
        )
        
        all_results[analysis_key] = results_df
    
    return all_results

# =============================================================================
# PHASE V: META-ANALYSIS
# =============================================================================

def calculate_meta_features(X_data):
    """Step 1: Calculate meta-features for each patient."""
    meta_features = pd.DataFrame(index=X_data.index)
    
    # Density metrics
    meta_features['total_measurement_count'] = (~X_data.isna()).sum(axis=1)
    meta_features['unique_feature_count'] = (~X_data.isna()).astype(int).sum(axis=1)
    meta_features['input_token_count'] = meta_features['total_measurement_count'] * 3
    
    # Volatility metrics
    numeric_cols = X_data.select_dtypes(include=[np.number]).columns
    stddev_cols = [col for col in numeric_cols if 'stddev' in col.lower()]
    slope_cols = [col for col in numeric_cols if 'slope' in col.lower()]
    
    if stddev_cols:
        meta_features['aggregate_stddev'] = X_data[stddev_cols].mean(axis=1)
    else:
        meta_features['aggregate_stddev'] = 0
    
    if slope_cols:
        meta_features['aggregate_slope'] = X_data[slope_cols].abs().mean(axis=1)
    else:
        meta_features['aggregate_slope'] = 0
    
    # Imputation metrics
    meta_features['total_imputation_count'] = X_data.isna().sum(axis=1)
    meta_features['imputation_proportion'] = (
        meta_features['total_imputation_count'] / len(X_data.columns)
    )
    
    return meta_features.fillna(0)

def analyze_meta_features(subgroup_results, cohorts, meta_features):
    """Step 2: Link phenotypes to data characteristics."""
    meta_analysis_results = []
    
    analysis_mappings = {
        'SM_miss': ('FN_SM', 'TP_concordant'),
        'SM_false_alarm': ('FP_SM', 'TN_concordant'),
        'NM_miss': ('FN_NM', 'TP_concordant'),
        'NM_false_alarm': ('FP_NM', 'TN_concordant')
    }
    
    for analysis_type, (failure_cohort, success_cohort) in analysis_mappings.items():
        if analysis_type not in subgroup_results or subgroup_results[analysis_type].empty:
            continue
        
        failure_mask = cohorts[failure_cohort]
        success_mask = cohorts[success_cohort]
        
        # Analyze meta-features for this comparison
        for meta_feature in meta_features.columns:
            failure_values = meta_features.loc[failure_mask, meta_feature].dropna()
            success_values = meta_features.loc[success_mask, meta_feature].dropna()
            
            if len(failure_values) > 0 and len(success_values) > 0:
                statistic, p_value = mannwhitneyu(
                    failure_values, success_values, alternative='two-sided'
                )
                
                meta_analysis_results.append({
                    'analysis_type': analysis_type,
                    'meta_feature': meta_feature,
                    'failure_median': failure_values.median(),
                    'success_median': success_values.median(),
                    'effect_size': failure_values.median() - success_values.median(),
                    'p_value': p_value
                })
    
    results_df = pd.DataFrame(meta_analysis_results)
    
    # Apply FDR correction
    if not results_df.empty:
        results_df['q_value'] = fdrcorrection(results_df['p_value'], alpha=0.05)[1]
        results_df['significant'] = results_df['q_value'] < 0.05
    
    return results_df

def test_meta_hypotheses(meta_results):
    """Step 3: Test meta-hypotheses."""
    if meta_results.empty:
        return {}
    
    significant = meta_results[meta_results['significant'] == True]
    
    hypotheses = {
        'H_meta_1_density': significant[
            significant['meta_feature'].isin(['total_measurement_count', 'input_token_count'])
        ].shape[0] > 0,
        'H_meta_2_volatility': significant[
            significant['meta_feature'].isin(['aggregate_stddev', 'aggregate_slope'])
        ].shape[0] > 0,
        'H_meta_3_imputation': significant[
            significant['meta_feature'].isin(['imputation_proportion', 'total_imputation_count'])
        ].shape[0] > 0
    }
    
    return hypotheses

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Execute streamlined H2 analysis."""
    config = ConfigH2()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'h2_analysis.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("H2 ANALYSIS: MODEL DISCORDANCE AND META-ANALYSIS")
    
    # Load data and models
    data = load_data(config)
    nm_model, sm_model = load_models(config)
    
    # Generate predictions
    nm_proba = nm_model.predict_proba(data['X_test_num'])[:, 1]
    sm_proba = sm_model.predict_proba(data['X_test_emb'])[:, 1]
    
    # Binary predictions using 0.5 threshold
    nm_pred = (nm_proba >= 0.5).astype(int)
    sm_pred = (sm_proba >= 0.5).astype(int)
    
    # H2a: Quantify discordance
    logging.info("\n=== H2a: DISCORDANCE QUANTIFICATION ===")
    y_test = data['y_test']
    discordance_metrics = quantify_discordance(nm_proba, sm_proba, nm_pred, sm_pred, y_test)    
    logging.info(f"Cohen's Kappa: {discordance_metrics['cohens_kappa']:.4f}")
    logging.info(f"McNemar p-value: {discordance_metrics['mcnemar_pvalue']:.4f}")
    logging.info(f"Pearson correlation: {discordance_metrics['pearson_correlation']:.4f}")
    logging.info(f"Brier Score NM: {discordance_metrics['brier_nm']:.4f}")
    logging.info(f"Brier Score SM: {discordance_metrics['brier_sm']:.4f}")
    logging.info(f"Brier Difference (NM - SM): {discordance_metrics['brier_difference']:.4f}")
    
    # Define cohorts
    cohorts = define_cohorts(nm_pred, sm_pred, data['y_test'])
    cohort_sizes = pd.DataFrame([
        {'Cohort': name, 'N': mask.sum()} for name, mask in cohorts.items()
    ])
    
    logging.info("\nCohort Sizes:")
    logging.info(cohort_sizes.to_string(index=False))
    
    # H2b: Subgroup discovery
    logging.info("\n=== H2b: DIFFERENTIAL FAILURE ANALYSIS ===")
    subgroup_results = analyze_differential_failures(cohorts, data['X_test_num'], config)
    
    # Log top patterns for each analysis
    for analysis_type, results_df in subgroup_results.items():
        if not results_df.empty:
            logging.info(f"\n{analysis_type} - Top patterns:")
            for idx in range(min(3, len(results_df))):
                row = results_df.iloc[idx]
                # Use 'description' column instead of 'rule'
                logging.info(f"  Rank {row['rank']}: {row['description'][:80]}...")
                logging.info(f"    WRAcc={row['quality_WRAcc']:.3f}, Lift={row['lift']}x")
    
    # Phase V: Meta-analysis
    logging.info("\n=== PHASE V: META-ANALYSIS ===")
    
    # Calculate meta-features
    meta_features = calculate_meta_features(data['X_test_num'])
    
    # Analyze meta-features
    meta_results = analyze_meta_features(subgroup_results, cohorts, meta_features)
    
    # Test meta-hypotheses
    hypotheses = test_meta_hypotheses(meta_results)
    
    logging.info("\nMeta-Hypothesis Results:")
    logging.info(f"H_meta_1 (Data Density): {'SUPPORTED' if hypotheses.get('H_meta_1_density') else 'NOT SUPPORTED'}")
    logging.info(f"H_meta_2 (Volatility): {'SUPPORTED' if hypotheses.get('H_meta_2_volatility') else 'NOT SUPPORTED'}")
    logging.info(f"H_meta_3 (Imputation): {'SUPPORTED' if hypotheses.get('H_meta_3_imputation') else 'NOT SUPPORTED'}")
    
    # Save all results
    cohort_sizes.to_csv(os.path.join(config.OUTPUT_DIR, 'cohort_sizes.csv'), index=False)
    pd.DataFrame([discordance_metrics]).to_csv(
        os.path.join(config.OUTPUT_DIR, 'discordance_metrics.csv'), index=False
    )
    
    for analysis_type, df in subgroup_results.items():
        if not df.empty:
            df.to_csv(
                os.path.join(config.OUTPUT_DIR, f'subgroups_{analysis_type}.csv'), 
                index=False
            )
    
    if not meta_results.empty:
        meta_results.to_csv(
            os.path.join(config.OUTPUT_DIR, 'meta_analysis_results.csv'), 
            index=False
        )
    
    logging.info(f"\nAnalysis complete. Results saved to: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()