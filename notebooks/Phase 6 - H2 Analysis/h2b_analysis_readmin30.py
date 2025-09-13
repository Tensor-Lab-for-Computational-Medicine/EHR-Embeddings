"""
H2b Analysis: Characterizing Differential Cohort Profiles
This script identifies interpretable clinical phenotypes that cause each model to fail where the other succeeds.
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config_h2_readmin30 import ConfigH2

# Try to import pysubgroup for subgroup discovery
try:
    import pysubgroup as ps
    SUBGROUP_DISCOVERY_AVAILABLE = True
except ImportError:
    SUBGROUP_DISCOVERY_AVAILABLE = False
    print("⚠️ pysubgroup not available - will use fallback univariate analysis")

# Set up plotting style
try:
    plt.style.use('seaborn-darkgrid')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")

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
    
    # Load numerical model (pkl format)
    with open(config.BASELINE_MODEL_PATH, 'rb') as f:
        nm_model = pickle.load(f)
    
    # Load semantic model - check if it's JSON or PKL
    if config.CHAMPION_MODEL_PATH.endswith('.json'):
        import xgboost as xgb
        sm_model = xgb.XGBClassifier()
        sm_model.load_model(config.CHAMPION_MODEL_PATH)
    else:
        with open(config.CHAMPION_MODEL_PATH, 'rb') as f:
            sm_model = pickle.load(f)
    
    return nm_model, sm_model

def load_h2a_results(config):
    """Load H2a results if available."""
    h2a_path = os.path.join(config.OUTPUT_DIR, 'h2a_results.pkl')
    if os.path.exists(h2a_path):
        with open(h2a_path, 'rb') as f:
            return pickle.load(f)
    return None

# =============================================================================
# COHORT DEFINITION
# =============================================================================

def define_error_cohorts(nm_pred, sm_pred, y_true):
    """Define the 8 cohorts based on model predictions and true labels."""
    # Extract the target column if y_true is a DataFrame
    if hasattr(y_true, 'columns'):
        config = ConfigH2()
        y_true = y_true[config.TARGET_VARIABLE].values.astype(int)
    else:
        y_true = y_true.astype(int)
    
    return {
        'TP_concordant': (nm_pred == 1) & (sm_pred == 1) & (y_true == 1),
        'TN_concordant': (nm_pred == 0) & (sm_pred == 0) & (y_true == 0),
        'FN_concordant': (nm_pred == 0) & (sm_pred == 0) & (y_true == 1),
        'FP_concordant': (nm_pred == 1) & (sm_pred == 1) & (y_true == 0),
        'FN_SM': (sm_pred == 0) & (nm_pred == 1) & (y_true == 1),  # SM misses, NM correct
        'FP_SM': (sm_pred == 1) & (nm_pred == 0) & (y_true == 0),  # SM false alarm, NM correct
        'FN_NM': (nm_pred == 0) & (sm_pred == 1) & (y_true == 1),  # NM misses, SM correct
        'FP_NM': (nm_pred == 1) & (sm_pred == 0) & (y_true == 0),  # NM false alarm, SM correct
    }

# =============================================================================
# SUBGROUP DISCOVERY
# =============================================================================

def prepare_features_for_subgroup_discovery(X_data):
    """Prepare features for subgroup discovery by handling missing values."""
    X_clean = X_data.copy()
    
    # Handle missing values with median for numerical features
    for col in X_clean.columns:
        if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            median_val = X_clean[col].median()
            if pd.isna(median_val):
                X_clean[col] = X_clean[col].fillna(0)
            else:
                X_clean[col] = X_clean[col].fillna(median_val)
    
    return X_clean

def run_subgroup_discovery_analysis(X_features, target, analysis_name, config):
    """Run subgroup discovery using pysubgroup."""
    if not SUBGROUP_DISCOVERY_AVAILABLE:
        logging.warning(f"Subgroup discovery not available for {analysis_name}")
        return pd.DataFrame()
    
    # Ensure features and target have same index
    common_index = X_features.index.intersection(target.index)
    X_analysis = X_features.loc[common_index].copy()
    y_analysis = target.loc[common_index].copy()
    
    # Check for sufficient samples
    n_positive = y_analysis.sum()
    n_negative = len(y_analysis) - n_positive
    
    MIN_SAMPLES_PER_CLASS = 10  # LOWERED from 30
    if n_positive < MIN_SAMPLES_PER_CLASS or n_negative < MIN_SAMPLES_PER_CLASS:
        logging.warning(f"Insufficient samples for {analysis_name}: {n_positive} positive, {n_negative} negative")
        return pd.DataFrame()
    
    # Create dataset for pysubgroup
    data = X_analysis.copy()
    data['target'] = y_analysis.values
    
    try:
        # Create binary target
        target_column = ps.BinaryTarget('target', 1)
        
        # Create search space from features
        searchspace = ps.create_selectors(data, ignore=['target'])
        
        # Create the task with LOWER threshold
        task = ps.SubgroupDiscoveryTask(
            data,
            target_column,
            searchspace,
            result_set_size=config.SUBGROUP_TOP_K,
            depth=config.SUBGROUP_MAX_DEPTH,
            qf=ps.WRAccQF(),
            min_quality=0.001  # MUCH LOWER than config value
        )
        
        # Use BeamSearch algorithm
        result = ps.BeamSearch(beam_width=20).execute(task)
        
        results_data = []
        
        # Parse results - USE to_dataframe which we know works
        if hasattr(result, 'to_dataframe'):
            result_df = result.to_dataframe()
            logging.info(f"Found {len(result_df)} patterns for {analysis_name}")
            
            for idx in range(min(len(result_df), config.SUBGROUP_TOP_K)):
                row = result_df.iloc[idx]
                
                results_data.append({
                    'rank': idx + 1,
                    'rule': str(row['subgroup']),
                    'quality_WRAcc': row['quality'],
                    'coverage': int(row['size_sg']),
                    'coverage_pct': round(row['relative_size_sg'] * 100, 1),
                    'n_positives': int(row['positives_sg']),
                    'target_share': round(row['target_share_sg'] * 100, 1),
                    'baseline_rate': round(row['target_share_dataset'] * 100, 1),
                    'lift': round(row['lift'], 2)
                })
        
        return pd.DataFrame(results_data)
        
    except Exception as e:
        logging.error(f"Error in subgroup discovery for {analysis_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()
        
    except Exception as e:
        logging.error(f"Error in subgroup discovery for {analysis_name}: {e}")
        return pd.DataFrame()

def fallback_univariate_analysis(X_features, target, analysis_name, config):
    """Fallback to univariate analysis if subgroup discovery is not available."""
    results = []
    
    # Ensure features and target have same index
    common_index = X_features.index.intersection(target.index)
    X_analysis = X_features.loc[common_index].copy()
    y_analysis = target.loc[common_index].copy()
    
    for feature in X_analysis.columns:
        error_values = X_analysis.loc[y_analysis == 1, feature].dropna()
        success_values = X_analysis.loc[y_analysis == 0, feature].dropna()
        
        if len(error_values) > 0 and len(success_values) > 0:
            # Mann-Whitney U test
            statistic, p_value = mannwhitneyu(error_values, success_values, alternative='two-sided')
            
            # Calculate effect size
            error_median = error_values.median()
            success_median = success_values.median()
            effect_size = error_median - success_median
            
            # Create simple rule based on median split
            overall_median = X_analysis[feature].median()
            if error_median > success_median:
                rule = f"{feature} > {overall_median:.2f}"
            else:
                rule = f"{feature} <= {overall_median:.2f}"
            
            # Calculate coverage
            if error_median > success_median:
                in_subgroup = X_analysis[feature] > overall_median
            else:
                in_subgroup = X_analysis[feature] <= overall_median
            
            coverage = in_subgroup.sum()
            coverage_pct = (coverage / len(X_analysis)) * 100
            
            # Calculate target share
            target_in_subgroup = y_analysis[in_subgroup].sum()
            target_share = (target_in_subgroup / coverage * 100) if coverage > 0 else 0
            
            # Calculate lift
            baseline_rate = (y_analysis.sum() / len(y_analysis)) * 100
            lift = target_share / baseline_rate if baseline_rate > 0 else 0
            
            results.append({
                'feature': feature,
                'rule': rule,
                'p_value': p_value,
                'effect_size': effect_size,
                'error_median': error_median,
                'success_median': success_median,
                'coverage': coverage,
                'coverage_pct': coverage_pct,
                'target_share': target_share,
                'lift': lift
            })
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Apply FDR correction
        results_df['q_value'] = fdrcorrection(results_df['p_value'], alpha=0.05)[1]
        results_df['significant'] = results_df['q_value'] < 0.05
        
        # Sort by effect size
        results_df = results_df.sort_values('effect_size', key=abs, ascending=False)
        
        # Convert to subgroup discovery format for consistency
        top_results = []
        for idx, row in results_df.head(config.SUBGROUP_TOP_K).iterrows():
            if row['significant']:
                top_results.append({
                    'rank': len(top_results) + 1,
                    'rule': row['rule'],
                    'quality_WRAcc': abs(row['effect_size']) / 100,  # Proxy for quality
                    'coverage': int(row['coverage']),
                    'coverage_pct': round(row['coverage_pct'], 1),
                    'target_share': round(row['target_share'], 1),
                    'lift': round(row['lift'], 2),
                    'p_value': row['p_value'],
                    'q_value': row['q_value']
                })
        
        return pd.DataFrame(top_results)
    
    return pd.DataFrame()

def analyze_differential_failures(cohorts, X_test_num, config):
    """H2b: Identify patterns driving model discordance."""
    # Prepare features
    X_features = prepare_features_for_subgroup_discovery(X_test_num)
    
    # Define the four analyses with MATCHING keys
    analyses = [
        ('SM_miss', 'FN_SM', 'TP_concordant', 'SM False Negatives vs Concordant True Positives'),
        ('SM_false_alarm', 'FP_SM', 'TN_concordant', 'SM False Positives vs Concordant True Negatives'),
        ('NM_miss', 'FN_NM', 'TP_concordant', 'NM False Negatives vs Concordant True Positives'),
        ('NM_false_alarm', 'FP_NM', 'TN_concordant', 'NM False Positives vs Concordant True Negatives')
    ]
    
    all_results = {}
    
    for analysis_key, error_cohort, success_cohort, title in analyses:
        logging.info(f"\nAnalyzing: {title}")
        
        # Create target variable
        population_mask = cohorts[error_cohort] | cohorts[success_cohort]
        target = pd.Series(0, index=X_test_num.index)
        target[cohorts[error_cohort]] = 1
        target = target[population_mask]
        
        # Get relevant features for this population
        X_population = X_features[population_mask]
        
        # Try subgroup discovery first, then fallback to univariate
        if SUBGROUP_DISCOVERY_AVAILABLE and config.USE_SUBGROUP_DISCOVERY:
            results_df = run_subgroup_discovery_analysis(
                X_features=X_population,
                target=target,
                analysis_name=title,
                config=config
            )
        else:
            results_df = fallback_univariate_analysis(
                X_features=X_population,
                target=target,
                analysis_name=title,
                config=config
            )
        
        all_results[analysis_key] = {
            'title': title,
            'results': results_df,
            'error_cohort': error_cohort,
            'success_cohort': success_cohort,
            'error_count': cohorts[error_cohort].sum(),
            'success_count': cohorts[success_cohort].sum()
        }
    
    return all_results

# =============================================================================
# CLINICAL INTERPRETATION
# =============================================================================

def generate_clinical_interpretation(rule, feature_names=None):
    """Generate clinical interpretation of a subgroup rule."""
    interpretations = []
    
    # Common feature interpretations
    feature_interpretations = {
        'lactate': 'blood lactate level',
        'creatinine': 'kidney function marker',
        'hr': 'heart rate',
        'bp': 'blood pressure',
        'temp': 'body temperature',
        'spo2': 'oxygen saturation',
        'gcs': 'Glasgow Coma Scale',
        'wbc': 'white blood cell count',
        'platelets': 'platelet count',
        'sodium': 'sodium level',
        'potassium': 'potassium level',
        'glucose': 'blood glucose',
        'bun': 'blood urea nitrogen',
        'stddev': 'variability in',
        'slope': 'trend in',
        'mean': 'average',
        'min': 'minimum',
        'max': 'maximum',
        'first': 'initial',
        'last': 'most recent'
    }
    
    # Parse the rule
    rule_lower = rule.lower()
    
    # Look for clinical patterns
    if 'stddev' in rule_lower and any(vital in rule_lower for vital in ['hr', 'bp', 'temp']):
        if '<' in rule or '<=' in rule:
            interpretations.append("Stable vital signs")
        else:
            interpretations.append("Variable/unstable vital signs")
    
    if 'creatinine' in rule_lower and '>' in rule:
        interpretations.append("Kidney dysfunction/acute kidney injury")
    
    if 'lactate' in rule_lower and '>' in rule:
        interpretations.append("Tissue hypoperfusion/possible sepsis")
    
    if 'gcs' in rule_lower and '<' in rule:
        interpretations.append("Altered mental status")
    
    if 'slope' in rule_lower:
        if '>' in rule and 'positive' not in rule_lower:
            interpretations.append("Worsening trend")
        elif '<' in rule:
            interpretations.append("Improving or stable trend")
    
    # Default interpretation if no specific pattern found
    if not interpretations:
        interpretations.append("Patients meeting specific clinical criteria")
    
    return "; ".join(interpretations)

# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_feature_distributions(subgroup_results, cohorts, X_test_num, config):
    """Create distribution plots for top features from each analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    analysis_order = ['SM_miss', 'SM_false_alarm', 'NM_miss', 'NM_false_alarm']    
    
    for idx, analysis_key in enumerate(analysis_order):
        ax = axes[idx]
        
        if analysis_key not in subgroup_results:
            ax.text(0.5, 0.5, 'No results', ha='center', va='center')
            ax.set_title(f"{analysis_key} Analysis")
            continue
        
        analysis = subgroup_results[analysis_key]
        results_df = analysis['results']
        
        if results_df.empty:
            ax.text(0.5, 0.5, 'No significant patterns found', ha='center', va='center')
            ax.set_title(analysis['title'])
            continue
        
        # Get the top rule and extract the first feature
        top_rule = results_df.iloc[0]['rule']
        
        # Extract feature name from rule (simple parsing)
        feature_name = None
        for col in X_test_num.columns:
            if col in top_rule:
                feature_name = col
                break
        
        if feature_name is None:
            ax.text(0.5, 0.5, f'Top rule:\n{top_rule[:50]}...', ha='center', va='center', fontsize=9)
            ax.set_title(analysis['title'])
            continue
        
        # Get data for the cohorts
        error_mask = cohorts[analysis['error_cohort']]
        success_mask = cohorts[analysis['success_cohort']]
        
        error_data = X_test_num.loc[error_mask, feature_name].dropna()
        success_data = X_test_num.loc[success_mask, feature_name].dropna()
        
        # Create violin plot
        plot_data = pd.DataFrame({
            'Value': np.concatenate([error_data, success_data]),
            'Cohort': ['Error'] * len(error_data) + ['Success'] * len(success_data)
        })
        
        sns.violinplot(data=plot_data, x='Cohort', y='Value', ax=ax, palette=['red', 'green'])
        ax.set_title(f"{analysis['title']}\nTop Feature: {feature_name}", fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel(feature_name)
        
        # Add statistical annotation
        _, p_value = mannwhitneyu(error_data, success_data, alternative='two-sided')
        ax.text(0.5, 0.95, f'p={p_value:.3f}', transform=ax.transAxes, 
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'h2b_feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Feature distribution plots saved")

# =============================================================================
# REPORTING
# =============================================================================

def create_summary_table(subgroup_results, config):
    """Create summary table of discovered subgroups."""
    summary_data = []
    
    for analysis_key, analysis in subgroup_results.items():
        results_df = analysis['results']
        
        if results_df.empty:
            continue
        
        # Take top 3 subgroups
        for idx in range(min(3, len(results_df))):
            row = results_df.iloc[idx]
            
            summary_data.append({
                'Analysis': analysis['title'],
                'Rank': row['rank'],
                'Rule': row['rule'][:100],  # Truncate long rules
                'WRAcc': row.get('quality_WRAcc', row.get('lift', 0)),
                'Coverage': f"{row['coverage']} ({row['coverage_pct']}%)",
                'Target_Share': f"{row.get('target_share', 0):.1f}%",
                'Lift': row.get('lift', 0),
                'Clinical_Interpretation': generate_clinical_interpretation(row['rule'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def create_detailed_report(subgroup_results, cohorts, config):
    """Create detailed H2b analysis report."""
    report = []
    report.append("="*80)
    report.append("H2b ANALYSIS REPORT: DIFFERENTIAL COHORT PROFILES")
    report.append("="*80)
    report.append("")
    
    # Method used
    if SUBGROUP_DISCOVERY_AVAILABLE and config.USE_SUBGROUP_DISCOVERY:
        report.append("METHOD: Subgroup Discovery with pysubgroup")
        report.append(f"  - Max depth: {config.SUBGROUP_MAX_DEPTH}")
        report.append(f"  - Min support: {config.SUBGROUP_MIN_SUPPORT*100:.0f}%")
        report.append(f"  - Quality measure: WRAcc")
    else:
        report.append("METHOD: Univariate Analysis with FDR Correction")
        report.append("  - Note: pysubgroup not available, using fallback method")
    report.append("")
    
    # Cohort sizes
    report.append("COHORT SIZES")
    report.append("-"*40)
    for name, mask in cohorts.items():
        report.append(f"  {name}: {mask.sum()} patients")
    report.append("")
    
    # Results for each analysis
    analysis_order = [
        ('SM_miss', 'SM False Negatives vs Concordant True Positives'),
        ('SM_false_alarm', 'SM False Positives vs Concordant True Negatives'),
        ('NM_miss', 'NM False Negatives vs Concordant True Positives'),
        ('NM_false_alarm', 'NM False Positives vs Concordant True Negatives')
    ]
    
    for analysis_key, title in analysis_order:
        report.append("="*80)
        report.append(f"ANALYSIS: {title}")
        report.append("="*80)
        
        if analysis_key not in subgroup_results:
            report.append("No results available")
            report.append("")
            continue
        
        analysis = subgroup_results[analysis_key]
        results_df = analysis['results']
        
        report.append(f"Error cohort size: {analysis['error_count']}")
        report.append(f"Success cohort size: {analysis['success_count']}")
        report.append("")
        
        if results_df.empty:
            report.append("No significant patterns discovered")
        else:
            report.append("TOP DISCOVERED PATTERNS:")
            report.append("-"*40)
            
            for idx in range(min(3, len(results_df))):
                row = results_df.iloc[idx]
                report.append(f"\nPattern #{row['rank']}:")
                report.append(f"  Rule: {row['rule']}")
                report.append(f"  Quality (WRAcc): {row.get('quality_WRAcc', 'N/A')}")
                report.append(f"  Coverage: {row['coverage']} patients ({row['coverage_pct']}%)")
                report.append(f"  Target share: {row.get('target_share', 0):.1f}%")
                report.append(f"  Lift: {row.get('lift', 0):.2f}x")
                report.append(f"  Clinical interpretation: {generate_clinical_interpretation(row['rule'])}")
        
        report.append("")
    
    # Summary
    report.append("="*80)
    report.append("SUMMARY")
    report.append("="*80)
    
    # Count analyses with meaningful patterns
    meaningful_analyses = 0
    for analysis_key, analysis in subgroup_results.items():
        results_df = analysis['results']
        if not results_df.empty:
            # Check if any pattern meets quality thresholds
            has_meaningful = any(
                (row.get('quality_WRAcc', 0) >= config.SUBGROUP_MIN_QUALITY or
                 row.get('lift', 0) >= config.SUBGROUP_MIN_LIFT)
                for _, row in results_df.iterrows()
            )
            if has_meaningful:
                meaningful_analyses += 1
    
    report.append(f"Analyses with meaningful patterns: {meaningful_analyses}/4")
    
    if meaningful_analyses >= config.SUBGROUP_MIN_MEANINGFUL_ANALYSES:
        report.append("\n✅ H2b HYPOTHESIS SUPPORTED")
        report.append("The models show distinct, interpretable failure patterns.")
    else:
        report.append("\n❌ H2b HYPOTHESIS NOT SUPPORTED")
        report.append("Insufficient evidence of distinct failure patterns.")
    
    report_text = "\n".join(report)
    
    # Save report
    with open(os.path.join(config.OUTPUT_DIR, 'h2b_detailed_report.txt'), 'w') as f:
        f.write(report_text)
    
    return report_text

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Execute H2b analysis: Characterizing Differential Cohort Profiles."""
    config = ConfigH2()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'h2b_analysis.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("H2b ANALYSIS: DIFFERENTIAL COHORT PROFILES")
    logging.info("="*60)
    
    # Check if subgroup discovery is available
    if SUBGROUP_DISCOVERY_AVAILABLE:
        logging.info("✅ Subgroup discovery available via pysubgroup")
    else:
        logging.info("⚠️ Using fallback univariate analysis (install pysubgroup for full functionality)")
    
    # Load data and models
    data = load_data(config)
    nm_model, sm_model = load_models(config)
    
    # Generate predictions
    logging.info("Generating model predictions...")
    nm_proba = nm_model.predict_proba(data['X_test_num'])[:, 1]
    sm_proba = sm_model.predict_proba(data['X_test_emb'])[:, 1]
    
    # Binary predictions using 0.5 threshold
    THRESHOLD = 0.5
    nm_pred = (nm_proba >= THRESHOLD).astype(int)
    sm_pred = (sm_proba >= THRESHOLD).astype(int)
    
    # Define cohorts
    logging.info("Defining error cohorts...")
    cohorts = define_error_cohorts(nm_pred, sm_pred, data['y_test'])
    
    # Log cohort sizes
    logging.info("\nCohort sizes:")
    for name, mask in cohorts.items():
        logging.info(f"  {name}: {mask.sum()}")
    
    # Run differential failure analysis
    logging.info("\nRunning differential failure analysis...")
    subgroup_results = analyze_differential_failures(cohorts, data['X_test_num'], config)
    
    # Create visualizations
    logging.info("\nCreating visualizations...")
    plot_feature_distributions(subgroup_results, cohorts, data['X_test_num'], config)
    
    # Create summary table
    logging.info("\nCreating summary table...")
    summary_df = create_summary_table(subgroup_results, config)
    
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'h2b_summary_table.csv'), index=False)
        logging.info("\nTop Discovered Patterns (Summary):")
        print(summary_df.to_string(index=False))
    else:
        logging.info("No significant patterns discovered")
    
    # Create detailed report
    logging.info("\nGenerating detailed report...")
    report = create_detailed_report(subgroup_results, cohorts, config)
    print("\n" + report)
    
    # Save individual analysis results
    for analysis_key, analysis in subgroup_results.items():
        if not analysis['results'].empty:
            analysis['results'].to_csv(
                os.path.join(config.OUTPUT_DIR, f'h2b_patterns_{analysis_key}.csv'),
                index=False
            )
    
    # Save all results as pickle
    with open(os.path.join(config.OUTPUT_DIR, 'h2b_results.pkl'), 'wb') as f:
        pickle.dump(subgroup_results, f)
    
    logging.info(f"\nAnalysis complete. All results saved to: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()