"""
H2a Analysis: Quantifying Error Discordance Between NM and SM Models
This script focuses on statistically proving that models disagree in a non-random way.
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, bootstrap
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config_h2_readmin30 import ConfigH2

# Set up plotting style - use compatible style for older matplotlib versions
try:
    plt.style.use('seaborn-darkgrid')
except:
    plt.style.use('ggplot')  # Fallback if seaborn styles not available
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

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def calculate_performance_metrics(y_true, y_proba, model_name="Model"):
    """Calculate AUROC, AUPRC, and Brier score for a model."""
    
    # Extract target column if y_true is a DataFrame
    if hasattr(y_true, 'columns'):
        config = ConfigH2()
        y_true_binary = y_true[config.TARGET_VARIABLE].values.astype(int)
    else:
        y_true_binary = y_true.astype(int)
    
    metrics = {
        'auroc': roc_auc_score(y_true_binary, y_proba),
        'auprc': average_precision_score(y_true_binary, y_proba),
        'brier': brier_score_loss(y_true_binary, y_proba)
    }
    
    return metrics

def create_disagreement_matrix(nm_pred, sm_pred, y_true):
    """
    Create a 2x2 disagreement matrix showing the four outcomes of binary predictions.
    
    Returns:
        dict: Contains the disagreement matrix and detailed cohort masks
    """
    # Extract the target column if y_true is a DataFrame
    if hasattr(y_true, 'columns'):
        config = ConfigH2()
        y_true = y_true[config.TARGET_VARIABLE].values.astype(int)
    else:
        y_true = y_true.astype(int)
    
    # Define the four cohorts based on correctness
    nm_correct = (nm_pred == y_true)
    sm_correct = (sm_pred == y_true)
    
    # Create 2x2 disagreement matrix
    matrix = pd.DataFrame(
        index=['NM Correct', 'NM Incorrect'],
        columns=['SM Correct', 'SM Incorrect']
    )
    
    matrix.loc['NM Correct', 'SM Correct'] = np.sum(nm_correct & sm_correct)
    matrix.loc['NM Correct', 'SM Incorrect'] = np.sum(nm_correct & ~sm_correct)
    matrix.loc['NM Incorrect', 'SM Correct'] = np.sum(~nm_correct & sm_correct)
    matrix.loc['NM Incorrect', 'SM Incorrect'] = np.sum(~nm_correct & ~sm_correct)
    
    # Also create detailed cohorts for visualization
    cohorts = {
        'concordant_success': nm_correct & sm_correct,
        'sm_only_error': nm_correct & ~sm_correct,
        'nm_only_error': ~nm_correct & sm_correct,
        'concordant_failure': ~nm_correct & ~sm_correct
    }
    
    return {
        'matrix': matrix,
        'cohorts': cohorts,
        'nm_correct': nm_correct,
        'sm_correct': sm_correct
    }

def perform_mcnemar_test(nm_pred, sm_pred, y_true):
    """
    Perform McNemar's test to determine if the models' errors are significantly different.
    
    Returns:
        dict: Test statistics including p-value and interpretation
    """
    # Extract the target column if y_true is a DataFrame
    if hasattr(y_true, 'columns'):
        config = ConfigH2()
        y_true = y_true[config.TARGET_VARIABLE].values.astype(int)
    else:
        y_true = y_true.astype(int)
    
    nm_correct = (nm_pred == y_true)
    sm_correct = (sm_pred == y_true)
    
    # Create contingency table for McNemar's test
    # Format: [[both_correct, nm_only_correct], [sm_only_correct, both_incorrect]]
    contingency = np.array([
        [np.sum(nm_correct & sm_correct), np.sum(nm_correct & ~sm_correct)],
        [np.sum(~nm_correct & sm_correct), np.sum(~nm_correct & ~sm_correct)]
    ])
    
    # Perform McNemar's test
    result = mcnemar(contingency)
    
    # Calculate the exact counts for interpretation
    nm_only_correct = np.sum(nm_correct & ~sm_correct)
    sm_only_correct = np.sum(~nm_correct & sm_correct)
    
    return {
        'statistic': result.statistic,
        'p_value': result.pvalue,
        'nm_only_correct': nm_only_correct,
        'sm_only_correct': sm_only_correct,
        'contingency_table': contingency,
        'significant': result.pvalue < 0.05,
        'interpretation': 'Models disagree significantly' if result.pvalue < 0.05 else 'No significant disagreement'
    }
    
def calculate_cohens_kappa_with_ci(nm_pred, sm_pred, n_bootstrap=1000):
    """
    Calculate Cohen's Kappa with 95% confidence interval using bootstrap.
    
    Returns:
        dict: Kappa value and confidence interval
    """
    kappa = cohen_kappa_score(nm_pred, sm_pred)
    
    # Bootstrap for confidence interval
    def kappa_statistic(indices):
        return cohen_kappa_score(nm_pred[indices], sm_pred[indices])
    
    rng = np.random.RandomState(42)
    indices = np.arange(len(nm_pred))
    bootstrap_kappas = []
    
    for _ in range(n_bootstrap):
        boot_indices = rng.choice(indices, size=len(indices), replace=True)
        bootstrap_kappas.append(kappa_statistic(boot_indices))
    
    ci_lower = np.percentile(bootstrap_kappas, 2.5)
    ci_upper = np.percentile(bootstrap_kappas, 97.5)
    
    return {
        'kappa': kappa,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'interpretation': interpret_kappa(kappa)
    }

def interpret_kappa(kappa):
    """Interpret Cohen's Kappa value according to standard guidelines."""
    if kappa < 0:
        return "Poor agreement (worse than chance)"
    elif kappa <= 0.20:
        return "Slight agreement"
    elif kappa <= 0.40:
        return "Fair agreement"
    elif kappa <= 0.60:
        return "Moderate agreement"
    elif kappa <= 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def calculate_pearson_correlation_with_ci(nm_proba, sm_proba, n_bootstrap=1000):
    """
    Calculate Pearson correlation with 95% confidence interval using bootstrap.
    
    Returns:
        dict: Correlation coefficient and confidence interval
    """
    correlation, p_value = pearsonr(nm_proba, sm_proba)
    
    # Bootstrap for confidence interval
    rng = np.random.RandomState(42)
    indices = np.arange(len(nm_proba))
    bootstrap_correlations = []
    
    for _ in range(n_bootstrap):
        boot_indices = rng.choice(indices, size=len(indices), replace=True)
        boot_corr, _ = pearsonr(nm_proba[boot_indices], sm_proba[boot_indices])
        bootstrap_correlations.append(boot_corr)
    
    ci_lower = np.percentile(bootstrap_correlations, 2.5)
    ci_upper = np.percentile(bootstrap_correlations, 97.5)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'interpretation': interpret_correlation(correlation)
    }

def interpret_correlation(correlation):
    """Interpret Pearson correlation coefficient."""
    abs_corr = abs(correlation)
    if abs_corr >= 0.9:
        strength = "Very strong"
    elif abs_corr >= 0.7:
        strength = "Strong"
    elif abs_corr >= 0.5:
        strength = "Moderate"
    elif abs_corr >= 0.3:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    direction = "positive" if correlation > 0 else "negative"
    return f"{strength} {direction} correlation"

# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_probability_scatter(nm_proba, sm_proba, y_true, output_dir):
    """
    Create scatter plot of NM vs SM predicted probabilities.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Extract the target column if y_true is a DataFrame
    if hasattr(y_true, 'columns'):
        config = ConfigH2()
        y_true = y_true[config.TARGET_VARIABLE].values
    else:
        y_true = y_true.values if hasattr(y_true, 'values') else y_true
    
    # Create color map for true outcomes
    colors = ['blue' if y == 0 else 'red' for y in y_true]
    
    # Create scatter plot
    scatter = ax.scatter(nm_proba, sm_proba, c=colors, alpha=0.5, s=20)
    
    # Add diagonal line for perfect agreement
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Agreement')
    
    # Add threshold lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('NM Predicted Probability', fontsize=12)
    ax.set_ylabel('SM Predicted Probability', fontsize=12)
    ax.set_title('Model Prediction Comparison\n(Red: Positive Outcome, Blue: Negative Outcome)', fontsize=14)
    
    # Add correlation text
    correlation, _ = pearsonr(nm_proba, sm_proba)
    ax.text(0.05, 0.95, f'Pearson r = {correlation:.3f}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='Positive Outcome'),
        Patch(facecolor='blue', alpha=0.5, label='Negative Outcome'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Perfect Agreement')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_scatter_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Probability scatter plot saved to {output_dir}")

def plot_disagreement_heatmap(matrix, output_dir):
    """
    Create heatmap visualization of the disagreement matrix.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Convert matrix to numpy for heatmap
    matrix_values = matrix.values.astype(int)
    
    # Create heatmap
    sns.heatmap(matrix_values, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=matrix.columns, yticklabels=matrix.index,
                cbar_kws={'label': 'Count'}, ax=ax, square=True,
                linewidths=1, linecolor='gray')
    
    ax.set_title('Model Disagreement Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Numerical Model (NM)', fontsize=12)
    ax.set_xlabel('Semantic Model (SM)', fontsize=12)
    
    # Calculate percentages for annotations
    total = matrix_values.sum()
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            count = matrix_values[i, j]
            percentage = (count / total) * 100
            current_text = ax.texts[i * len(matrix.columns) + j]
            current_text.set_text(f'{count}\n({percentage:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disagreement_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Disagreement heatmap saved to {output_dir}")

def create_summary_report(results, output_dir):
    """
    Create a formatted summary report of all H2a results.
    """
    report = []
    report.append("="*60)
    report.append("H2a ANALYSIS REPORT: ERROR DISCORDANCE QUANTIFICATION")
    report.append("="*60)
    report.append("")
    
    # McNemar's Test Results
    report.append("MCNEMAR'S TEST")
    report.append("-"*30)
    report.append(f"Test Statistic: {results['mcnemar']['statistic']:.4f}")
    report.append(f"P-value: {results['mcnemar']['p_value']:.4f}")
    report.append(f"Result: {results['mcnemar']['interpretation']}")
    report.append(f"NM-only correct: {results['mcnemar']['nm_only_correct']} cases")
    report.append(f"SM-only correct: {results['mcnemar']['sm_only_correct']} cases")
    report.append("")
    
    # Cohen's Kappa Results
    report.append("COHEN'S KAPPA")
    report.append("-"*30)
    report.append(f"Kappa: {results['kappa']['kappa']:.4f}")
    report.append(f"95% CI: [{results['kappa']['ci_lower']:.4f}, {results['kappa']['ci_upper']:.4f}]")
    report.append(f"Interpretation: {results['kappa']['interpretation']}")
    report.append("")
    
    # Pearson Correlation Results
    report.append("PEARSON CORRELATION")
    report.append("-"*30)
    report.append(f"Correlation: {results['correlation']['correlation']:.4f}")
    report.append(f"95% CI: [{results['correlation']['ci_lower']:.4f}, {results['correlation']['ci_upper']:.4f}]")
    report.append(f"P-value: {results['correlation']['p_value']:.4f}")
    report.append(f"Interpretation: {results['correlation']['interpretation']}")
    report.append("")
    
    # Disagreement Matrix Summary
    report.append("DISAGREEMENT MATRIX SUMMARY")
    report.append("-"*30)
    matrix = results['disagreement']['matrix']
    total = matrix.values.sum()
    
    for idx in matrix.index:
        for col in matrix.columns:
            count = matrix.loc[idx, col]
            pct = (count / total) * 100
            report.append(f"{idx} & {col}: {count} ({pct:.1f}%)")
    
    report.append("")
    report.append(f"Total concordant predictions: {results['concordant_count']} ({results['concordant_pct']:.1f}%)")
    report.append(f"Total discordant predictions: {results['discordant_count']} ({results['discordant_pct']:.1f}%)")
    
    # Save report
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'h2a_summary_report.txt'), 'w') as f:
        f.write(report_text)
    
    # Also print to console
    print(report_text)
    
    return report_text

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Execute H2a analysis: Quantifying Error Discordance."""
    config = ConfigH2()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'h2a_analysis.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("H2a ANALYSIS: QUANTIFYING ERROR DISCORDANCE")
    logging.info("="*60)
    
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
    
    logging.info(f"Using threshold: {THRESHOLD}")
    
    # Create disagreement matrix
    logging.info("\nCreating disagreement matrix...")
    disagreement_results = create_disagreement_matrix(nm_pred, sm_pred, data['y_test'])
    
    # Perform McNemar's test
    logging.info("Performing McNemar's test...")
    mcnemar_results = perform_mcnemar_test(nm_pred, sm_pred, data['y_test'])
    
    # Calculate Cohen's Kappa with CI
    logging.info("Calculating Cohen's Kappa with confidence interval...")
    kappa_results = calculate_cohens_kappa_with_ci(nm_pred, sm_pred, config.N_BOOTSTRAP)
    
    # Calculate Pearson correlation with CI
    logging.info("Calculating Pearson correlation with confidence interval...")
    correlation_results = calculate_pearson_correlation_with_ci(nm_proba, sm_proba, config.N_BOOTSTRAP)
    
    # Calculate summary statistics
    matrix = disagreement_results['matrix']
    total = matrix.values.sum()
    concordant_count = matrix.loc['NM Correct', 'SM Correct'] + matrix.loc['NM Incorrect', 'SM Incorrect']
    discordant_count = matrix.loc['NM Correct', 'SM Incorrect'] + matrix.loc['NM Incorrect', 'SM Correct']

    nm_metrics = calculate_performance_metrics(data['y_test'], nm_proba, "NM")
    sm_metrics = calculate_performance_metrics(data['y_test'], sm_proba, "SM")

    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Numerical Model (NM):")
    print(f"  AUROC: {nm_metrics['auroc']:.4f}")
    print(f"  AUPRC: {nm_metrics['auprc']:.4f}")
    print(f"  Brier: {nm_metrics['brier']:.4f}")
    print(f"\nSemantic Model (SM):")
    print(f"  AUROC: {sm_metrics['auroc']:.4f}")
    print(f"  AUPRC: {sm_metrics['auprc']:.4f}")
    print(f"  Brier: {sm_metrics['brier']:.4f}")
    
    # Compile all results
    results = {
        'mcnemar': mcnemar_results,
        'kappa': kappa_results,
        'correlation': correlation_results,
        'disagreement': disagreement_results,
        'concordant_count': concordant_count,
        'concordant_pct': (concordant_count / total) * 100,
        'discordant_count': discordant_count,
        'discordant_pct': (discordant_count / total) * 100
    }
    
    # Create visualizations
    logging.info("\nCreating visualizations...")
    plot_probability_scatter(nm_proba, sm_proba, data['y_test'], config.OUTPUT_DIR)
    plot_disagreement_heatmap(disagreement_results['matrix'], config.OUTPUT_DIR)
    
    # Create summary report
    logging.info("\nGenerating summary report...")
    create_summary_report(results, config.OUTPUT_DIR)
    
    # Save detailed results to CSV
    disagreement_results['matrix'].to_csv(
        os.path.join(config.OUTPUT_DIR, 'disagreement_matrix.csv')
    )
    
    # Save all results as pickle for potential reuse
    with open(os.path.join(config.OUTPUT_DIR, 'h2a_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save key metrics to CSV for easy reference
    metrics_df = pd.DataFrame([{
        'mcnemar_pvalue': mcnemar_results['p_value'],
        'mcnemar_significant': mcnemar_results['significant'],
        'cohens_kappa': kappa_results['kappa'],
        'kappa_ci_lower': kappa_results['ci_lower'],
        'kappa_ci_upper': kappa_results['ci_upper'],
        'pearson_correlation': correlation_results['correlation'],
        'correlation_ci_lower': correlation_results['ci_lower'],
        'correlation_ci_upper': correlation_results['ci_upper'],
        'concordant_pct': results['concordant_pct'],
        'discordant_pct': results['discordant_pct']
    }])
    metrics_df.to_csv(os.path.join(config.OUTPUT_DIR, 'h2a_metrics.csv'), index=False)
    
    logging.info(f"\nAnalysis complete. All results saved to: {config.OUTPUT_DIR}")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL H2a RESULTS SUMMARY")
    print("="*60)
    print(f"McNemar's test p-value: {mcnemar_results['p_value']:.4f} - {mcnemar_results['interpretation']}")
    print(f"Cohen's Kappa: {kappa_results['kappa']:.4f} [{kappa_results['ci_lower']:.4f}, {kappa_results['ci_upper']:.4f}]")
    print(f"Pearson Correlation: {correlation_results['correlation']:.4f} [{correlation_results['ci_lower']:.4f}, {correlation_results['ci_upper']:.4f}]")
    print(f"Concordant predictions: {results['concordant_pct']:.1f}%")
    print(f"Discordant predictions: {results['discordant_pct']:.1f}%")

if __name__ == "__main__":
    main()