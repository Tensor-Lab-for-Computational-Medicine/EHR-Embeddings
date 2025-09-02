
"""
Visualization script for XGBoost MODEL4B results across different prediction tasks.
This script loads results from pickle files and creates comparison plots.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects
import seaborn as sns
from pathlib import Path
import logging
import re
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set professional plotting style
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3
})

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "xgboost_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Task labels for better visualization
TASK_LABELS = {
    'mort_hosp': 'Hospital Mortality',
    'los_3': 'Length-of-Stay > 3 Days', 
    'los_7': 'Length-of-Stay > 7 Days',
    'readmission_30': '30-Day Readmission',
    'intervention_vent': 'Mechanical Ventilation',
    'intervention_vaso': 'Vasopressor Administration'
}

def load_results():
    """Load XGBoost results from all task directories."""
    results = []
    
    # Find all XGBoost result directories
    result_dirs = list(BASE_DIR.glob("xgboost_results_model4b_*"))
    
    for result_dir in result_dirs:
        # Extract task name from directory name
        task_match = re.match(r'xgboost_results_model4b_(.+)', result_dir.name)
        if not task_match:
            continue
        task = task_match.group(1)
        
        # Look for results pickle files
        result_files = list(result_dir.glob("results_*.pkl"))
        
        for result_file in result_files:
            try:
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract arm from filename 
                arm_match = re.match(r'results_(.+)\.pkl', result_file.name)
                arm = arm_match.group(1) if arm_match else "Unknown"
                
                # Extract metrics with error handling
                full_eval = data.get('full_evaluation', data)
                auroc_data = full_eval.get('auroc', {})
                auprc_data = full_eval.get('auprc', {})
                
                results.append({
                    'Task': task,
                    'Arm': arm,
                    'AUROC': auroc_data.get('point_estimate'),
                    'AUROC_CI_Lower': auroc_data.get('ci_lower'),
                    'AUROC_CI_Upper': auroc_data.get('ci_upper'),
                    'AUPRC': auprc_data.get('point_estimate'),
                    'AUPRC_CI_Lower': auprc_data.get('ci_lower'),
                    'AUPRC_CI_Upper': auprc_data.get('ci_upper'),
                })
                
                logging.info(f"Loaded {task} - {arm}: AUROC={auroc_data.get('point_estimate', 'N/A'):.4f}")
                
            except Exception as e:
                logging.warning(f"Failed to load {result_file}: {e}")
    
    return pd.DataFrame(results)

def create_performance_comparison(df, save_path):
    """Create professional bar plot comparing AUROC across tasks."""
    df_plot = df.copy()
    df_plot['TaskLabel'] = df_plot['Task'].map(TASK_LABELS)
    
    # Sort by AUROC for better visualization
    df_plot = df_plot.sort_values('AUROC', ascending=True)
    
    # Create figure with professional styling
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    
    # Professional color palette - gradient from low to high performance
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df_plot)))
    
    # Create horizontal bar plot with gradient colors
    bars = ax.barh(df_plot['TaskLabel'], df_plot['AUROC'], 
                   color=colors, edgecolor='#2C3E50', linewidth=1.5, alpha=0.85)
    
    # Add subtle shadow effect
    for bar in bars:
        bar.set_path_effects([
            plt.matplotlib.patheffects.SimplePatchShadow(offset=(2, -2), shadow_rgbFace='#CCCCCC', alpha=0.3),
            plt.matplotlib.patheffects.Normal()
        ])
    
    # Add confidence intervals with professional styling
    for i, (_, row) in enumerate(df_plot.iterrows()):
        if pd.notna(row['AUROC_CI_Lower']) and pd.notna(row['AUROC_CI_Upper']):
            ax.errorbar(row['AUROC'], i, 
                       xerr=[[row['AUROC'] - row['AUROC_CI_Lower']], 
                             [row['AUROC_CI_Upper'] - row['AUROC']]],
                       fmt='none', color='#2C3E50', capsize=6, capthick=2.5, 
                       elinewidth=2.5, alpha=0.8, zorder=10)
    
    # Add value labels positioned after error bars
    for i, (_, row) in enumerate(df_plot.iterrows()):
        text_x = row['AUROC_CI_Upper'] + 0.02 if pd.notna(row['AUROC_CI_Upper']) else row['AUROC'] + 0.02
        ax.text(text_x, i, f"{row['AUROC']:.3f}", 
                va='center', ha='left', fontweight='bold', 
                fontsize=11, color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.8))
    
    # Professional styling
    ax.set_xlabel('Test Set AUROC (with 95% Confidence Interval)', fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Prediction Task', fontweight='bold', color='#2C3E50')
    ax.set_title('XGBoost MODEL4B Performance Comparison\nAcross Clinical Prediction Tasks', 
                fontweight='bold', color='#2C3E50', pad=25)
    
    # Enhanced grid
    ax.grid(axis='x', alpha=0.4, linestyle='-', linewidth=0.8, color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # Set professional axis limits
    max_ci_upper = df_plot['AUROC_CI_Upper'].max() if df_plot['AUROC_CI_Upper'].notna().any() else df_plot['AUROC'].max()
    ax.set_xlim(0.45, max_ci_upper + 0.12)
    
    # Add reference line at 0.5 (random performance)
    ax.axvline(x=0.5, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7, 
               label='Random Performance (0.5)', zorder=1)
    
    # Professional legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, 
              facecolor='white', edgecolor='#BDC3C7')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    logging.info(f"Professional AUROC comparison plot saved to {save_path}")

def create_metrics_comparison(df, save_path):
    """Create professional side-by-side comparison of AUROC and AUPRC."""
    df_plot = df.copy()
    df_plot['TaskLabel'] = df_plot['Task'].map(TASK_LABELS)
    
    # Create figure with professional styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor('white')
    
    # Professional color schemes
    auroc_colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(df_plot)))
    auprc_colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df_plot)))
    
    # AUROC plot with enhanced styling
    bars1 = ax1.bar(range(len(df_plot)), df_plot['AUROC'], 
                    color=auroc_colors, edgecolor='#2C3E50', linewidth=1.5, alpha=0.85)
    
    # Add shadow effects
    for bar in bars1:
        bar.set_path_effects([
            plt.matplotlib.patheffects.SimplePatchShadow(offset=(1, -1), shadow_rgbFace='#CCCCCC', alpha=0.3),
            plt.matplotlib.patheffects.Normal()
        ])
    
    # Enhanced error bars
    ax1.errorbar(range(len(df_plot)), df_plot['AUROC'],
                yerr=[df_plot['AUROC'] - df_plot['AUROC_CI_Lower'],
                      df_plot['AUROC_CI_Upper'] - df_plot['AUROC']],
                fmt='none', color='#2C3E50', capsize=5, capthick=2, 
                elinewidth=2, alpha=0.8, zorder=10)
    
    # Professional styling for AUROC plot
    ax1.set_xlabel('Prediction Task', fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('Test Set AUROC (with 95% CI)', fontweight='bold', color='#2C3E50')
    ax1.set_title('Area Under ROC Curve', fontweight='bold', color='#C0392B', pad=20)
    ax1.set_xticks(range(len(df_plot)))
    ax1.set_xticklabels(df_plot['TaskLabel'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.8, color='#BDC3C7')
    ax1.set_axisbelow(True)
    
    # Add reference line at 0.5
    ax1.axhline(y=0.5, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, 
                label='Random (0.5)', zorder=1)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Value labels for AUROC
    for i, (_, row) in enumerate(df_plot.iterrows()):
        text_y = row['AUROC_CI_Upper'] + 0.02 if pd.notna(row['AUROC_CI_Upper']) else row['AUROC'] + 0.02
        ax1.text(i, text_y, f"{row['AUROC']:.3f}", ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='#C0392B', alpha=0.8))
    
    # Set y-axis limits for AUROC
    max_ci_upper = df_plot['AUROC_CI_Upper'].max() if df_plot['AUROC_CI_Upper'].notna().any() else df_plot['AUROC'].max()
    ax1.set_ylim(0.4, max_ci_upper + 0.08)
    
    # AUPRC plot with enhanced styling
    bars2 = ax2.bar(range(len(df_plot)), df_plot['AUPRC'], 
                    color=auprc_colors, edgecolor='#2C3E50', linewidth=1.5, alpha=0.85)
    
    # Add shadow effects
    for bar in bars2:
        bar.set_path_effects([
            plt.matplotlib.patheffects.SimplePatchShadow(offset=(1, -1), shadow_rgbFace='#CCCCCC', alpha=0.3),
            plt.matplotlib.patheffects.Normal()
        ])
    
    # Enhanced error bars
    ax2.errorbar(range(len(df_plot)), df_plot['AUPRC'],
                yerr=[df_plot['AUPRC'] - df_plot['AUPRC_CI_Lower'],
                      df_plot['AUPRC_CI_Upper'] - df_plot['AUPRC']],
                fmt='none', color='#2C3E50', capsize=5, capthick=2, 
                elinewidth=2, alpha=0.8, zorder=10)
    
    # Professional styling for AUPRC plot
    ax2.set_xlabel('Prediction Task', fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('Test Set AUPRC (with 95% CI)', fontweight='bold', color='#2C3E50')
    ax2.set_title('Area Under Precision-Recall Curve', fontweight='bold', color='#2980B9', pad=20)
    ax2.set_xticks(range(len(df_plot)))
    ax2.set_xticklabels(df_plot['TaskLabel'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.8, color='#BDC3C7')
    ax2.set_axisbelow(True)
    
    # Value labels for AUPRC
    for i, (_, row) in enumerate(df_plot.iterrows()):
        text_y = row['AUPRC_CI_Upper'] + 0.02 if pd.notna(row['AUPRC_CI_Upper']) else row['AUPRC'] + 0.02
        ax2.text(i, text_y, f"{row['AUPRC']:.3f}", ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='#2980B9', alpha=0.8))
    
    # Set y-axis limits for AUPRC
    max_ci_upper = df_plot['AUPRC_CI_Upper'].max() if df_plot['AUPRC_CI_Upper'].notna().any() else df_plot['AUPRC'].max()
    ax2.set_ylim(0, max_ci_upper + 0.08)
    
    # Clean up spines for both plots
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#BDC3C7')
        ax.spines['bottom'].set_color('#BDC3C7')
    
    plt.suptitle('XGBoost MODEL4B Performance Evaluation\nClinical Prediction Task Comparison', 
                fontsize=16, fontweight='bold', color='#2C3E50', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    logging.info(f"Professional metrics comparison plot saved to {save_path}")

def create_summary_table(df, save_path):
    """Create and save a professional summary table."""
    df_table = df.copy()
    df_table['TaskLabel'] = df_table['Task'].map(TASK_LABELS)
    
    # Format metrics with confidence intervals
    df_table['AUROC (95% CI)'] = df_table.apply(
        lambda row: f"{row['AUROC']:.4f} ({row['AUROC_CI_Lower']:.4f}-{row['AUROC_CI_Upper']:.4f})",
        axis=1
    )
    df_table['AUPRC (95% CI)'] = df_table.apply(
        lambda row: f"{row['AUPRC']:.4f} ({row['AUPRC_CI_Lower']:.4f}-{row['AUPRC_CI_Upper']:.4f})",
        axis=1
    )
    
    # Select and order columns
    summary_cols = ['TaskLabel', 'Arm', 'AUROC (95% CI)', 'AUPRC (95% CI)']
    df_summary = df_table[summary_cols].rename(columns={
        'TaskLabel': 'Clinical Prediction Task',
        'Arm': 'Model Configuration'
    })
    
    # Sort by AUROC for better readability (extracting numeric value for sorting)
    df_summary['AUROC_numeric'] = df_table['AUROC']
    df_summary = df_summary.sort_values('AUROC_numeric', ascending=False).drop('AUROC_numeric', axis=1)
    
    # Save as CSV and markdown
    df_summary.to_csv(save_path.with_suffix('.csv'), index=False)
    
    with open(save_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
        f.write("# XGBoost MODEL4B Performance Summary\n")
        f.write("## Clinical Prediction Task Results\n\n")
        f.write("This table presents the performance of XGBoost MODEL4B across different clinical prediction tasks. ")
        f.write("All metrics are reported with 95% confidence intervals.\n\n")
        f.write("**Performance Metrics:**\n")
        f.write("- **AUROC**: Area Under the Receiver Operating Characteristic Curve\n")
        f.write("- **AUPRC**: Area Under the Precision-Recall Curve\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write(f"\n\n*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    logging.info(f"Professional summary table saved to {save_path.with_suffix('.csv')} and {save_path.with_suffix('.md')}")
    return df_summary

def main():
    """Main execution function."""
    print("="*80)
    print("XGBoost MODEL4B Results Visualization")
    print("="*80)
    logging.info("Starting professional visualization generation...")
    
    # Load all results
    df = load_results()
    
    if df.empty:
        logging.error("❌ No results found! Check that result files exist.")
        return
    
    logging.info(f"✅ Successfully loaded {len(df)} result records")
    print(f"\n📊 Data Summary:")
    print("-" * 60)
    summary_data = df[['Task', 'Arm', 'AUROC', 'AUPRC']].copy()
    summary_data['Task'] = summary_data['Task'].map(TASK_LABELS)
    print(summary_data.to_string(index=False, float_format='%.4f'))
    
    print(f"\n🎨 Generating professional visualizations...")
    print("-" * 60)
    
    # Create visualizations
    create_performance_comparison(df, OUTPUT_DIR / "professional_auroc_comparison.png")
    create_metrics_comparison(df, OUTPUT_DIR / "professional_metrics_comparison.png")
    
    # Create summary table
    summary_df = create_summary_table(df, OUTPUT_DIR / "professional_results_summary")
    
    print(f"\n📋 Results Summary Table:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    print(f"\n✨ All professional outputs saved to: {OUTPUT_DIR}")
    print("Generated files:")
    print("  📈 professional_auroc_comparison.png")
    print("  📊 professional_metrics_comparison.png") 
    print("  📄 professional_results_summary.csv")
    print("  📄 professional_results_summary.md")
    print("="*80)
    
    logging.info("🎉 Professional visualization generation complete!")

if __name__ == "__main__":
    main() 