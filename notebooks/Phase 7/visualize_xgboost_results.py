
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
                
                results.append({
                    'Task': task,
                    'Arm': arm,
                    'AUROC': auroc_data.get('point_estimate'),
                    'AUROC_CI_Lower': auroc_data.get('ci_lower'),
                    'AUROC_CI_Upper': auroc_data.get('ci_upper'),
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

def create_summary_table(df, save_path):
    """Create and save a professional summary table.
    Now outputs a standalone LaTeX .tex table (AUROC only) compilable in TeXworks.
    """
    df_table = df.copy()
    df_table['TaskLabel'] = df_table['Task'].map(TASK_LABELS)
    # Fallback to raw task if mapping missing
    df_table['TaskLabel'] = df_table['TaskLabel'].fillna(df_table['Task'])
    
    # Format metrics with confidence intervals (AUROC only)
    df_table['AUROC (95% CI)'] = df_table.apply(
        lambda row: f"{row['AUROC']:.4f} ({row['AUROC_CI_Lower']:.4f}-{row['AUROC_CI_Upper']:.4f})",
        axis=1
    )
    
    # Select and order columns
    summary_cols = ['TaskLabel', 'Arm', 'AUROC (95% CI)']
    df_summary = df_table[summary_cols].rename(columns={
        'TaskLabel': 'Clinical Prediction Task',
        'Arm': 'Model Configuration'
    })
    
    # Sort by AUROC numeric for readability
    df_summary['AUROC_numeric'] = df_table['AUROC']
    df_summary = df_summary.sort_values('AUROC_numeric', ascending=False).drop('AUROC_numeric', axis=1)
    
    # Generate standalone LaTeX content similar to provided example
    header_lines = [
        "\\documentclass[border=0pt]{standalone}",
        "\\usepackage{booktabs}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{adjustbox}",
        "\\begin{document}",
        "\\begin{adjustbox}{width=\\textwidth}",
        "\\begin{tabular}{@{}llr@{}}",
        "\\toprule",
        "Clinical Prediction Task & Model configuration & AUROC (95\\% CI) \\",
        "\\midrule",
    ]
    rows = []
    for _, r in df_summary.iterrows():
        task = str(r['Clinical Prediction Task']).replace('_', '\\_')
        cfg = str(r['Model Configuration']).replace('_', '\\_')
        rows.append(f"{task} & {cfg} & {r['AUROC (95% CI)']} \\")
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{adjustbox}\n"
        "\\end{document}\n"
    )
    latex_str = "\n".join(header_lines) + "\n" + "\n".join(rows) + "\n" + footer
    
    # Write .tex file
    tex_path = save_path.with_suffix('.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    logging.info(f"Standalone LaTeX table saved to {tex_path}")
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
    summary_data = df[['Task', 'Arm', 'AUROC']].copy()
    summary_data['Task'] = summary_data['Task'].map(TASK_LABELS)
    print(summary_data.to_string(index=False, float_format='%.4f'))
    
    print(f"\n🎨 Generating professional visualizations...")
    print("-" * 60)
    
    # Create visualizations
    create_performance_comparison(df, OUTPUT_DIR / "professional_auroc_comparison.png")
    # (Removed AUPRC comparison plot)
    
    # Create summary table
    summary_df = create_summary_table(df, OUTPUT_DIR / "professional_results_summary")
    
    print(f"\n📋 Results Summary Table:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    print(f"\n✨ All professional outputs saved to: {OUTPUT_DIR}")
    print("Generated files:")
    print("  📈 professional_auroc_comparison.png")
    print("  📄 professional_results_summary.tex")
    print("="*80)
    
    logging.info("🎉 Professional visualization generation complete!")

if __name__ == "__main__":
    main() 