# generate_manuscript_figures.py
"""
A streamlined script to analyze experimental results and generate a focused set of
publication-quality figures and tables for a manuscript.
"""
import os
import pickle
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---

# Directories for input results and output figures.
RESULTS_DIR = 'notebooks/Phase 5'
BASELINE_RESULTS_DIR = 'notebooks/Phase 1 and 2/phase_1_outputs'
OUTPUT_DIR = 'manuscript_figures'

# Define the structure of your experiment for ordering and labeling.
REPRESENTATIONS = ['Baseline', 'F1', 'F2', 'F3']
PROMPTS = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
PROMPT_LABELS = {
    'P0': 'P0 (Control)', 'P1': 'P1 (Task-Specific)', 'P2': 'P2 (Persona-Driven)',
    'P3': 'P3 (Relational-Focus)', 'P4': 'P4 (Acute Dysregulation)',
    'P5': 'P5 (Dominant Pathophysiology)', 'XGBoost': 'XGBoost', 'ElasticNet': 'Elastic Net'
}
REP_LABELS = {
    'F1': 'F1 (Uninterpreted)', 'F2': 'F2 (Interpreted)',
    'F3': 'F3 (Narrative Summary)', 'Baseline': 'Baseline (Numeric)'
}


# --- Main Functions ---

def setup_logging_and_dirs() -> None:
    """Set up logging and create the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s', handlers=[logging.StreamHandler()])

def load_all_results(results_dir: str, baseline_dir: str) -> Optional[List[Dict[str, Any]]]:
    """Load all result pkl files, including baselines, into a list of dictionaries."""
    all_results = []
    logging.info(f"Scanning for result files in: {results_dir} and {baseline_dir}")

    # Load embedding model results
    if not os.path.isdir(results_dir):
        logging.error(f"Results directory not found: {results_dir}")
        return None
    
    for root, _, files in os.walk(results_dir):
        for filename in files:
            if filename.endswith('.pkl'):
                try:
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        data['model_name'] = os.path.basename(root)  # Inject model name from dir
                        all_results.append(data)
                except Exception as e:
                    logging.warning(f"Could not load file {filepath}: {e}")

    # Load baseline model results
    baseline_files = {
        'results_xgboost_baseline.pkl': 'Baseline_XGBoost',
        'results_elastic_net_baseline.pkl': 'Baseline_ElasticNet'
    }
    for filename, arm_name in baseline_files.items():
        filepath = os.path.join(baseline_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    data['experimental_arm'] = arm_name
                    if 'XGBoost' in arm_name:
                        data['model_name'] = 'XGBoost'
                    elif 'ElasticNet' in arm_name:
                        data['model_name'] = 'ElasticNet'
                    all_results.append(data)
            except Exception as e:
                logging.warning(f"Could not load baseline file {filepath}: {e}")
        else:
            logging.warning(f"Baseline result file not found: {filepath}")

    if not all_results:
        logging.error("No result files were found. Cannot proceed.")
        return None
        
    logging.info(f"Successfully loaded {len(all_results)} total result files.")
    return all_results

def create_summary_dataframe(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a clean pandas DataFrame from the loaded results."""
    records = []
    for res in all_results:
        arm = res.get('experimental_arm', 'Unknown')
        parts = arm.split('_', 1)
        rep, prompt = parts if len(parts) == 2 else (parts[0], "Unknown")
        eval_data = res.get('full_evaluation', res)
        
        auprc_ci_lower = eval_data.get('auprc', {}).get('ci_lower')
        auprc_ci_upper = eval_data.get('auprc', {}).get('ci_upper')

        records.append({
            'Representation': rep,
            'Prompt': prompt,
            'Model': res.get('model_name', 'Unknown'),  # Use injected model name
            'AUROC': eval_data.get('auroc', {}).get('point_estimate'),
            'AUROC_CI_Lower': eval_data.get('auroc', {}).get('ci_lower'),
            'AUROC_CI_Upper': eval_data.get('auroc', {}).get('ci_upper'),
            'AUPRC': eval_data.get('auprc', {}).get('point_estimate'),
            'AUPRC_CI_Lower': auprc_ci_lower,
            'AUPRC_CI_Upper': auprc_ci_upper,
        })
    
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).dropna(subset=['AUROC', 'AUPRC'])
    
    df['Representation'] = pd.Categorical(df['Representation'], categories=REPRESENTATIONS, ordered=True)
    return df.sort_values(by=['Representation', 'Prompt', 'Model'])

def generate_heatmap(df: pd.DataFrame, metric: str = 'AUROC', filename: str = 'figure_1_auroc_heatmap.png', title_prefix: str = 'Figure 1') -> None:
    """Generate and save a heatmap of AUROC or AUPRC values for embedding models only."""
    logging.info(f"Generating {metric} heatmap...")
    df_embedding_only = df[df['Representation'] != 'Baseline'].copy()
    
    if df_embedding_only.empty:
        logging.warning(f"No embedding model data to generate {metric} heatmap.")
        return

    df_embedding_only['Prompt'] = pd.Categorical(df_embedding_only['Prompt'], categories=PROMPTS, ordered=True)
    heatmap_data = df_embedding_only.pivot(index='Representation', columns='Prompt', values=metric)
    heatmap_data.index = heatmap_data.index.map(REP_LABELS)
    heatmap_data.columns = heatmap_data.columns.map(PROMPT_LABELS)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="white")
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", linewidths=.5, cbar_kws={'label': f'{metric} Score'})
    ax.set_title(f'{title_prefix}: Test Set {metric} by Representation and Prompting Strategy', fontsize=16, pad=20)
    ax.set_xlabel('Prompting Strategy', fontsize=12)
    ax.set_ylabel('Data Representation', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    logging.info(f"Heatmap saved to: {save_path}")
    plt.close()

def generate_interaction_plot(df: pd.DataFrame, metric: str = 'AUROC', filename: str = 'figure_2_interaction_plot.png', title_prefix: str = 'Figure 2') -> None:
    """Generate and save an interaction plot showing embedding models against baseline performance."""
    logging.info(f"Generating {metric} interaction plot...")
    
    df_embedding_only = df[df['Representation'] != 'Baseline'].copy()
    df_baselines = df[df['Representation'] == 'Baseline']

    embedding_reps = [rep for rep in REPRESENTATIONS if rep != 'Baseline']
    df_embedding_only['Representation'] = pd.Categorical(df_embedding_only['Representation'], categories=embedding_reps, ordered=True)

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    df_embedding_only_plot = df_embedding_only.copy()
    df_embedding_only_plot['Representation'] = df_embedding_only_plot['Representation'].map(REP_LABELS)
    df_embedding_only_plot['Prompt'] = pd.Categorical(df_embedding_only_plot['Prompt'], categories=PROMPTS, ordered=True).map(PROMPT_LABELS)
    
    ax = sns.lineplot(data=df_embedding_only_plot, x='Prompt', y=metric, hue='Representation', style='Representation', markers=True, dashes=False, markersize=8)
    
    if not df_baselines.empty:
        baseline_colors = {'XGBoost': 'crimson', 'ElasticNet': 'darkgreen'}
        for _, baseline_row in df_baselines.iterrows():
            model_name_key = baseline_row['Prompt']
            model_name_label = PROMPT_LABELS.get(model_name_key, model_name_key)
            score = baseline_row[metric]
            color = baseline_colors.get(model_name_key, 'gray')
            ax.axhline(y=score, color=color, linestyle='--', label=f'{model_name_label} Baseline ({score:.4f})')

    ax.set_title(f'{title_prefix}: Interaction between Representation and Prompting Strategy for {metric}', fontsize=16, pad=20)
    ax.set_xlabel('Prompting Strategy', fontsize=12)
    ax.set_ylabel(f'Test Set {metric}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model Type')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    logging.info(f"Interaction plot saved to: {save_path}")
    plt.close()

def generate_performance_lift_plot(df: pd.DataFrame, metric: str = 'AUROC', filename: str = 'figure_3_performance_lift.png', title_prefix: str = 'Figure 3') -> None:
    """Generate a plot showing performance lift over the best baseline model."""
    logging.info(f"Generating {metric} performance lift plot...")
    
    baseline_scores = df[df['Representation'] == 'Baseline'][metric]
    if baseline_scores.empty:
        logging.warning(f"No baseline scores for {metric}, cannot generate lift plot.")
        return
    best_baseline_score = baseline_scores.max()
    
    df_copy = df.copy()
    lift_metric_name = f'{metric}_Lift'
    df_copy[lift_metric_name] = df_copy[metric] - best_baseline_score
    df_copy['Arm'] = df_copy.apply(lambda row: f"{REP_LABELS.get(row['Representation'], row['Representation'])} - {PROMPT_LABELS.get(row['Prompt'], row['Prompt'])}", axis=1)
    df_copy = df_copy.sort_values(lift_metric_name, ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df_copy, x=lift_metric_name, y='Arm', palette='coolwarm', hue='Arm', legend=False)
    ax.set_title(f'{title_prefix}: {metric} Improvement Over Best Baseline Model', fontsize=16, pad=20)
    ax.set_xlabel(f'Change in {metric} vs. Best Baseline', fontsize=12)
    ax.set_ylabel('Experimental Arm', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    logging.info(f"Performance lift plot saved to: {save_path}")
    plt.close()

def generate_model_comparison_plot(df: pd.DataFrame, metric: str = 'AUROC', filename: str = 'figure_7_model_comparison_auroc.png', title_prefix: str = 'Figure 7') -> None:
    """Generate a plot comparing the performance of different embedding models."""
    logging.info(f"Generating {metric} model comparison plot...")
    
    df_embedding_only = df[df['Representation'] != 'Baseline'].copy()
    
    if df_embedding_only.empty:
        logging.warning("No embedding model data to generate model comparison plot.")
        return

    df_embedding_only['Model'] = df_embedding_only['Model'].str.replace('embedding_model_results_', '', regex=False)

    g = sns.catplot(
        data=df_embedding_only,
        x='Representation',
        y=metric,
        hue='Model',
        col='Prompt',
        kind='bar',
        height=6,
        aspect=0.8,
        palette='viridis',
        legend_out=True,
        col_order=PROMPTS
    )

    g.fig.suptitle(f'{title_prefix}: {metric} Comparison of Embedding Models Across Representations and Prompts', y=1.03, fontsize=16)
    g.set_axis_labels("Representation", f"Test Set {metric}")
    g.set_titles("Prompt: {col_name}")
    g.despine(left=True)

    for ax in g.axes.flat:
        labels = [REP_LABELS.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45, ha='right')

    g.add_legend(title='Embedding Model')
    g.fig.tight_layout(rect=(0, 0, 1, 0.97))

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    logging.info(f"Model comparison plot saved to: {save_path}")
    plt.close('all')

def generate_results_table(df: pd.DataFrame) -> None:
    """Generate and save a formatted Markdown table of all results."""
    logging.info("Generating Markdown results table...")
    
    df_copy = df.copy()
    df_copy['AUROC_Formatted'] = df_copy.apply(
        lambda row: f"{row['AUROC']:.4f} ({row['AUROC_CI_Lower']:.4f} - {row['AUROC_CI_Upper']:.4f})"
        if pd.notnull(row['AUROC_CI_Lower']) else f"{row['AUROC']:.4f}", axis=1
    )
    df_copy['AUPRC_Formatted'] = df_copy.apply(
        lambda row: f"{row['AUPRC']:.4f} ({row['AUPRC_CI_Lower']:.4f} - {row['AUPRC_CI_Upper']:.4f})"
        if pd.notnull(row['AUPRC_CI_Lower']) else f"{row['AUPRC']:.4f}", axis=1
    )
    
    df_copy['Representation'] = df_copy['Representation'].map(REP_LABELS)
    df_copy['Prompt'] = df_copy['Prompt'].map(PROMPT_LABELS)
    df_copy['Model'] = df_copy['Model'].str.replace('embedding_model_results_', '', regex=False)

    
    table_df = df_copy[['Representation', 'Prompt', 'Model', 'AUROC_Formatted', 'AUPRC_Formatted']]
    table_df.columns = ['Representation', 'Prompt', 'Model', 'AUROC (95% CI)', 'AUPRC (95% CI)']
    
    markdown_table = table_df.to_markdown(index=False)
    full_table_text = "## Table 1: Full Experimental Results\n\n" + markdown_table
    
    save_path = os.path.join(OUTPUT_DIR, 'table_1_results.md')
    with open(save_path, 'w') as f:
        f.write(full_table_text)
    logging.info(f"Markdown table saved to: {save_path}")


def main():
    """Main function to run the analysis and generate outputs."""
    setup_logging_and_dirs()
    
    results = load_all_results(RESULTS_DIR, BASELINE_RESULTS_DIR)
    if not results:
        return
        
    full_df = create_summary_dataframe(results)
    
    if full_df.empty:
        logging.error("Full dataframe is empty, cannot generate figures.")
        return

    # Create a summary df with only the best performing model for each (Rep, Prompt) pair
    embedding_df = full_df[full_df['Representation'] != 'Baseline']
    best_embedding_df = pd.DataFrame()
    if not embedding_df.empty:
        best_embedding_indices = embedding_df.groupby(['Representation', 'Prompt'], observed=True)['AUROC'].idxmax()
        best_embedding_df = embedding_df.loc[best_embedding_indices]
    
    baseline_df = full_df[full_df['Representation'] == 'Baseline']
    summary_df = pd.concat([best_embedding_df, baseline_df], ignore_index=True)


    champion = summary_df.loc[summary_df['AUROC'].idxmax()]
    logging.info("\n" + "="*50)
    logging.info("🏆 Champion Semantic System Identified (based on best model per configuration) 🏆")
    logging.info(f"   Best Representation: {REP_LABELS.get(champion['Representation'], champion['Representation'])}")
    logging.info(f"   Best Prompt: {PROMPT_LABELS.get(champion['Prompt'], champion['Prompt'])}")
    logging.info(f"   Best Test AUROC: {champion['AUROC']:.4f} (Model: {champion['Model']})")
    logging.info("="*50 + "\n")
    
    # Generate all outputs
    generate_heatmap(summary_df, metric='AUROC', filename='figure_1_auroc_heatmap.png', title_prefix='Figure 1')
    generate_heatmap(summary_df, metric='AUPRC', filename='figure_2_auprc_heatmap.png', title_prefix='Figure 2')
    
    generate_interaction_plot(summary_df, metric='AUROC', filename='figure_3_auroc_interaction_plot.png', title_prefix='Figure 3')
    generate_interaction_plot(summary_df, metric='AUPRC', filename='figure_4_auprc_interaction_plot.png', title_prefix='Figure 4')
    
    generate_performance_lift_plot(summary_df, metric='AUROC', filename='figure_5_auroc_performance_lift.png', title_prefix='Figure 5')
    generate_performance_lift_plot(summary_df, metric='AUPRC', filename='figure_6_auprc_performance_lift.png', title_prefix='Figure 6')
    
    generate_model_comparison_plot(full_df, metric='AUROC', filename='figure_7_model_comparison_auroc.png', title_prefix='Figure 7')
    generate_model_comparison_plot(full_df, metric='AUPRC', filename='figure_8_model_comparison_auprc.png', title_prefix='Figure 8')

    generate_results_table(full_df)
    
    logging.info("All manuscript figures and tables have been generated successfully.")

if __name__ == "__main__":
    main()
