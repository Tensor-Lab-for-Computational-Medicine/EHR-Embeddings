# generate_manuscript_figures.py
"""
A streamlined script to analyze experimental results and generate a focused set of
publication-quality figures and tables for a manuscript.
"""
import os
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---

# Directories for input results and output figures.
RESULTS_DIR = 'notebooks/Phase 5/embedding_model_results'
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

def setup_logging_and_dirs():
    """Set up logging and create the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s', handlers=[logging.StreamHandler()])

def load_all_results(results_dir, baseline_dir):
    """Load all result.pkl files, including baselines, into a list of dictionaries."""
    all_results = []
    logging.info(f"Scanning for result files in: {results_dir} and {baseline_dir}")
    
    # Load embedding model results
    if os.path.isdir(results_dir):
        for filename in os.listdir(results_dir):
            if filename.startswith('results_') and filename.endswith('.pkl'):
                try:
                    with open(os.path.join(results_dir, filename), 'rb') as f:
                        all_results.append(pickle.load(f))
                except Exception as e:
                    logging.warning(f"Could not load file {filename}: {e}")

    # Load baseline model results
    baseline_files = {
        'Baseline_XGBoost': os.path.join(baseline_dir, 'results_xgboost_baseline.pkl'),
        'Baseline_ElasticNet': os.path.join(baseline_dir, 'results_elastic_net_baseline.pkl')
    }
    for arm_name, filepath in baseline_files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    data['experimental_arm'] = arm_name
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

def create_summary_dataframe(all_results):
    """Create a clean pandas DataFrame from the loaded results."""
    records = []
    for res in all_results:
        arm = res.get('experimental_arm', 'Unknown')
        rep, prompt = arm.split('_', 1)
        eval_data = res.get('full_evaluation', res)
        records.append({
            'Representation': rep, 'Prompt': prompt,
            'AUROC': eval_data['auroc']['point_estimate'],
            'AUROC_CI_Lower': eval_data['auroc']['ci_lower'],
            'AUROC_CI_Upper': eval_data['auroc']['ci_upper'],
            'AUPRC': eval_data['auprc']['point_estimate']
        })
    df = pd.DataFrame(records)
    df['Representation'] = pd.Categorical(df['Representation'], categories=REPRESENTATIONS, ordered=True)
    return df.sort_values(by=['Representation', 'Prompt'])

def generate_heatmap(df, metric='AUROC', filename='figure_1_auroc_heatmap.png', title_prefix='Figure 1'):
    """Generate and save a heatmap of AUROC or AUPRC values for embedding models only."""
    logging.info(f"Generating {metric} heatmap...")
    df_embedding_only = df[df['Representation'] != 'Baseline'].copy()
    
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

def generate_interaction_plot(df):
    """Generate and save an interaction plot showing embedding models against baseline performance."""
    logging.info("Generating interaction plot...")
    
    # Separate embedding models from baselines
    df_embedding_only = df[df['Representation'] != 'Baseline'].copy()
    df_baselines = df[df['Representation'] == 'Baseline']

    # FIX: Redefine the categorical type on the filtered dataframe to ensure 'Baseline' is not included in the legend.
    embedding_reps = ['F1', 'F2', 'F3']
    df_embedding_only['Representation'] = pd.Categorical(df_embedding_only['Representation'], categories=embedding_reps, ordered=True)

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Map labels for plotting
    df_embedding_only['Representation'] = df_embedding_only['Representation'].map(REP_LABELS)
    df_embedding_only['Prompt'] = pd.Categorical(df_embedding_only['Prompt'], categories=PROMPTS, ordered=True).map(PROMPT_LABELS)
    
    # Create the main line plot for embedding models
    ax = sns.lineplot(data=df_embedding_only, x='Prompt', y='AUROC', hue='Representation', style='Representation', markers=True, dashes=False, markersize=8)
    
    # Add horizontal dashed lines for each baseline model
    if not df_baselines.empty:
        baseline_colors = {'XGBoost': 'crimson', 'ElasticNet': 'darkgreen'}
        for _, baseline_row in df_baselines.iterrows():
            model_name = PROMPT_LABELS.get(baseline_row['Prompt'])
            score = baseline_row['AUROC']
            color = baseline_colors.get(model_name, 'gray')
            ax.axhline(y=score, color=color, linestyle='--', label=f'{model_name} Baseline ({score:.4f})')

    ax.set_title('Figure 2: Interaction between Representation and Prompting Strategy', fontsize=16, pad=20)
    ax.set_xlabel('Prompting Strategy', fontsize=12)
    ax.set_ylabel('Test Set AUROC', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    # Update legend to include the baseline lines
    plt.legend(title='Model Type')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'figure_2_interaction_plot.png')
    plt.savefig(save_path, dpi=300)
    logging.info(f"Interaction plot saved to: {save_path}")
    plt.close()

def generate_performance_lift_plot(df):
    """Generate a plot showing performance lift over the best baseline model."""
    logging.info("Generating performance lift plot...")
    
    baseline_scores = df[df['Representation'] == 'Baseline']['AUROC']
    if baseline_scores.empty:
        logging.warning("No baseline scores found, cannot generate lift plot.")
        return
    best_baseline_auroc = baseline_scores.max()
    
    df_copy = df.copy()
    df_copy['AUROC_Lift'] = df_copy['AUROC'] - best_baseline_auroc
    df_copy['Arm'] = df_copy.apply(lambda row: f"{REP_LABELS.get(row['Representation'])} - {PROMPT_LABELS.get(row['Prompt'])}", axis=1)
    df_copy = df_copy.sort_values('AUROC_Lift', ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df_copy, x='AUROC_Lift', y='Arm', palette='coolwarm', hue='Arm', legend=False)
    ax.set_title('Figure 3: AUROC Improvement Over Best Baseline Model', fontsize=16, pad=20)
    ax.set_xlabel('Change in AUROC vs. Best Baseline', fontsize=12)
    ax.set_ylabel('Experimental Arm', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'figure_3_performance_lift.png')
    plt.savefig(save_path, dpi=300)
    logging.info(f"Performance lift plot saved to: {save_path}")
    plt.close()

def generate_results_table(df):
    """Generate and save a formatted Markdown table of all results."""
    logging.info("Generating Markdown results table...")
    
    df_copy = df.copy()
    df_copy['AUROC_Formatted'] = df_copy.apply(lambda row: f"{row['AUROC']:.4f} ({row['AUROC_CI_Lower']:.4f} - {row['AUROC_CI_Upper']:.4f})", axis=1)
    df_copy['AUPRC_Formatted'] = df_copy.apply(lambda row: f"{row['AUPRC']:.4f} ({row['AUROC_CI_Lower']:.4f} - {row['AUROC_CI_Upper']:.4f})", axis=1) # Corrected to use AUPRC CIs
    
    df_copy['Representation'] = df_copy['Representation'].map(REP_LABELS)
    df_copy['Prompt'] = df_copy['Prompt'].map(PROMPT_LABELS)
    
    table_df = df_copy[['Representation', 'Prompt', 'AUROC_Formatted', 'AUPRC_Formatted']]
    table_df.columns = ['Representation', 'Prompt', 'AUROC (95% CI)', 'AUPRC (95% CI)']
    
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
        
    summary_df = create_summary_dataframe(results)
    
    champion = summary_df.loc[summary_df['AUROC'].idxmax()]
    logging.info("\n" + "="*50)
    logging.info("🏆 Champion Semantic System Identified 🏆")
    logging.info(f"   Best Representation: {REP_LABELS.get(champion['Representation'], champion['Representation'])}")
    logging.info(f"   Best Prompt: {PROMPT_LABELS.get(champion['Prompt'], champion['Prompt'])}")
    logging.info(f"   Best Test AUROC: {champion['AUROC']:.4f}")
    logging.info("="*50 + "\n")
    
    # Generate all outputs
    generate_heatmap(summary_df, metric='AUROC', filename='figure_1_auroc_heatmap.png', title_prefix='Figure 1')
    generate_interaction_plot(summary_df)
    generate_performance_lift_plot(summary_df)
    generate_heatmap(summary_df, metric='AUPRC', filename='figure_4_auprc_heatmap.png', title_prefix='Figure 4')
    generate_results_table(summary_df)
    
    logging.info("All manuscript figures and tables have been generated successfully.")

if __name__ == "__main__":
    main()
