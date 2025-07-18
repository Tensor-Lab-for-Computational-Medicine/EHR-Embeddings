# generate_manuscript_figures_corrected.py
"""
A streamlined script to analyze experimental results and generate a focused set of
publication-quality figures and tables for a manuscript.

Refactored for improved readability, modularity, and maintainability.
This version incorporates user feedback for plotting corrections, including confidence intervals on Figure 2.
This version also adds a new figure (Figure 6) to compare different models within a task.
"""
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# --- Configuration ---

# Directories for input results and output figures.
# Corrected BASE_DIR as per user request.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / 'notebooks/Phase 5/embedding_model_results'
BASELINE_RESULTS_DIR = BASE_DIR / 'notebooks/Phase 1 and 2/phase_1_outputs'
OUTPUT_DIR = BASE_DIR / 'manuscript_figures'

# Define the structure of your experiment for ordering and labeling.
REPRESENTATIONS = ['Baseline', 'F1', 'F2', 'F3']
PROMPTS = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'XGBoost', 'ElasticNet']
PROMPT_LABELS = {
    'P0': 'P0 (Control)', 'P1': 'P1 (Task-Specific)', 'P2': 'P2 (Persona-Driven)',
    'P3': 'P3 (Relational-Focus)', 'P4': 'P4 (Acute Dysregulation)',
    'P5': 'P5 (Dominant Pathophysiology)', 'XGBoost': 'XGBoost', 'ElasticNet': 'Elastic Net'
}
REP_LABELS = {
    'F1': 'F1 (Uninterpreted)', 'F2': 'F2 (Interpreted)',
    'F3': 'F3 (Narrative Summary)', 'Baseline': 'Baseline (Numeric)'
}
TASK_LABELS = {
    'mort_hosp': 'Hospital Mortality',
    'los_3': 'Length-of-Stay > 3 Days',
    'los_7': 'Length-of-Stay > 7 Days',
    'readmission_30': '30-Day Readmission',
    'intervention_vent': 'Mechanical Ventilation',
    'intervention_vaso': 'Vasopressor Administration'
}


# --- Utility & Setup Functions ---

def setup_environment() -> None:
    """Set up logging configuration and create the output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def _load_pickle_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Safely load a single pickle file."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning(f"Could not load or read file {filepath}: {e}")
        return None

def save_figure(fig: plt.Figure, filename: str, sub_dir: Optional[str] = None) -> None:
    """Save a matplotlib figure to the output directory."""
    save_dir = OUTPUT_DIR / sub_dir if sub_dir else OUTPUT_DIR
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Figure saved to: {save_path}")
    plt.close(fig)


# --- Data Loading and Processing Functions ---

def _load_embedding_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load results from the embedding model directory structure."""
    results = []
    if not results_dir.is_dir():
        logging.error(f"Embedding results directory not found: {results_dir}")
        return []

    for filepath in results_dir.glob('**/results_*.pkl'):
        data = _load_pickle_file(filepath)
        if data:
            data['task'] = filepath.parent.name
            data['model_name'] = filepath.parent.parent.name
            results.append(data)
    return results

def _load_baseline_results(baseline_dir: Path) -> List[Dict[str, Any]]:
    """Load results from the baseline model directory structure."""
    results = []
    baseline_files = {
        'results_xgboost_baseline.pkl': 'Baseline_XGBoost',
        'results_elastic_net_baseline.pkl': 'Baseline_ElasticNet'
    }
    if not baseline_dir.is_dir():
        logging.error(f"Baseline results directory not found: {baseline_dir}")
        return []

    for task_dir in baseline_dir.iterdir():
        if task_dir.is_dir():
            for filename, arm_name in baseline_files.items():
                filepath = task_dir / filename
                if filepath.exists():
                    data = _load_pickle_file(filepath)
                    if data:
                        model_key = 'XGBoost' if 'XGBoost' in arm_name else 'ElasticNet'
                        record = data.copy()
                        record.update({
                            'experimental_arm': arm_name,
                            'model_name': model_key,
                            'task': task_dir.name
                        })
                        results.append(record)
    return results

def load_all_results(results_dir: Path, baseline_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """Load all result pkl files, including baselines, into a list of dictionaries."""
    logging.info(f"Scanning for result files in: {results_dir} and {baseline_dir}")
    all_results = _load_embedding_results(results_dir) + _load_baseline_results(baseline_dir)

    if not all_results:
        logging.error("No result files were found. Cannot proceed.")
        return None

    logging.info(f"Successfully loaded {len(all_results)} total result files.")
    return all_results

def create_summary_dataframe(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a clean, structured pandas DataFrame from the loaded results."""
    records = []
    for res in all_results:
        arm = res.get('experimental_arm', 'Unknown')
        parts = arm.split('_', 1)
        rep, prompt = parts if len(parts) == 2 else (parts[0], "Unknown")
        
        eval_data = res.get('full_evaluation', res)
        auroc_data = eval_data.get('auroc', {})
        auprc_data = eval_data.get('auprc', {})

        records.append({
            'Task': res.get('task', 'Unknown'),
            'Representation': rep,
            'Prompt': prompt,
            'Model': res.get('model_name', 'Unknown'),
            'AUROC': auroc_data.get('point_estimate'),
            'AUROC_CI_Lower': auroc_data.get('ci_lower'),
            'AUROC_CI_Upper': auroc_data.get('ci_upper'),
            'AUPRC': auprc_data.get('point_estimate'),
            'AUPRC_CI_Lower': auprc_data.get('ci_lower'),
            'AUPRC_CI_Upper': auprc_data.get('ci_upper'),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).dropna(subset=['AUROC', 'AUPRC'])
    
    # Clean up model names for better plot labels
    df['Model'] = df['Model'].str.replace('embedding_model_results_', '', regex=False)
    
    df['Task'] = pd.Categorical(df['Task'], categories=TASK_LABELS.keys(), ordered=True)
    df['Representation'] = pd.Categorical(df['Representation'], categories=REPRESENTATIONS, ordered=True)
    df['Prompt'] = pd.Categorical(df['Prompt'], categories=PROMPTS, ordered=True)
    
    return df.sort_values(by=['Task', 'Representation', 'Prompt', 'Model'])


# --- Plotting Functions ---

def generate_representation_barplot(df: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    """
    Generate a bar plot comparing representations for each prompt, including baselines.
    This version now plots confidence intervals for each bar.
    """
    logging.info(f"Generating {metric} representation bar plot (with CI): {title}")

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    df_plot = df.copy()
    df_plot['RepresentationLabel'] = df_plot['Representation'].map(REP_LABELS)
    df_plot['PromptLabel'] = df_plot['Prompt'].map(PROMPT_LABELS)
    
    # Ensure labels are categorical for consistent ordering
    df_plot['RepresentationLabel'] = pd.Categorical(df_plot['RepresentationLabel'], categories=[REP_LABELS[r] for r in REPRESENTATIONS], ordered=True)
    df_plot['PromptLabel'] = pd.Categorical(df_plot['PromptLabel'], categories=[PROMPT_LABELS[p] for p in PROMPTS if p in PROMPT_LABELS], ordered=True)
    df_plot = df_plot.dropna(subset=['PromptLabel', 'RepresentationLabel'])


    # Define a color palette
    rep_colors = sns.color_palette('viridis', n_colors=3)
    palette = {
        REP_LABELS['F1']: rep_colors[0],
        REP_LABELS['F2']: rep_colors[1],
        REP_LABELS['F3']: rep_colors[2],
        REP_LABELS['Baseline']: 'crimson'
    }

    bplot = sns.barplot(
        data=df_plot, x='PromptLabel', y=metric, hue='RepresentationLabel',
        palette=palette, ax=ax
    )

    # Add confidence intervals
    hue_order = [h.get_text() for h in ax.get_legend().get_texts()]
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    # Calculate bar and group properties
    num_hues = len(hue_order)
    bar_width = bplot.patches[0].get_width() if bplot.patches else 0.8
    gap = 0.1 # A guess for the gap seaborn leaves between bars
    group_width = num_hues * bar_width + (num_hues-1)*gap
    
    # Create a map for x-tick positions
    x_pos_map = {label: i for i, label in enumerate(x_labels)}

    # Iterate through the plotted data to add error bars at correct positions
    for _, row in df_plot.iterrows():
        prompt_label = row['PromptLabel']
        rep_label = row['RepresentationLabel']
        
        if prompt_label in x_pos_map and rep_label in hue_order:
            x_pos_group = x_pos_map[prompt_label]
            hue_index = hue_order.index(rep_label)
            
            # Calculate center of the specific bar
            center_of_group = x_pos_group
            bar_offset = (hue_index - (num_hues - 1) / 2) * bar_width
            x_coord = center_of_group + bar_offset
            
            y_val = row[metric]
            ci_lower = row[f'{metric}_CI_Lower']
            ci_upper = row[f'{metric}_CI_Upper']
            
            if pd.notna(ci_lower) and pd.notna(ci_upper):
                y_err = [[y_val - ci_lower], [ci_upper - y_val]]
                ax.errorbar(x=x_coord, y=y_val, yerr=y_err, fmt='none', c='black', capsize=3, zorder=10)


    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Prompting Strategy / Baseline Model', fontsize=12)
    ax.set_ylabel(f'Test Set {metric} (with 95% CI)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Representation Type')
    
    if df_plot[metric].min() > 0.4:
        ax.set_ylim(0.5, max(1.0, df_plot[f'{metric}_CI_Upper'].max() * 1.05))

    save_figure(fig, filename)
    
def generate_performance_lift_plot(df: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    """
    Generate a plot showing performance lift over the best baseline model for a given task.
    Now includes baseline models in the plot for comprehensive comparison.
    """
    logging.info(f"Generating {metric} performance lift plot: {title}")

    baseline_scores = df[df['Representation'] == 'Baseline'][metric]
    if baseline_scores.empty:
        logging.warning(f"No baseline scores for {metric}, cannot generate lift plot.")
        return
    best_baseline_score = baseline_scores.max()

    df_lift = df.copy()
    lift_metric = f'{metric}_Lift'
    df_lift[lift_metric] = df_lift[metric] - best_baseline_score
    df_lift['Arm'] = df_lift.apply(lambda r: f"{REP_LABELS.get(r['Representation'], r['Representation'])} - {PROMPT_LABELS.get(r['Prompt'], r['Prompt'])}", axis=1)
    df_lift = df_lift.sort_values(lift_metric, ascending=False)
    
    lower_err = df_lift[lift_metric] - (df_lift[f'{metric}_CI_Lower'] - best_baseline_score)
    upper_err = (df_lift[f'{metric}_CI_Upper'] - best_baseline_score) - df_lift[lift_metric]
    errors = [lower_err, upper_err] if lower_err.notna().all() and upper_err.notna().all() else None

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_lift, x=lift_metric, y='Arm', palette='coolwarm', ax=ax, hue='Arm', dodge=False)

    if errors:
        ax.errorbar(x=df_lift[lift_metric], y=np.arange(len(df_lift)), xerr=errors,
                    fmt='none', ecolor='black', capsize=3)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(f'Change in {metric} vs. Best Baseline (with 95% CI)', fontsize=12)
    ax.set_ylabel('Experimental Arm', fontsize=12)
    if ax.get_legend(): ax.get_legend().remove()
    
    save_figure(fig, filename)

def generate_model_comparison_plot(df: pd.DataFrame, metric: str, filename: str, title: str):
    """
    Generate a faceted bar plot comparing underlying embedding model performance for a single task,
    broken down by representation and prompt.
    """
    logging.info(f"Generating detailed model comparison plot for {metric}: {title}")
    
    plot_df = df.copy()
    if plot_df.empty:
        logging.warning("No embedding model data available for the detailed comparison plot.")
        return

    plot_df['RepresentationLabel'] = plot_df['Representation'].map(REP_LABELS)
    plot_df['PromptLabel'] = plot_df['Prompt'].map(PROMPT_LABELS)

    # Ensure categorical ordering for consistent plots
    rep_cats = [REP_LABELS[r] for r in REPRESENTATIONS if r in plot_df['Representation'].unique()]
    prompt_cats = [PROMPT_LABELS[p] for p in PROMPTS if p in plot_df['Prompt'].unique()]
    model_names = sorted(plot_df['Model'].unique())
    
    plot_df['RepresentationLabel'] = pd.Categorical(plot_df['RepresentationLabel'], categories=rep_cats, ordered=True)
    plot_df['PromptLabel'] = pd.Categorical(plot_df['PromptLabel'], categories=prompt_cats, ordered=True)
    plot_df = plot_df.dropna(subset=['PromptLabel', 'RepresentationLabel'])

    # Use a specific palette for the models
    palette = sns.color_palette('cubehelix', n_colors=len(model_names))
    color_map = dict(zip(model_names, palette))

    g = sns.catplot(
        data=plot_df, x='PromptLabel', y=metric, hue='Model', col='RepresentationLabel',
        kind='bar', palette=color_map, height=6, aspect=1.2, legend=False,
        col_wrap=3
    )

    # Manually add confidence intervals to each facet
    for ax in g.axes.flatten():
        if not ax.patches: continue

        current_rep_label = ax.get_title().split(' = ')[-1]
        ax_df = plot_df[plot_df['RepresentationLabel'] == current_rep_label]

        hue_order = sorted(ax_df['Model'].unique())
        num_hues = len(hue_order)
        bar_width = ax.patches[0].get_width()
        
        x_labels = [label.get_text() for label in ax.get_xticklabels()]
        x_pos_map = {label: i for i, label in enumerate(x_labels)}

        for _, row in ax_df.iterrows():
            prompt_label, model_name = row['PromptLabel'], row['Model']
            if prompt_label in x_pos_map and model_name in hue_order:
                x_pos_group = x_pos_map[prompt_label]
                hue_index = hue_order.index(model_name)
                
                bar_offset = (hue_index - (num_hues - 1) / 2) * bar_width
                x_coord = x_pos_group + bar_offset
                
                y_val = row[metric]
                y_err = [[y_val - row[f'{metric}_CI_Lower']], [row[f'{metric}_CI_Upper'] - y_val]]
                
                ax.errorbar(x=x_coord, y=y_val, yerr=y_err, fmt='none', c='black', capsize=2, zorder=10)
        
        if ax_df[metric].min() > 0.4:
            ax.set_ylim(0.5, max(1.0, ax_df[f'{metric}_CI_Upper'].max() * 1.05))

    # Final plot adjustments
    g.fig.suptitle(title, fontsize=18, y=1.03)
    g.set_axis_labels("Prompting Strategy", f'Test Set {metric} (with 95% CI)')
    g.set_titles("Representation: {col_name}")
    g.set_xticklabels(rotation=45, ha='right')

    # Add a single, figure-level legend
    legend_handles = [mpatches.Patch(color=color_map[name], label=name) for name in model_names]
    g.fig.legend(handles=legend_handles, title="Embedding Model", bbox_to_anchor=(1.0, 0.5), loc="center left")
    g.fig.tight_layout(rect=[0, 0, 0.92, 1]) # Adjust for suptitle and legend

    save_figure(g.fig, filename)


def generate_task_comparison_plot(df: pd.DataFrame, metric: str, filename: str, title: str):
    """Generate a bar plot comparing representation performance across all prediction tasks."""
    logging.info(f"Generating {metric} comparison plot across tasks: {title}")
    
    best_indices = df.groupby(['Task', 'Representation'], observed=True)[metric].idxmax()
    plot_df = df.loc[best_indices].copy()
    
    plot_df['TaskLabel'] = plot_df['Task'].map(TASK_LABELS)
    plot_df['RepresentationLabel'] = plot_df['Representation'].map(REP_LABELS)

    plot_df['TaskLabel'] = pd.Categorical(plot_df['TaskLabel'], categories=TASK_LABELS.values(), ordered=True)
    plot_df['RepresentationLabel'] = pd.Categorical(plot_df['RepresentationLabel'], categories=REP_LABELS.values(), ordered=True)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.set_theme(style="whitegrid")
    
    bplot = sns.barplot(data=plot_df, x='TaskLabel', y=metric, hue='RepresentationLabel', palette='viridis', ax=ax)
    
    # Get hue order from the legend to match colors and positions
    hue_order = [x.get_text() for x in ax.get_legend().get_texts()]
    num_hues = len(hue_order)
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_pos_map = {label: i for i, label in enumerate(x_tick_labels)}
    bar_width = bplot.patches[0].get_width() if bplot.patches else 0.8
    
    # Iterate over plotted data to add error bars
    for _, row in plot_df.iterrows():
        task_label = row['TaskLabel']
        rep_label = row['RepresentationLabel']
        
        if task_label in x_pos_map and rep_label in hue_order:
            x_pos_group = x_pos_map[task_label]
            hue_index = hue_order.index(rep_label)
            
            # Calculate center x-coordinate for the bar
            x_coord = x_pos_group + (hue_index - (num_hues - 1) / 2) * bar_width
            
            y_val = row[metric]
            y_err = [[y_val - row[f'{metric}_CI_Lower']], [row[f'{metric}_CI_Upper'] - y_val]]
            
            ax.errorbar(x=x_coord, y=y_val, yerr=y_err, fmt='none', c='black', capsize=2)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Prediction Task', fontsize=12)
    ax.set_ylabel(f'Test Set {metric}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Data Representation')
    
    save_figure(fig, filename)
    
def generate_champion_model_plot(df: pd.DataFrame, metric: str, filename: str, title: str):
    """
    Generate a plot showing the best performing semantic model for each task vs. baseline.
    Now includes a confidence interval for the baseline model marker.
    """
    logging.info(f"Generating champion model plot for {metric}: {title}")
    
    semantic_df = df[df['Representation'] != 'Baseline']
    if semantic_df.empty:
        logging.warning("No semantic models found for champion plot.")
        return
        
    best_indices = semantic_df.groupby('Task', observed=True)[metric].idxmax()
    champion_df = semantic_df.loc[best_indices].copy()

    champion_df['TaskLabel'] = champion_df['Task'].map(TASK_LABELS)
    champion_df['ChampionLabel'] = champion_df.apply(
        lambda r: f"{REP_LABELS.get(r['Representation'])} - {PROMPT_LABELS.get(r['Prompt'], r['Prompt'])} ({r['Model']})", axis=1
    )

    # Fetch baseline scores including confidence intervals
    baseline_cols = ['Task', metric, f'{metric}_CI_Lower', f'{metric}_CI_Upper']
    baseline_data = df[df['Prompt'] == 'XGBoost'][baseline_cols].set_index('Task')
    champion_df = champion_df.join(baseline_data.rename(columns={
        metric: 'Baseline_Metric',
        f'{metric}_CI_Lower': 'Baseline_CI_Lower',
        f'{metric}_CI_Upper': 'Baseline_CI_Upper'
    }), on='Task')

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=champion_df, x=metric, y='TaskLabel', hue='ChampionLabel', palette='magma', dodge=False, ax=ax)

    x_err = [champion_df[metric] - champion_df[f'{metric}_CI_Lower'], champion_df[f'{metric}_CI_Upper'] - champion_df[metric]]
    ax.errorbar(x=champion_df[metric], y=np.arange(len(champion_df)), xerr=x_err,
                fmt='none', ecolor='black', capsize=3)
    
    y_coords_map = {label.get_text(): i for i, label in enumerate(ax.get_yticklabels())}
    for _, row in champion_df.iterrows():
        if pd.notna(row['Baseline_Metric']) and row['TaskLabel'] in y_coords_map:
            y_coord = y_coords_map[row['TaskLabel']]
            
            # Calculate and plot baseline error bar
            lower_err = row['Baseline_Metric'] - row['Baseline_CI_Lower']
            upper_err = row['Baseline_CI_Upper'] - row['Baseline_Metric']
            ax.errorbar(x=[row['Baseline_Metric']], y=[y_coord], xerr=[[lower_err], [upper_err]],
                        fmt='D', color='blue', markersize=8, capsize=5, zorder=5)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(f'Best Test Set {metric} (with 95% CI)', fontsize=12)
    ax.set_ylabel('Prediction Task', fontsize=12)
    if ax.get_legend(): ax.get_legend().remove()

    legend_elements = [Line2D([0], [0], marker='D', color='w', label='XGBoost Baseline (with 95% CI)',
                              markerfacecolor='blue', markersize=10)]
    ax.legend(handles=legend_elements, title="Reference")

    save_figure(fig, filename)


# --- Table Generation ---

def generate_results_table(df: pd.DataFrame, filename: str) -> None:
    """Generate and save a formatted Markdown table of all results."""
    logging.info("Generating Markdown results table...")
    
    df_table = df.copy()
    
    for metric in ['AUROC', 'AUPRC']:
        ci_lower, ci_upper = f'{metric}_CI_Lower', f'{metric}_CI_Upper'
        df_table[f'{metric} (95% CI)'] = df_table.apply(
            lambda r: f"{r[metric]:.4f} ({r[ci_lower]:.4f} - {r[ci_upper]:.4f})"
            if pd.notnull(r[ci_lower]) else f"{r[metric]:.4f}", axis=1
        )
    
    df_table['Representation'] = df_table['Representation'].map(REP_LABELS)
    df_table['Prompt'] = df_table['Prompt'].map(PROMPT_LABELS)
    
    table_cols = ['Task', 'Representation', 'Prompt', 'Model', 'AUROC (95% CI)', 'AUPRC (95% CI)']
    df_table = df_table[table_cols]
    
    markdown_table = df_table.to_markdown(index=False)
    full_table_text = "## Table 1: Full Experimental Results\n\n" + markdown_table
    
    save_path = OUTPUT_DIR / filename
    save_path.write_text(full_table_text, encoding='utf-8')
    logging.info(f"Markdown table saved to: {save_path}")


# --- Main Execution ---

def main():
    """Main function to run the analysis and generate outputs."""
    setup_environment()
    
    results = load_all_results(RESULTS_DIR, BASELINE_RESULTS_DIR)
    if not results:
        return
        
    full_df = create_summary_dataframe(results)
    if full_df.empty:
        logging.error("DataFrame is empty after processing results. Cannot generate figures.")
        return
    
    # This dataframe contains all embedding model results, before picking the 'best'
    embedding_df_full = full_df[full_df['Representation'] != 'Baseline']

    # Create a summary dataframe that picks the single best model for each Rep-Prompt-Task combo
    # This is used for the higher-level summary plots (Figs 2-5)
    best_indices = embedding_df_full.loc[embedding_df_full.groupby(['Task', 'Representation', 'Prompt'], observed=True)['AUROC'].idxmax()]
    baseline_df = full_df[full_df['Representation'] == 'Baseline']
    summary_df = pd.concat([best_indices, baseline_df], ignore_index=True)

    # --- Generate Plots based on user feedback (Per-Task) ---
    logging.info("Generating per-task performance plots...")
    tasks = summary_df['Task'].unique()
    for task in tasks:
        task_df_summary = summary_df[summary_df['Task'] == task]
        task_label = TASK_LABELS.get(task, task)
        
        # Figure 2: Compares representation types (using best model for each)
        generate_representation_barplot(
            task_df_summary, 'AUROC', f'figure_2_auroc_representation_barplot_{task}.png',
            f'Figure 2: AUROC by Representation for {task_label} (with 95% CI)'
        )
        
        # Figure 3: Shows performance lift over baseline
        generate_performance_lift_plot(
            task_df_summary, 'AUROC', f'figure_3_auroc_lift_{task}.png',
            f'Figure 3: AUROC Improvement Over Best Baseline for {task_label}'
        )

        # **NEW** Figure 6: Detailed comparison of all embedding models for the task
        task_df_full_embeddings = embedding_df_full[embedding_df_full['Task'] == task]
        generate_model_comparison_plot(
            task_df_full_embeddings, 'AUROC', f'figure_6_auroc_model_comparison_{task}.png',
            f'Figure 6: Detailed Embedding Model Comparison for {task_label}'
        )

    # --- Generate Cross-Task Summary Plots ---
    logging.info("Generating cross-task summary plots...")
    # Figure 4: Compares best representations across all tasks
    generate_task_comparison_plot(
        summary_df, 'AUROC', 'figure_4_task_comparison_auroc.png',
        'Figure 4: Best Model AUROC by Representation and Task (with 95% CI)'
    )
    # Figure 5: Shows the single champion model for each task vs. baseline
    generate_champion_model_plot(
        summary_df, 'AUROC', 'figure_5_champion_models_auroc.png',
        'Figure 5: Champion Semantic Model vs. XGBoost Baseline by AUROC (with 95% CI)'
    )
    
    # --- Generate Final Table ---
    generate_results_table(full_df, 'table_1_full_results.md')
    
    logging.info("All manuscript figures and tables have been generated successfully.")

if __name__ == "__main__":
    main()