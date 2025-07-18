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
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


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
TASK_LABELS = {
    'mort_hosp': 'Hospital Mortality',
    'los_3': 'Length-of-Stay > 3 Days',
    'los_7': 'Length-of-Stay > 7 Days',
    'readmission_30': '30-Day Readmission',
    'intervention_vent': 'Mechanical Ventilation',
    'intervention_vaso': 'Vasopressor Administration'
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
    
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            for task_dir in os.listdir(model_path):
                task_path = os.path.join(model_path, task_dir)
                if os.path.isdir(task_path):
                    for filename in os.listdir(task_path):
                        if filename.startswith('results_') and filename.endswith('.pkl'):
                            try:
                                filepath = os.path.join(task_path, filename)
                                with open(filepath, 'rb') as f:
                                    data = pickle.load(f)
                                    data['model_name'] = model_dir
                                    data['task'] = task_dir
                                    all_results.append(data)
                            except Exception as e:
                                logging.warning(f"Could not load file {filepath}: {e}")

    # Load baseline model results from task-specific pickle files
    for task_dir in os.listdir(baseline_dir):
        task_path = os.path.join(baseline_dir, task_dir)
        if os.path.isdir(task_path):
            baseline_files = {
                'results_xgboost_baseline.pkl': 'Baseline_XGBoost',
                'results_elastic_net_baseline.pkl': 'Baseline_ElasticNet'
            }
            for filename, arm_name in baseline_files.items():
                filepath = os.path.join(task_path, filename)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                            new_record = data.copy()
                            new_record['experimental_arm'] = arm_name
                            if 'XGBoost' in arm_name:
                                new_record['model_name'] = 'XGBoost'
                            elif 'ElasticNet' in arm_name:
                                new_record['model_name'] = 'ElasticNet'
                            new_record['task'] = task_dir 
                            all_results.append(new_record)
                    except Exception as e:
                        logging.warning(f"Could not load baseline file {filepath}: {e}")
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
            'Task': res.get('task', 'Unknown'),
            'Representation': rep,
            'Prompt': prompt,
            'Model': res.get('model_name', 'Unknown'),
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
    
    df['Task'] = pd.Categorical(df['Task'], categories=TASK_LABELS.keys(), ordered=True)
    df['Representation'] = pd.Categorical(df['Representation'], categories=REPRESENTATIONS, ordered=True)
    return df.sort_values(by=['Task', 'Representation', 'Prompt', 'Model'])

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

    # Calculate confidence intervals for the lift
    lower_ci_col = f'{metric}_CI_Lower'
    upper_ci_col = f'{metric}_CI_Upper'
    error_bars = None
    xlabel = f'Change in {metric} vs. Best Baseline'
    if lower_ci_col in df_copy.columns and upper_ci_col in df_copy.columns:
        valid_ci = df_copy[lower_ci_col].notna() & df_copy[upper_ci_col].notna()
        
        lower_error = pd.Series(np.nan, index=df_copy.index)
        upper_error = pd.Series(np.nan, index=df_copy.index)
        
        lower_error[valid_ci] = df_copy.loc[valid_ci, lift_metric_name] - (df_copy.loc[valid_ci, lower_ci_col] - best_baseline_score)
        upper_error[valid_ci] = (df_copy.loc[valid_ci, upper_ci_col] - best_baseline_score) - df_copy.loc[valid_ci, lift_metric_name]
        
        error_bars = [lower_error, upper_error]
        xlabel += ' (with 95% CI)'

    df_copy['Arm'] = df_copy.apply(lambda row: f"{REP_LABELS.get(row['Representation'], row['Representation'])} - {PROMPT_LABELS.get(row['Prompt'], row['Prompt'])}", axis=1)
    df_copy = df_copy.sort_values(lift_metric_name, ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df_copy, x=lift_metric_name, y='Arm', palette='coolwarm', hue='Arm')
    if ax.get_legend():
        ax.get_legend().remove()

    if error_bars:
        data_for_errorbars = df_copy[[lift_metric_name]].copy()
        data_for_errorbars['lower'] = error_bars[0]
        data_for_errorbars['upper'] = error_bars[1]
        data_for_errorbars = data_for_errorbars.dropna()
        
        if not data_for_errorbars.empty:
            y_coords = np.array([list(df_copy.index).index(i) for i in data_for_errorbars.index])
            ax.errorbar(x=data_for_errorbars[lift_metric_name], y=y_coords, 
                        xerr=[data_for_errorbars['lower'], data_for_errorbars['upper']], 
                        fmt='none', ecolor='black', capsize=3)

    ax.set_title(f'{title_prefix}: {metric} Improvement Over Best Baseline Model', fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Experimental Arm', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    logging.info(f"Performance lift plot saved to: {save_path}")
    plt.close()


def generate_model_comparison_plot(df: pd.DataFrame, metric: str = 'AUROC', filename: str = 'figure_7_model_comparison_auroc.png', title_prefix: str = 'Figure 7') -> None:
    """Generate a plot comparing the performance of different embedding models against baselines."""
    logging.info(f"Generating {metric} model comparison plot...")
    
    df_embedding_only = df[df['Representation'] != 'Baseline'].copy()
    df_baselines = df[df['Representation'] == 'Baseline']
    
    if df_embedding_only.empty:
        logging.warning("No embedding model data to generate model comparison plot.")
        return

    # Remove 'Baseline' from the categories to prevent it from showing up on the plot
    embedding_reps = [rep for rep in REPRESENTATIONS if rep != 'Baseline']
    df_embedding_only['Representation'] = pd.Categorical(
        df_embedding_only['Representation'], categories=embedding_reps, ordered=True
    )

    df_embedding_only['Model'] = df_embedding_only['Model'].str.replace('embedding_model_results_', '', regex=False)
    
    hue_order = sorted(df_embedding_only['Model'].unique())

    g = sns.catplot(
        data=df_embedding_only,
        x='Representation',
        y=metric,
        hue='Model',
        hue_order=hue_order,
        col='Prompt',
        kind='bar',
        height=6,
        aspect=0.8,
        palette='viridis',
        legend=False,  # We will create a manual legend
        col_order=PROMPTS,
        errorbar=None # Turn off seaborn's error bars
    )

    # Add our own error bars from pre-computed CIs
    lower_ci_col = f'{metric}_CI_Lower'
    upper_ci_col = f'{metric}_CI_Upper'
    if lower_ci_col in df_embedding_only.columns and upper_ci_col in df_embedding_only.columns:
        err_df = df_embedding_only.copy()
        err_df['err_lower'] = err_df[metric] - err_df[lower_ci_col]
        err_df['err_upper'] = err_df[upper_ci_col] - err_df[metric]
        
        for ax_idx, ax in enumerate(g.axes.flat):
            prompt = g.col_names[ax_idx]

            # Get models present in this facet's data, preserving the original hue_order
            facet_data = df_embedding_only[df_embedding_only['Prompt'] == prompt]
            models_in_facet = [m for m in hue_order if m in facet_data['Model'].unique()]
            
            for container_idx, container in enumerate(ax.containers):
                if container_idx >= len(models_in_facet):
                    continue
                model = models_in_facet[container_idx]

                for bar_idx, bar in enumerate(container):
                    rep = embedding_reps[bar_idx]
                    
                    current_data = err_df[
                        (err_df['Prompt'] == prompt) &
                        (err_df['Representation'] == rep) &
                        (err_df['Model'] == model)
                    ]

                    if not current_data.empty and current_data[[lower_ci_col, upper_ci_col]].notna().all().all():
                        y = bar.get_height()
                        x = bar.get_x() + bar.get_width() / 2
                        err_lower = current_data['err_lower'].iloc[0]
                        err_upper = current_data['err_upper'].iloc[0]
                        ax.errorbar(x, y, yerr=[[err_lower], [err_upper]], fmt='none', ecolor='black', capsize=2)

    # Add baseline lines to all subplots
    if not df_baselines.empty:
        baseline_colors = {'XGBoost': 'crimson', 'ElasticNet': 'darkgreen'}
        for _, baseline_row in df_baselines.iterrows():
            score = baseline_row[metric]
            color = baseline_colors.get(baseline_row['Prompt'], 'gray')
            for ax in g.axes.flat:
                ax.axhline(y=score, color=color, linestyle='--')

    g.fig.suptitle(f'{title_prefix}: {metric} Comparison of Embedding Models Across Representations and Prompts', y=1.03, fontsize=16)
    g.set_axis_labels("Representation", f"Test Set {metric} (with 95% CI)")
    g.set_titles("Prompt: {col_name}")
    g.despine(left=True)

    for ax in g.axes.flat:
        labels = [REP_LABELS.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # Manually create a combined legend
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    handles = []
    
    # Create handles for embedding models (bars)
    palette = sns.color_palette('viridis', n_colors=len(hue_order))
    for i, level in enumerate(hue_order):
        handles.append(mpatches.Patch(color=palette[i], label=level))
        
    # Create handles for baseline models (lines)
    if not df_baselines.empty:
        baseline_colors = {'XGBoost': 'crimson', 'ElasticNet': 'darkgreen'}
        for _, baseline_row in df_baselines.iterrows():
            model_name_key = baseline_row['Prompt']
            model_name_label = PROMPT_LABELS.get(model_name_key, model_name_key)
            score = baseline_row[metric]
            color = baseline_colors.get(model_name_key, 'gray')
            label = f'{model_name_label} Baseline ({score:.4f})'
            handles.append(Line2D([0], [0], color=color, linestyle='--', label=label))
            
    g.fig.legend(handles=handles, title='Model Type', bbox_to_anchor=(1.02, 0.5), loc='center left')

    g.fig.tight_layout(rect=(0, 0, 0.9, 0.97))  # Adjust for legend

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Model comparison plot saved to: {save_path}")
    plt.close('all')

def generate_task_comparison_plot(df: pd.DataFrame, metric: str = 'AUROC', filename: str = 'figure_9_task_comparison.png', title_prefix: str = 'Figure 9'):
    """Generate a bar plot comparing representation performance across all prediction tasks."""
    logging.info(f"Generating {metric} comparison plot across tasks...")
    
    # Use the best performing model for each (Task, Representation) pair
    best_indices = df.groupby(['Task', 'Representation'], observed=True)[metric].idxmax()
    plot_df = df.loc[best_indices].copy()
    
    plot_df['Task'] = plot_df['Task'].map(TASK_LABELS)
    plot_df['Representation'] = plot_df['Representation'].map(REP_LABELS)

    # Re-order categories based on the mapped labels for correct plotting
    plot_df['Task'] = pd.Categorical(plot_df['Task'], categories=TASK_LABELS.values(), ordered=True)
    plot_df['Representation'] = pd.Categorical(plot_df['Representation'], categories=REP_LABELS.values(), ordered=True)

    plt.figure(figsize=(16, 9))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=plot_df, x='Task', y=metric, hue='Representation', palette='viridis')

    # Add error bars for confidence intervals
    lower_ci_col = f'{metric}_CI_Lower'
    upper_ci_col = f'{metric}_CI_Upper'
    if lower_ci_col in plot_df.columns and upper_ci_col in plot_df.columns:
        # Get the positions of the bars
        hue_order = [h.get_label() for h in ax.legend_.get_texts()]
        x_ticks = [t.get_text() for t in ax.get_xticklabels()]
        
        # Calculate errors
        err_data = plot_df.copy()
        err_data['err_lower'] = err_data[metric] - err_data[lower_ci_col]
        err_data['err_upper'] = err_data[upper_ci_col] - err_data[metric]

        for i, bar in enumerate(ax.patches):
            # Find the corresponding data
            task_idx = i // len(hue_order)
            hue_idx = i % len(hue_order)
            task_label = x_ticks[task_idx]
            hue_label = hue_order[hue_idx]

            current_data = err_data[(err_data['Task'] == task_label) & (err_data['Representation'] == hue_label)]
            
            if not current_data.empty and current_data[[lower_ci_col, upper_ci_col]].notna().all().all():
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                err_lower = current_data['err_lower'].iloc[0]
                err_upper = current_data['err_upper'].iloc[0]
                ax.errorbar(x, y, yerr=[[err_lower], [err_upper]], fmt='none', ecolor='black', capsize=2)

    ax.set_title(f'{title_prefix}: Best Model {metric} by Representation and Prediction Task (with 95% CI)', fontsize=16, pad=20)
    ax.set_xlabel('Prediction Task', fontsize=12)
    ax.set_ylabel(f'Test Set {metric}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Data Representation')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    logging.info(f"Task comparison plot saved to: {save_path}")
    plt.close()

def generate_task_heatmaps(df: pd.DataFrame, metric: str = 'AUROC', filename_prefix: str = 'figure_10', title_prefix: str = 'Figure 10'):
    """Generate a separate heatmap for each prediction task."""
    logging.info(f"Generating {metric} heatmaps for each task...")
    
    tasks = df['Task'].unique()
    df_embedding_only = df[df['Representation'] != 'Baseline'].copy()

    for i, task in enumerate(tasks):
        task_df = df_embedding_only[df_embedding_only['Task'] == task]
        if task_df.empty:
            continue

        heatmap_data = task_df.pivot_table(index='Representation', columns='Prompt', values=metric, aggfunc='max') # Use max to handle multiple models
        heatmap_data.index = heatmap_data.index.map(REP_LABELS)
        heatmap_data.columns = heatmap_data.columns.map(PROMPT_LABELS)
        
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="white")
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", linewidths=.5, cbar_kws={'label': f'{metric} Score'})
        
        task_title = TASK_LABELS.get(task, task)
        ax.set_title(f'{title_prefix}.{i+1}: Test Set {metric} for {task_title}', fontsize=16, pad=20)
        ax.set_xlabel('Prompting Strategy', fontsize=12)
        ax.set_ylabel('Data Representation', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        filename = f"{filename_prefix}_{task}_{metric.lower()}_heatmap.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        logging.info(f"Task heatmap saved to: {save_path}")
        plt.close()

def generate_champion_model_plot(df: pd.DataFrame, metric: str = 'AUROC', filename: str = 'figure_11_champion_models_auroc.png', title_prefix: str = 'Figure 11'):
    """Generate a plot showing the best performing model for each task."""
    logging.info(f"Generating champion model plot for {metric}...")

    # Find the best overall SEMANTIC model for each task
    semantic_df = df[df['Representation'] != 'Baseline']
    if semantic_df.empty:
        logging.warning("No semantic models found to generate champion plot.")
        return
        
    best_indices = semantic_df.groupby('Task')[metric].idxmax()
    champion_df = semantic_df.loc[best_indices].copy()

    champion_df['TaskLabel'] = champion_df['Task'].map(TASK_LABELS)
    champion_df['ChampionLabel'] = champion_df.apply(
        lambda row: f"{REP_LABELS.get(row['Representation'])} - {PROMPT_LABELS.get(row['Prompt'], row['Prompt'])} ({row['Model']})", axis=1
    )

    # Find the XGBoost baseline model for each task for comparison
    baseline_df = df[(df['Representation'] == 'Baseline') & (df['Prompt'] == 'XGBoost')].copy()
    if not baseline_df.empty:
        # Each task has one XGBoost baseline model
        xgboost_baselines = baseline_df[['Task', metric]].rename(columns={metric: 'XGBoost_Baseline_Metric'})
        champion_df = pd.merge(champion_df, xgboost_baselines, on='Task', how='left')

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(data=champion_df, x=metric, y='TaskLabel', hue='ChampionLabel', palette='magma', dodge=False)

    # Add error bars for the champion semantic model
    lower_ci_col = f'{metric}_CI_Lower'
    upper_ci_col = f'{metric}_CI_Upper'
    if lower_ci_col in champion_df.columns and upper_ci_col in champion_df.columns:
        # Create a map from y-tick labels to their numerical position
        y_tick_labels = [label.get_text() for label in ax.get_yticklabels()]
        y_coords_map = {label: i for i, label in enumerate(y_tick_labels)}

        err_data = champion_df.copy()
        err_data['err_lower'] = err_data[metric] - err_data[lower_ci_col]
        err_data['err_upper'] = err_data[upper_ci_col] - err_data[metric]
        
        for _, row in err_data.iterrows():
            if pd.notna(row[lower_ci_col]) and pd.notna(row[upper_ci_col]):
                task_label = row['TaskLabel']
                if task_label in y_coords_map:
                    y_coord = y_coords_map[task_label]
                    ax.errorbar(x=row[metric], y=y_coord, 
                                xerr=[[row['err_lower']], [row['err_upper']]], 
                                fmt='none', ecolor='black', capsize=3)

    # The 'hue' argument automatically creates a legend. We remove it as it's too cluttered.
    if ax.get_legend():
        ax.get_legend().remove()

    # Add baseline markers
    if 'XGBoost_Baseline_Metric' in champion_df.columns:
        # Create a map from y-tick labels to their numerical position again for safety
        y_tick_labels = [label.get_text() for label in ax.get_yticklabels()]
        y_coords_map = {label: i for i, label in enumerate(y_tick_labels)}

        for _, row in champion_df.iterrows():
            if pd.notna(row['XGBoost_Baseline_Metric']):
                task_label = row['TaskLabel']
                if task_label in y_coords_map:
                    y_coord = y_coords_map[task_label]
                    ax.plot(row['XGBoost_Baseline_Metric'], y_coord, 'D', color='blue', markersize=8, label='XGBoost Baseline' if not y_coords_map else "")


    ax.set_title(f'{title_prefix}: Champion Semantic Model vs. XGBoost Baseline by {metric} (with 95% CI)', fontsize=16, pad=20)
    ax.set_xlabel(f'Best Test Set {metric}', fontsize=12)
    ax.set_ylabel('Prediction Task', fontsize=12)
    
    # Create a clean legend for the baseline marker
    if 'XGBoost_Baseline_Metric' in champion_df.columns:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='D', color='w', label='XGBoost Baseline', markerfacecolor='blue', markersize=10)]
        ax.legend(handles=legend_elements, title="Reference")

    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    logging.info(f"Champion model plot saved to: {save_path}")
    plt.close()

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

    
    table_df = df_copy[['Task', 'Representation', 'Prompt', 'Model', 'AUROC_Formatted', 'AUPRC_Formatted']]
    table_df.columns = ['Task', 'Representation', 'Prompt', 'Model', 'AUROC (95% CI)', 'AUPRC (95% CI)']
    
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

    # Create a summary df with only the best performing model for each (Task, Rep, Prompt) triplet
    embedding_df = full_df[full_df['Representation'] != 'Baseline']
    best_embedding_df = pd.DataFrame()
    if not embedding_df.empty:
        best_embedding_indices = embedding_df.groupby(['Task', 'Representation', 'Prompt'], observed=True)['AUROC'].idxmax()
        best_embedding_df = embedding_df.loc[best_embedding_indices]
    
    baseline_df = full_df[full_df['Representation'] == 'Baseline']
    summary_df = pd.concat([best_embedding_df, baseline_df], ignore_index=True)


    champion = summary_df.loc[summary_df['AUROC'].idxmax()]
    logging.info("\n" + "="*50)
    logging.info("🏆 Champion Semantic System Identified (based on best model per configuration) 🏆")
    logging.info(f"   Task: {TASK_LABELS.get(champion['Task'], champion['Task'])}")
    logging.info(f"   Best Representation: {REP_LABELS.get(champion['Representation'], champion['Representation'])}")
    logging.info(f"   Best Prompt: {PROMPT_LABELS.get(champion['Prompt'], champion['Prompt'])}")
    logging.info(f"   Best Test AUROC: {champion['AUROC']:.4f} (Model: {champion['Model']})")
    logging.info("="*50 + "\n")
    
    # --- Handle incomplete baselines for aggregated plots ---
    tasks = full_df['Task'].unique()
    df_for_agg = summary_df.copy()
    full_df_for_agg = full_df.copy()

    elastic_net_task_count = full_df[full_df['Prompt'] == 'ElasticNet']['Task'].nunique()
    if 0 < elastic_net_task_count < len(tasks):
        logging.warning("ElasticNet results found for only a subset of tasks. Excluding from aggregated plots.")
        df_for_agg = df_for_agg[df_for_agg['Prompt'] != 'ElasticNet']
        full_df_for_agg = full_df_for_agg[full_df_for_agg['Prompt'] != 'ElasticNet']
    
    # Generate original outputs using the summary dataframe (best model per config)
    # To generate the original figures, we need a dataframe that averages over tasks.
    agg_df_groups = df_for_agg.groupby(['Representation', 'Prompt', 'Model'], observed=True)
    agg_summary_df = agg_df_groups[['AUROC', 'AUPRC']].mean().reset_index()

    generate_heatmap(agg_summary_df, metric='AUROC', filename='figure_1_auroc_heatmap.png', title_prefix='Figure 1')
    generate_heatmap(agg_summary_df, metric='AUPRC', filename='figure_2_auprc_heatmap.png', title_prefix='Figure 2')
    
    generate_interaction_plot(agg_summary_df, metric='AUROC', filename='figure_3_auroc_interaction_plot.png', title_prefix='Figure 3')
    generate_interaction_plot(agg_summary_df, metric='AUPRC', filename='figure_4_auprc_interaction_plot.png', title_prefix='Figure 4')
    
    generate_performance_lift_plot(agg_summary_df, metric='AUROC', filename='figure_5_auroc_performance_lift.png', title_prefix='Figure 5')
    generate_performance_lift_plot(agg_summary_df, metric='AUPRC', filename='figure_6_auprc_performance_lift.png', title_prefix='Figure 6')
    
    # This plot requires CIs, so we should aggregate the full_df instead
    agg_full_df_groups = full_df_for_agg.groupby(['Representation', 'Prompt', 'Model'], observed=True)
    agg_full_df = agg_full_df_groups.agg({
        'AUROC': 'mean', 'AUROC_CI_Lower': 'mean', 'AUROC_CI_Upper': 'mean',
        'AUPRC': 'mean', 'AUPRC_CI_Lower': 'mean', 'AUPRC_CI_Upper': 'mean'
    }).reset_index()
    generate_model_comparison_plot(agg_full_df, metric='AUROC', filename='figure_7_model_comparison_auroc.png', title_prefix='Figure 7')
    generate_model_comparison_plot(agg_full_df, metric='AUPRC', filename='figure_8_model_comparison_auprc.png', title_prefix='Figure 8')

    # --- Generate New Task-Based Figures ---
    generate_task_comparison_plot(summary_df, metric='AUROC', filename='figure_9_task_comparison_auroc.png', title_prefix='Figure 9')
    generate_task_heatmaps(summary_df, metric='AUROC', filename_prefix='figure_10', title_prefix='Figure 10')
    generate_champion_model_plot(summary_df, metric='AUROC', filename='figure_11_champion_models_auroc.png', title_prefix='Figure 11')
    
    generate_results_table(full_df)
    
    logging.info("All manuscript figures and tables have been generated successfully.")

if __name__ == "__main__":
    main()
