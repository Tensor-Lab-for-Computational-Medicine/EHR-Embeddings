# generate_manuscript_figures_corrected.py
"""
Generate publication figures/tables from experiment results with concise, modular plotting.
Includes CIs, numeric baselines, and per-model/task comparison (prompts vs representations).
"""
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from scipy import stats

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
    # Manuscript-friendly defaults: paper context, readable fonts
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.4)
    plt.rcParams.update({
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
    })

def _fast_delong_auc_covariance(predictions: np.ndarray, labels: np.ndarray):
    """Fast DeLong covariance estimate for correlated ROC AUCs."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    m, n = pos_mask.sum(), neg_mask.sum()
    if m == 0 or n == 0:
        raise ValueError("Both positive and negative samples are required for DeLong test.")

    k = predictions.shape[0]
    aucs = np.zeros(k)
    v10 = np.zeros((k, m))
    v01 = np.zeros((k, n))
    for idx in range(k):
        pred = predictions[idx]
        pos_pred = pred[pos_mask]
        neg_pred = pred[neg_mask]
        comp = np.add.outer(pos_pred, -neg_pred)
        aucs[idx] = np.mean(comp > 0) + 0.5 * np.mean(comp == 0)
        for i, p in enumerate(pos_pred):
            v10[idx, i] = np.mean(neg_pred < p) + 0.5 * np.mean(neg_pred == p)
        for j, q in enumerate(neg_pred):
            v01[idx, j] = np.mean(pos_pred > q) + 0.5 * np.mean(pos_pred == q)
    s10 = np.cov(v10) if k > 1 else np.var(v10, ddof=1).reshape(1, 1)
    s01 = np.cov(v01) if k > 1 else np.var(v01, ddof=1).reshape(1, 1)
    return aucs, (s10 / m) + (s01 / n)


def calculate_delong_pvalue(y_true, p_baseline, p_champion):
    """Two-sided DeLong test p-value for AUROC difference (champion vs baseline)."""
    if y_true is None or p_baseline is None or p_champion is None:
        return None
    y_true = np.asarray(y_true).ravel()
    p_baseline = np.asarray(p_baseline).ravel()
    p_champion = np.asarray(p_champion).ravel()
    if len(y_true) != len(p_baseline) or len(y_true) != len(p_champion):
        logging.warning("Shape mismatch in predictions for DeLong p-value calculation.")
        return None
    try:
        _, cov = _fast_delong_auc_covariance(np.vstack([p_champion, p_baseline]), y_true)
        var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        if var_diff <= 0:
            return 1.0
        z = (roc_auc_score(y_true, p_champion) - roc_auc_score(y_true, p_baseline)) / np.sqrt(var_diff)
        return float(2 * stats.norm.sf(abs(z)))
    except Exception as e:
        logging.warning(f"DeLong test failed: {e}")
        return None

# Shared styles and light helpers (kept minimal for clarity)
BASELINE_STYLES = {
    'XGBoost': {'marker': 'D', 'color': 'crimson', 'label': 'XGBoost Baseline'},
    'ElasticNet': {'marker': 's', 'color': '#663399', 'label': 'ElasticNet Baseline'}
}
TASK_KEY_MAP = {v: k for k, v in TASK_LABELS.items()}

def add_errorbar(ax: plt.Axes, x: float, y: float, lo: float, hi: float, **kwargs) -> None:
    ax.errorbar(x=x, y=y, yerr=[[lo], [hi]], fmt=kwargs.pop('fmt', 'none'), capsize=kwargs.pop('capsize', 3), **kwargs)

def add_ci_for_catplot(ax: plt.Axes, df: pd.DataFrame, x_field: str, hue_field: str, metric: str,
                       hue_order: List[str], bar_width: float, x_pos_map: Dict[str, int]) -> None:
    num_hues = len(hue_order)
    for _, row in df.iterrows():
        xl, hl = row[x_field], row[hue_field]
        if xl in x_pos_map and hl in hue_order:
            xg = x_pos_map[xl]
            hi = hue_order.index(hl)
            x_coord = xg + (hi - (num_hues - 1) / 2) * bar_width
            y_val = row[metric]
            add_errorbar(ax, x_coord, y_val, y_val - row[f'{metric}_CI_Lower'], row[f'{metric}_CI_Upper'] - y_val,
                         c='black', elinewidth=1, zorder=10)

def add_baseline_markers(ax: plt.Axes, baseline_df: pd.DataFrame, task_cats: List[str], metric: str) -> None:
    x_pos_map = {label: i for i, label in enumerate(task_cats)}
    for task_label, x_pos in x_pos_map.items():
        task_key = TASK_KEY_MAP.get(task_label)
        if not task_key: continue
        task_baselines = baseline_df[baseline_df['Task'] == task_key]
        if task_baselines.empty: continue
        for b_model_name, style in BASELINE_STYLES.items():
            bd = task_baselines[task_baselines['Prompt'] == b_model_name]
            if bd.empty: continue
            r = bd.iloc[0]
            y_val = r[metric]
            add_errorbar(ax, x_pos, y_val, y_val - r[f'{metric}_CI_Lower'], r[f'{metric}_CI_Upper'] - y_val,
                         fmt=style['marker'], color=style['color'], markersize=7, elinewidth=1.5,
                         zorder=12, markeredgewidth=1.5, markerfacecolor='none')

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
            data['filepath'] = str(filepath)
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
    
    # DEBUG: Check for predictions in baselines
    baselines_with_pred = 0
    for res in all_results:
        if 'Baseline' in res.get('experimental_arm', ''):
             yt = res.get('y_true')
             if yt is None and 'full_evaluation' in res:
                 yt = res['full_evaluation'].get('y_true')
             
             if yt is not None:
                 baselines_with_pred += 1
    logging.info(f"DEBUG: Found {baselines_with_pred} baseline results with predictions out of {len([r for r in all_results if 'Baseline' in r.get('experimental_arm', '')])} baselines.")

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

        # Extract raw predictions if available
        # xgboost_analysis.py stores in 'full_evaluation', xgboost_embedding_analysis.py in root
        y_true = res.get('y_true')
        y_pred = res.get('y_pred_proba')
        if y_true is None and 'full_evaluation' in res:
             y_true = res['full_evaluation'].get('y_true')
             y_pred = res['full_evaluation'].get('y_pred_proba')
        
        # DEBUG LOGGING FOR DATAFRAME CREATION
        # if res.get('model_name') in ['Baseline_XGBoost', 'Baseline_ElasticNet'] and y_true is None:
        #    logging.warning(f"Baseline model {res.get('model_name')} for task {res.get('task')} is missing predictions!")

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
            'y_true': y_true,
            'y_pred_proba': y_pred,
            'filepath': res.get('filepath', 'Unknown')
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

def generate_representation_barplot(df: pd.DataFrame, metric: str, filename: str) -> None:
    """
    Generate a bar plot comparing representations for each prompt, including baselines.
    This version now plots confidence intervals for each bar.
    """
    logging.info(f"Generating {metric} representation bar plot (with CI)")

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
    bar_width = bplot.patches[0].get_width() if bplot.patches else 0.8
    x_pos_map = {label: i for i, label in enumerate(x_labels)}

    # Iterate through the plotted data to add error bars at correct positions
    add_ci_for_catplot(
        ax,
        df_plot,
        x_field='PromptLabel',
        hue_field='RepresentationLabel',
        metric=metric,
        hue_order=hue_order,
        bar_width=bar_width,
        x_pos_map=x_pos_map
    )


    ax.set_xlabel('Prompting Strategy / Baseline Model', fontsize=12)
    ax.set_ylabel(f'Test Set {metric} (with 95% CI)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Representation Type')
    
    if df_plot[metric].min() > 0.4:
        ax.set_ylim(0.5, max(1.0, df_plot[f'{metric}_CI_Upper'].max() * 1.05))

    save_figure(fig, filename)
    
def generate_performance_lift_plot(df: pd.DataFrame, metric: str, filename: str) -> None:
    """
    Generate a plot showing performance lift over the best baseline model for a given task.
    Now includes baseline models in the plot for comprehensive comparison.
    """
    logging.info(f"Generating {metric} performance lift plot")

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
    ax.set_xlabel(f'Change in {metric} vs. Best Baseline (with 95% CI)', fontsize=12)
    ax.set_ylabel('Experimental Arm', fontsize=12)
    if ax.get_legend(): ax.get_legend().remove()
    
    save_figure(fig, filename)

def generate_task_comparison_by_prompt_plot(df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str, filename: str):
    """
    Generate a faceted bar plot comparing task performance for each embedding model,
    but grouping bars by Prompt (P0–P5) instead of Representation, with numeric
    baseline markers overlaid per task.
    """
    logging.info(f"Generating per-model task comparison by prompt plot for {metric}")

    if df.empty:
        logging.warning("No embedding model data available for this plot.")
        return

    # For each model, task, and prompt, find the best-performing representation
    best_rep_indices = df.groupby(['Model', 'Task', 'Prompt'], observed=True)[metric].idxmax()
    plot_df = df.loc[best_rep_indices].copy()

    plot_df['TaskLabel'] = plot_df['Task'].map(TASK_LABELS)
    plot_df['PromptLabel'] = plot_df['Prompt'].map(PROMPT_LABELS)

    # Ensure categorical ordering for consistent plots
    model_names = sorted(plot_df['Model'].unique())
    task_cats = [TASK_LABELS[t] for t in TASK_LABELS if t in plot_df['Task'].unique()]
    prompt_cats = [PROMPT_LABELS[p] for p in ['P0','P1','P2','P3','P4','P5'] if p in plot_df['Prompt'].unique()]

    plot_df['TaskLabel'] = pd.Categorical(plot_df['TaskLabel'], categories=task_cats, ordered=True)
    plot_df['PromptLabel'] = pd.Categorical(plot_df['PromptLabel'], categories=prompt_cats, ordered=True)
    plot_df = plot_df.sort_values(['Model', 'TaskLabel', 'PromptLabel'])

    palette = sns.color_palette('viridis', n_colors=len(prompt_cats))
    color_map = dict(zip(prompt_cats, palette))

    g = sns.catplot(
        data=plot_df, x='TaskLabel', y=metric, hue='PromptLabel', col='Model',
        kind='bar', palette=color_map, height=6, aspect=1.45, legend=False,
        col_wrap=2, sharey=False, col_order=model_names,
        hue_order=prompt_cats
    )

    # --- ADDING CONFIDENCE INTERVALS AND BASELINE MARKERS ---
    baseline_styles = {
        'XGBoost': {'marker': 'D', 'color': 'crimson', 'label': 'XGBoost Baseline'},
        'ElasticNet': {'marker': 's', 'color': '#663399', 'label': 'ElasticNet Baseline'}
    }
    task_key_map = {v: k for k, v in TASK_LABELS.items()}

    for model_name, ax in zip(g.col_names, g.axes.flat):
        if not ax.patches: continue

        ax_df = plot_df[plot_df['Model'] == model_name]
        if ax_df.empty: continue

        hue_order = prompt_cats
        bar_width = ax.patches[0].get_width()
        x_labels = task_cats
        x_pos_map = {label: i for i, label in enumerate(x_labels)}

        # Plot CIs for embedding model bars
        add_ci_for_catplot(
            ax,
            ax_df,
            x_field='TaskLabel',
            hue_field='PromptLabel',
            metric=metric,
            hue_order=hue_order,
            bar_width=bar_width,
            x_pos_map=x_pos_map
        )

        # Baseline markers per task
        add_baseline_markers(ax, baseline_df, task_cats, metric)

        if not ax_df.empty and ax_df[metric].min() > 0.4:
            y_max_ci = ax_df[f'{metric}_CI_Upper'].max()
            ax.set_ylim(0.5, max(1.0, y_max_ci * 1.05) if pd.notna(y_max_ci) else 1.0)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    g.set_axis_labels("Prediction Task", f'Test Set {metric} (with 95% CI)')
    g.set_titles("Model: {col_name}")

    # --- Combined Legend inside at bottom to reduce whitespace ---
    rep_legend_handles = [mpatches.Patch(color=color_map[name], label=name) for name in prompt_cats]
    baseline_legend_handles = [
        Line2D([0], [0], marker=style['marker'], color='w', label=style['label'],
               markerfacecolor='none', markeredgecolor=style['color'], 
               markeredgewidth=1.5, markersize=9)
        for _, style in baseline_styles.items()
    ]
    all_handles = rep_legend_handles + baseline_legend_handles

    # Ensure x tick labels are visible and readable across all facets
    positions = np.arange(len(task_cats))
    for ax in g.axes.flatten():
        ax.set_xticks(positions)
        ax.set_xticklabels(task_cats, rotation=45, ha='right')
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    g.fig.tight_layout()
    g.fig.subplots_adjust(bottom=0.15)
    g.fig.legend(handles=all_handles, loc='lower center', ncol=max(3, len(all_handles)), frameon=False)

    save_figure(g.fig, filename)

def generate_task_comparison_plot(df: pd.DataFrame, metric: str, filename: str):
    """Generate a bar plot comparing representation performance across all prediction tasks."""
    logging.info(f"Generating {metric} comparison plot across tasks")
    
    best_indices = df.groupby(['Task', 'Representation'], observed=True)[metric].idxmax()
    plot_df = df.loc[best_indices].copy()
    
    plot_df['TaskLabel'] = plot_df['Task'].map(TASK_LABELS)
    plot_df['RepresentationLabel'] = plot_df['Representation'].map(REP_LABELS)

    plot_df['TaskLabel'] = pd.Categorical(plot_df['TaskLabel'], categories=TASK_LABELS.values(), ordered=True)
    plot_df['RepresentationLabel'] = pd.Categorical(plot_df['RepresentationLabel'], categories=REP_LABELS.values(), ordered=True)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.set_theme(style="whitegrid")
    
    bplot = sns.barplot(data=plot_df, x='TaskLabel', y=metric, hue='RepresentationLabel', palette='viridis', ax=ax)
    
    hue_order = [x.get_text() for x in ax.get_legend().get_texts()]
    num_hues = len(hue_order)
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_pos_map = {label: i for i, label in enumerate(x_tick_labels)}
    bar_width = bplot.patches[0].get_width() if bplot.patches else 0.8
    
    for _, row in plot_df.iterrows():
        task_label = row['TaskLabel']
        rep_label = row['RepresentationLabel']
        
        if task_label in x_pos_map and rep_label in hue_order:
            x_pos_group = x_pos_map[task_label]
            hue_index = hue_order.index(rep_label)
            
            x_coord = x_pos_group + (hue_index - (num_hues - 1) / 2) * bar_width
            
            y_val = row[metric]
            y_err = [[y_val - row[f'{metric}_CI_Lower']], [row[f'{metric}_CI_Upper'] - y_val]]
            
            ax.errorbar(x=x_coord, y=y_val, yerr=y_err, fmt='none', c='black', capsize=2)

    ax.set_xlabel('Prediction Task', fontsize=12)
    ax.set_ylabel(f'Test Set {metric}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Data Representation')
    
    save_figure(fig, filename)
    
def generate_champion_model_plot(df: pd.DataFrame, metric: str, filename: str):
    """
    Generate a plot showing the best performing semantic model for each task vs. baseline.
    Now includes a confidence interval for the baseline model marker and shows champion model details.
    """
    logging.info(f"Generating champion model plot for {metric}")
    
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
    
    # Create shorter labels for annotations
    champion_df['ChampionShort'] = champion_df.apply(
        lambda r: f"{r['Representation']}-{r['Prompt']} ({r['Model'].split('_')[-1] if '_' in r['Model'] else r['Model']})", axis=1
    )

    # Select XGBoost numeric baseline per task
    baseline_df = df[(df['Representation'] == 'Baseline') & (df['Prompt'] == 'XGBoost')]
    if baseline_df.empty:
        logging.warning("No baseline rows found; champion plot will omit baseline markers.")
        baseline_best = pd.DataFrame(columns=['Task', metric, f'{metric}_CI_Lower', f'{metric}_CI_Upper', 'Prompt', 'y_true', 'y_pred_proba'])
    else:
        best_idx = baseline_df.groupby('Task', observed=True)[metric].idxmax()
        # Include predictions for stats
        baseline_cols = ['Task', 'Prompt', metric, f'{metric}_CI_Lower', f'{metric}_CI_Upper', 'y_true', 'y_pred_proba']
        baseline_best = baseline_df.loc[best_idx, baseline_cols]
        baseline_best = baseline_best.rename(columns={'Prompt': 'Baseline_Model'}).set_index('Task')

    champion_df = champion_df.join(baseline_best.rename(columns={
        metric: 'Baseline_Metric',
        f'{metric}_CI_Lower': 'Baseline_CI_Lower',
        f'{metric}_CI_Upper': 'Baseline_CI_Upper',
        'y_true': 'Baseline_y_true',
        'y_pred_proba': 'Baseline_y_pred'
    }), on='Task')

    # Calculate significance
    logging.info("Calculating DeLong p-values for champion models...")
    p_values = []
    for _, row in champion_df.iterrows():
        # Use champion y_true (should be identical to baseline y_true)
        p = calculate_delong_pvalue(
            row.get('y_true'),
            row.get('Baseline_y_pred'),
            row.get('y_pred_proba')
        )
        p_values.append(p)
    champion_df['p_value'] = p_values

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.set_theme(style="whitegrid")
    
    # Use a single color for all bars since we'll annotate with champion details
    bars = sns.barplot(data=champion_df, x=metric, y='TaskLabel', 
                       palette=['skyblue'], ax=ax)

    x_err = [champion_df[metric] - champion_df[f'{metric}_CI_Lower'], champion_df[f'{metric}_CI_Upper'] - champion_df[metric]]
    ax.errorbar(x=champion_df[metric], y=np.arange(len(champion_df)), xerr=x_err,
                fmt='none', ecolor='black', capsize=3)
    
    # Add text annotations showing champion model details and significance
    x_min, x_max = ax.get_xlim()
    text_x_pos = x_min + (x_max - x_min) * 0.05  # Position at 5% from left edge
    
    for i, (_, row) in enumerate(champion_df.iterrows()):
        champion_text = row['ChampionShort']
        pval = row.get('p_value')
        
        if pd.notnull(pval):
            if pval < 0.001: sig = "***"
            elif pval < 0.01: sig = "**"
            elif pval < 0.05: sig = "*"
            else: sig = "ns"
            champion_text += f" [{sig}]"
        
        ax.text(text_x_pos, i, champion_text, va='center', ha='left', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                zorder=10)
    
    y_coords_map = {label.get_text(): i for i, label in enumerate(ax.get_yticklabels())}
    for _, row in champion_df.iterrows():
        if pd.notna(row['Baseline_Metric']) and row['TaskLabel'] in y_coords_map:
            y_coord = y_coords_map[row['TaskLabel']]
            
            lower_err = row['Baseline_Metric'] - row['Baseline_CI_Lower']
            upper_err = row['Baseline_CI_Upper'] - row['Baseline_Metric']
            ax.errorbar(x=[row['Baseline_Metric']], y=[y_coord], xerr=[[lower_err], [upper_err]],
                        fmt='D', color='blue', markersize=8, capsize=5, zorder=5)

    ax.set_xlabel(f'Best Test Set {metric} (with 95% CI)', fontsize=12)
    ax.set_ylabel('Prediction Task', fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker='D', color='w', label='Best Baseline (with 95% CI)',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], color='skyblue', linewidth=6, label='Champion Semantic Model')
    ]
    ax.legend(handles=legend_elements, title="Model Type", loc='lower right')

    save_figure(fig, filename)

def generate_task_comparison_by_model_plot(df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str, filename: str):
    """
    Generate a faceted bar plot comparing task performance for each embedding model.
    Includes baseline model performance as reference markers on each subplot.
    """
    logging.info(f"Generating per-model task comparison plot for {metric}")
    
    if df.empty:
        logging.warning("No embedding model data available for this plot.")
        return

    # For each model, task, and representation, find the best-performing prompt
    best_prompt_indices = df.groupby(['Model', 'Task', 'Representation'], observed=True)[metric].idxmax()
    plot_df = df.loc[best_prompt_indices].copy()

    plot_df['TaskLabel'] = plot_df['Task'].map(TASK_LABELS)
    plot_df['RepresentationLabel'] = plot_df['Representation'].map(REP_LABELS)

    # Ensure categorical ordering for consistent plots
    model_names = sorted(plot_df['Model'].unique())
    task_cats = [TASK_LABELS[t] for t in TASK_LABELS if t in plot_df['Task'].unique()]
    rep_cats = [REP_LABELS[r] for r in REPRESENTATIONS if r in plot_df['Representation'].unique()]
    
    plot_df['TaskLabel'] = pd.Categorical(plot_df['TaskLabel'], categories=task_cats, ordered=True)
    plot_df['RepresentationLabel'] = pd.Categorical(plot_df['RepresentationLabel'], categories=rep_cats, ordered=True)
    plot_df = plot_df.sort_values(['Model', 'TaskLabel', 'RepresentationLabel'])

    palette = sns.color_palette('viridis', n_colors=len(rep_cats))
    color_map = dict(zip(rep_cats, palette))

    g = sns.catplot(
        data=plot_df, x='TaskLabel', y=metric, hue='RepresentationLabel', col='Model',
        kind='bar', palette=color_map, height=6, aspect=1.45, legend=False,
        col_wrap=2, sharey=False, col_order=model_names,
        hue_order=rep_cats
    )

    # --- ADDING CONFIDENCE INTERVALS AND BASELINE MARKERS ---
    baseline_styles = {
        'XGBoost': {'marker': 'D', 'color': 'crimson', 'label': 'XGBoost Baseline'},
        'ElasticNet': {'marker': 's', 'color': '#663399', 'label': 'ElasticNet Baseline'} # rebeccapurple
    }
    task_key_map = {v: k for k, v in TASK_LABELS.items()}

    for model_name, ax in zip(g.col_names, g.axes.flat):
        if not ax.patches: continue
        
        ax_df = plot_df[plot_df['Model'] == model_name]
        if ax_df.empty: continue

        hue_order = rep_cats
        bar_width = ax.patches[0].get_width()
        x_labels = task_cats
        x_pos_map = {label: i for i, label in enumerate(x_labels)}

        # Plot CIs for embedding model bars
        add_ci_for_catplot(
            ax,
            ax_df,
            x_field='TaskLabel',
            hue_field='RepresentationLabel',
            metric=metric,
            hue_order=hue_order,
            bar_width=bar_width,
            x_pos_map=x_pos_map
        )

        # Plot baseline markers for each task
        add_baseline_markers(ax, baseline_df, task_cats, metric)

        if not ax_df.empty and ax_df[metric].min() > 0.4:
            y_max_ci = ax_df[f'{metric}_CI_Upper'].max()
            ax.set_ylim(0.5, max(1.0, y_max_ci * 1.05) if pd.notna(y_max_ci) else 1.0)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    g.set_axis_labels("Prediction Task", f'Test Set {metric} (with 95% CI)')
    g.set_titles("Model: {col_name}")

    # --- Combined Legend inside at bottom to reduce whitespace ---
    rep_legend_handles = [mpatches.Patch(color=color_map[name], label=name) for name in rep_cats]
    baseline_legend_handles = [
        Line2D([0], [0], marker=style['marker'], color='w', label=style['label'],
               markerfacecolor='none', markeredgecolor=style['color'], 
               markeredgewidth=1.5, markersize=9)
        for _, style in baseline_styles.items()
    ]
    all_handles = rep_legend_handles + baseline_legend_handles

    # Ensure x tick labels are visible and readable across all facets
    positions = np.arange(len(task_cats))
    for ax in g.axes.flatten():
        ax.set_xticks(positions)
        ax.set_xticklabels(task_cats, rotation=45, ha='right')
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    g.fig.tight_layout()
    # Place legend at bottom inside with minimal whitespace
    g.fig.subplots_adjust(bottom=0.15)
    g.fig.legend(handles=all_handles, loc='lower center', ncol=max(3, len(all_handles)), frameon=False)

    save_figure(g.fig, filename)

# --- Table Generation (LaTeX only) ---

def generate_results_table(df: pd.DataFrame, filename: str) -> None:
    """Generate and save a LaTeX table of all results (no markdown)."""
    logging.info("Generating LaTeX results table...")
    
    df_table = df.copy()
    
    for metric in ['AUROC', 'AUPRC']:
        ci_lower, ci_upper = f'{metric}_CI_Lower', f'{metric}_CI_Upper'
        df_table[f'{metric} (95% CI)'] = df_table.apply(
            lambda r: f"{r[metric]:.4f} ({r[ci_lower]:.4f}-{r[ci_upper]:.4f})"
            if pd.notnull(r[ci_lower]) else f"{r[metric]:.4f}", axis=1
        )
    
    df_table['Representation'] = df_table['Representation'].map(REP_LABELS)
    df_table['Prompt'] = df_table['Prompt'].map(PROMPT_LABELS)
    df_table['Task'] = df_table['Task'].map(TASK_LABELS)
    
    # Sanitize LaTeX special chars
    df_table['Task'] = df_table['Task'].astype(str).str.replace('>', '\\textgreater{}', regex=False)
    df_table['Model'] = df_table['Model'].astype(str).str.replace('_', '\\_', regex=False)
    
    table_cols = ['Task', 'Representation', 'Prompt', 'Model', 'AUROC (95% CI)', 'AUPRC (95% CI)']
    df_table = df_table[table_cols]
    
    # Escape percent in headers
    df_table = df_table.rename(columns={
        'AUROC (95% CI)': 'AUROC (95\\% CI)',
        'AUPRC (95% CI)': 'AUPRC (95\\% CI)'
    })
    
    latex_table = df_table.to_latex(index=False, escape=False)
    save_path = OUTPUT_DIR / filename
    save_path.write_text(latex_table, encoding='utf-8')
    # Standalone wrapper for direct compilation in TeXworks (cropped to table)
    latex_table_standalone = (
        "\\documentclass[border=0pt]{standalone}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{adjustbox}\n"
        "\\begin{document}\n"
        "\\begin{adjustbox}{width=\\textwidth}\n"
        + latex_table +
        "\n\\end{adjustbox}\n"
        "\\end{document}\n"
    )
    (OUTPUT_DIR / (Path(filename).stem + '_standalone.tex')).write_text(latex_table_standalone, encoding='utf-8')
    logging.info(f"LaTeX tables saved to: {save_path} and standalone version")


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
    
    embedding_df_full = full_df[full_df['Representation'] != 'Baseline'].copy()
    baseline_df_full = full_df[full_df['Representation'] == 'Baseline'].copy()

    best_model_indices = embedding_df_full.groupby(['Task', 'Representation', 'Prompt'], observed=True)['AUROC'].idxmax()
    summary_embedding_df = embedding_df_full.loc[best_model_indices]
    summary_df = pd.concat([summary_embedding_df, baseline_df_full], ignore_index=True)

    # --- Generate Plots based on user feedback (Per-Task) ---
    logging.info("Generating per-task performance plots...")
    tasks = summary_df['Task'].unique()
    for task in tasks:
        task_df_summary = summary_df[summary_df['Task'] == task]
        task_label = TASK_LABELS.get(task, task)
        
        generate_representation_barplot(
            task_df_summary, 'AUROC', f'figure_2_auroc_representation_barplot_{task}.png'
        )
        
        generate_performance_lift_plot(
            task_df_summary, 'AUROC', f'figure_3_auroc_lift_{task}.png'
        )

        # Figure 6: New - Per-Model Task Comparison by Prompt (with baselines)
        # (Single figure across all tasks rather than per-task file)

    # --- Generate Cross-Task Summary Plots ---
    logging.info("Generating cross-task summary plots...")
    generate_task_comparison_plot(
        summary_df, 'AUROC', 'figure_4_task_comparison_auroc.png'
    )
    generate_champion_model_plot(
        summary_df, 'AUROC', 'figure_5_champion_models_auroc.png'
    )
    
    # --- Generate NEW Plot: Per-Model Task Comparison with Baselines (Figure 7)
    generate_task_comparison_by_model_plot(
        embedding_df_full, 
        baseline_df_full,
        'AUROC', 
        'figure_7_task_comparison_by_model_auroc_with_baselines.png'
    )

    # --- Generate NEW Plot: Per-Model Task Comparison by Prompt (Figure 6)
    generate_task_comparison_by_prompt_plot(
        embedding_df_full,
        baseline_df_full,
        'AUROC',
        'figure_6_task_comparison_by_model_auroc_by_prompt_with_baselines.png'
    )
    
    # --- Generate Final Table (LaTeX only) ---
    generate_results_table(full_df, 'table_1_full_results.tex')
    
    # --- NEW: Generate LaTeX tables ---
    def _fmt_ci(val, lo, hi):
        try:
            return f"{val:.4f} ({lo:.4f}-{hi:.4f})"
        except Exception:
            return ""

    # Champion semantic vs champion numeric (table representation of champion plot)
    try:
        # Reuse the logic from generate_champion_model_plot to get the exact same rows/models
        # but re-calculate for the table (or we could return DF from that function, but this is cleaner to keep separate)
        
        # 1. Get Champion Semantic Models
        semantic_only = summary_df[summary_df['Representation'] != 'Baseline']
        best_idx = semantic_only.groupby('Task', observed=True)['AUROC'].idxmax()
        champion_df = semantic_only.loc[best_idx].copy()
        champion_df['TaskLabel'] = champion_df['Task'].map(TASK_LABELS)
        champion_df['Champion'] = champion_df.apply(
            lambda r: f"{r['Representation']}-{r['Prompt']} ({r['Model']})", axis=1
        )

        # 2. Get XGBoost Numeric Models
        numeric = summary_df[(summary_df['Representation'] == 'Baseline') & (summary_df['Prompt'] == 'XGBoost')]
        best_num_idx = numeric.groupby('Task', observed=True)['AUROC'].idxmax()
        # Include predictions for stats
        best_numeric = numeric.loc[best_num_idx, ['Task', 'Prompt', 'AUROC', 'AUROC_CI_Lower', 'AUROC_CI_Upper', 'y_true', 'y_pred_proba']]
        best_numeric = best_numeric.rename(columns={
            'Prompt': 'Numeric_Model',
            'AUROC': 'NUM_AUROC',
            'AUROC_CI_Lower': 'NUM_CI_L',
            'AUROC_CI_Upper': 'NUM_CI_U',
            'y_true': 'NUM_y_true',
            'y_pred_proba': 'NUM_y_pred'
        }).set_index('Task')

        merged = champion_df.join(best_numeric, on='Task')

        # 3. Calculate p-values and format columns based on superiority
        semantic_formatted = []
        numeric_formatted = []
        p_val_strings = []
        
        for task_idx, row in merged.iterrows():
            yt = row.get('y_true')
            yp_base = row.get('NUM_y_pred')
            yp_champ = row.get('y_pred_proba')
            
            # Calculate p-value
            p = calculate_delong_pvalue(yt, yp_base, yp_champ)
            
            # Base Strings
            sem_val = row['AUROC']
            sem_str = _fmt_ci(sem_val, row['AUROC_CI_Lower'], row['AUROC_CI_Upper'])
            
            num_val = row['NUM_AUROC']
            num_str = _fmt_ci(num_val, row['NUM_CI_L'], row['NUM_CI_U']) if pd.notna(num_val) else ''
            
            # Apply formatting if significant
            if pd.notnull(p) and p < 0.05 and pd.notna(num_val):
                # Determine stars
                if p < 0.001: stars = "***"
                elif p < 0.01: stars = "**"
                else: stars = "*"
                
                if num_val > sem_val:
                    # Numeric wins
                    num_str = f"\\textbf{{{num_str}}}{stars}"
                else:
                    # Semantic wins
                    sem_str = f"\\textbf{{{sem_str}}}{stars}"
            
            semantic_formatted.append(sem_str)
            numeric_formatted.append(num_str)
            if pd.notnull(p):
                if p < 0.001:
                    p_val_strings.append("<0.001")
                else:
                    p_val_strings.append(f"{p:.3f}")
            else:
                p_val_strings.append("")

        table_champion = pd.DataFrame({
            'Task': merged['TaskLabel'],
            'Numeric Baseline AUROC (95% CI)': numeric_formatted,
            'Champion Semantic': merged['Champion'],
            'Champion Semantic AUROC (95% CI)': semantic_formatted,
            'p-value': p_val_strings
        })
        # Sanitize symbols for LaTeX rendering
        table_champion['Task'] = table_champion['Task'].astype(str).str.replace('>', '\\textgreater{}', regex=False)
        table_champion['Champion Semantic'] = table_champion['Champion Semantic'].astype(str).str.replace('_', '\\_', regex=False)
        
        # Rename for LaTeX
        table_champion = table_champion.rename(columns={
            'Numeric Baseline AUROC (95% CI)': 'Numeric Baseline AUROC (95\\% CI)',
            'Champion Semantic AUROC (95% CI)': 'Champion Semantic AUROC (95\\% CI)'
        }).sort_values('Task')

        latex_champion = table_champion.to_latex(index=False, escape=False)
        out_champ = OUTPUT_DIR / 'table_champion_semantic_vs_numeric.tex'
        out_champ.write_text(latex_champion, encoding='utf-8')
        # Standalone wrapper for TeXworks
        latex_champion_standalone = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{geometry}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{adjustbox}\n"
            "\\geometry{margin=1in}\n"
            "\\begin{document}\n"
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\begin{adjustbox}{width=\\textwidth}\n"
            + latex_champion +
            "\n\\end{adjustbox}\n"
            "\\end{table}\n"
            "\\end{document}\n"
        )
        (OUTPUT_DIR / 'table_champion_semantic_vs_numeric_standalone.tex').write_text(latex_champion_standalone, encoding='utf-8')
        logging.info(f"LaTeX table saved: {out_champ} and standalone version")
    except Exception as e:
        logging.warning(f"Failed to generate champion vs baseline LaTeX table: {e}")

    # Best numerical baseline per task (XGBoost vs ElasticNet)
    try:
        numeric = summary_df[summary_df['Representation'] == 'Baseline'].copy()
        best_num_idx = numeric.groupby('Task', observed=True)['AUROC'].idxmax()
        best_num = numeric.loc[best_num_idx].copy()
        best_num['TaskLabel'] = best_num['Task'].map(TASK_LABELS)
        best_num['Best Numeric Model'] = best_num['Prompt']
        table_numeric = pd.DataFrame({
            'Task': best_num['TaskLabel'],
            'Best Numeric Model': best_num['Best Numeric Model'],
            'AUROC (95% CI)': [_fmt_ci(best_num['AUROC'].iloc[i], best_num['AUROC_CI_Lower'].iloc[i], best_num['AUROC_CI_Upper'].iloc[i]) for i in range(len(best_num))]
        })
        # Sanitize symbols for LaTeX rendering
        table_numeric['Task'] = table_numeric['Task'].astype(str).str.replace('>', '\\textgreater{}', regex=False)
        table_numeric['Best Numeric Model'] = table_numeric['Best Numeric Model'].astype(str).str.replace('_', '\\_', regex=False)
        table_numeric = table_numeric.rename(columns={
            'AUROC (95% CI)': 'AUROC (95\\% CI)'
        }).sort_values('Task')

        latex_numeric = table_numeric.to_latex(index=False, escape=False)
        out_num = OUTPUT_DIR / 'table_best_numeric_baseline.tex'
        out_num.write_text(latex_numeric, encoding='utf-8')
        # Standalone wrapper for TeXworks
        latex_numeric_standalone = (
            "\\documentclass[border=0pt]{standalone}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{adjustbox}\n"
            "\\begin{document}\n"
            "\\begin{adjustbox}{width=\\textwidth}\n"
            + latex_numeric +
            "\n\\end{adjustbox}\n"
            "\\end{document}\n"
        )
        (OUTPUT_DIR / 'table_best_numeric_baseline_standalone.tex').write_text(latex_numeric_standalone, encoding='utf-8')
        logging.info(f"LaTeX table saved: {out_num} and standalone version")
    except Exception as e:
        logging.warning(f"Failed to generate best numeric baseline LaTeX table: {e}")
    
    logging.info("All manuscript figures and tables have been generated successfully.")

if __name__ == "__main__":
    main()