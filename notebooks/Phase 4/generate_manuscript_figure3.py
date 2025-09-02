# generate_manuscript_figures_corrected.py
"""
A streamlined script to analyze experimental results and generate a focused set of
publication-quality figures and tables for a manuscript.

Professional version with improved graphic design, typography, and PDF output.
Optimized for academic manuscript publication with high-resolution figures.
"""
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Configure matplotlib for professional output
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
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'axes.axisbelow': True
})

# Professional color palettes
PROFESSIONAL_COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Professional magenta
    'accent': '#F18F01',       # Professional orange
    'success': '#C73E1D',      # Professional red
    'neutral': '#6C757D',      # Professional gray
    'light': '#F8F9FA',        # Light background
    'dark': '#212529'          # Dark text
}

# Academic color palette for representations
REP_COLORS = {
    'F1 (Uninterpreted)': '#1f77b4',     # Blue
    'F2 (Interpreted)': '#ff7f0e',       # Orange  
    'F3 (Narrative Summary)': '#2ca02c',  # Green
    'Baseline (Numeric)': '#d62728'       # Red
}

# --- Configuration ---

# Directories for input results and output figures.
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
    """Save a matplotlib figure as high-resolution PDF to the output directory."""
    save_dir = OUTPUT_DIR / sub_dir if sub_dir else OUTPUT_DIR
    save_dir.mkdir(exist_ok=True)
    
    # Change extension to .pdf
    if filename.endswith('.png'):
        filename = filename.replace('.png', '.pdf')
    elif not filename.endswith('.pdf'):
        filename += '.pdf'
    
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf', 
                facecolor='white', edgecolor='none')
    logging.info(f"Figure saved to: {save_path}")
    plt.close(fig)

def set_professional_style():
    """Apply professional styling to plots."""
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.color': '#E5E5E5',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

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


# --- Professional Plotting Functions ---

def generate_representation_barplot(df: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    """
    Generate a professional bar plot comparing representations for each prompt, including baselines.
    """
    logging.info(f"Generating {metric} representation bar plot (professional): {title}")
    
    set_professional_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    df_plot = df.copy()
    df_plot['RepresentationLabel'] = df_plot['Representation'].map(REP_LABELS)
    df_plot['PromptLabel'] = df_plot['Prompt'].map(PROMPT_LABELS)
    
    # Ensure labels are categorical for consistent ordering
    df_plot['RepresentationLabel'] = pd.Categorical(df_plot['RepresentationLabel'], 
                                                   categories=[REP_LABELS[r] for r in REPRESENTATIONS], ordered=True)
    df_plot['PromptLabel'] = pd.Categorical(df_plot['PromptLabel'], 
                                           categories=[PROMPT_LABELS[p] for p in PROMPTS if p in PROMPT_LABELS], ordered=True)
    df_plot = df_plot.dropna(subset=['PromptLabel', 'RepresentationLabel'])

    # Professional bar plot
    bplot = sns.barplot(
        data=df_plot, x='PromptLabel', y=metric, hue='RepresentationLabel',
        palette=REP_COLORS, ax=ax, saturation=0.8, edgecolor='white', linewidth=0.5
    )

    # Professional error bars
    hue_order = [h.get_text() for h in ax.get_legend().get_texts()]
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    num_hues = len(hue_order)
    bar_width = 0.8 / num_hues if num_hues > 0 else 0.8
    x_pos_map = {label: i for i, label in enumerate(x_labels)}

    # Add professional error bars
    for _, row in df_plot.iterrows():
        prompt_label = row['PromptLabel']
        rep_label = row['RepresentationLabel']
        
        if prompt_label in x_pos_map and rep_label in hue_order:
            x_pos_group = x_pos_map[prompt_label]
            hue_index = hue_order.index(rep_label)
            
            bar_offset = (hue_index - (num_hues - 1) / 2) * bar_width
            x_coord = x_pos_group + bar_offset
            
            y_val = row[metric]
            ci_lower = row[f'{metric}_CI_Lower']
            ci_upper = row[f'{metric}_CI_Upper']
            
            if pd.notna(ci_lower) and pd.notna(ci_upper):
                y_err = [[y_val - ci_lower], [ci_upper - y_val]]
                ax.errorbar(x=x_coord, y=y_val, yerr=y_err, fmt='none', 
                           c='black', capsize=2, capthick=1, elinewidth=1, zorder=10)

    # Professional styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Prompting Strategy / Baseline Model', fontsize=12, fontweight='medium')
    ax.set_ylabel(f'Test Set {metric} (95% CI)', fontsize=12, fontweight='medium')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Professional legend
    legend = ax.legend(title='Representation Type', frameon=True, fancybox=True, 
                      shadow=True, ncol=1, loc='upper left', bbox_to_anchor=(1.02, 1))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Set appropriate y-axis limits
    if df_plot[metric].min() > 0.4:
        ax.set_ylim(0.5, min(1.0, df_plot[f'{metric}_CI_Upper'].max() * 1.15))

    plt.tight_layout()
    save_figure(fig, filename)
    
def generate_performance_lift_plot(df: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    """
    Generate a professional plot showing performance lift over the best baseline model.
    """
    logging.info(f"Generating {metric} performance lift plot (professional): {title}")

    baseline_scores = df[df['Representation'] == 'Baseline'][metric]
    if baseline_scores.empty:
        logging.warning(f"No baseline scores for {metric}, cannot generate lift plot.")
        return
    best_baseline_score = baseline_scores.max()

    set_professional_style()
    
    df_lift = df.copy()
    lift_metric = f'{metric}_Lift'
    df_lift[lift_metric] = df_lift[metric] - best_baseline_score
    df_lift['Arm'] = df_lift.apply(
        lambda r: f"{REP_LABELS.get(r['Representation'], r['Representation'])} - {PROMPT_LABELS.get(r['Prompt'], r['Prompt'])}", 
        axis=1
    )
    df_lift = df_lift.sort_values(lift_metric, ascending=True)  # Sort ascending for horizontal bars
    
    lower_err = df_lift[lift_metric] - (df_lift[f'{metric}_CI_Lower'] - best_baseline_score)
    upper_err = (df_lift[f'{metric}_CI_Upper'] - best_baseline_score) - df_lift[lift_metric]
    errors = [lower_err, upper_err] if lower_err.notna().all() and upper_err.notna().all() else None

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color mapping based on performance
    colors = ['#d62728' if x < 0 else PROFESSIONAL_COLORS['primary'] for x in df_lift[lift_metric]]
    
    bars = ax.barh(np.arange(len(df_lift)), df_lift[lift_metric], 
                   color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    if errors:
        ax.errorbar(x=df_lift[lift_metric], y=np.arange(len(df_lift)), xerr=errors,
                    fmt='none', ecolor='black', capsize=2, capthick=1, elinewidth=1, zorder=10)

    # Professional reference line
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7, zorder=5)
    
    # Professional styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'Change in {metric} vs. Best Baseline (95% CI)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Experimental Arm', fontsize=12, fontweight='medium')
    
    # Set y-tick labels
    ax.set_yticks(np.arange(len(df_lift)))
    ax.set_yticklabels(df_lift['Arm'], fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, filename)

def generate_model_comparison_plot(df: pd.DataFrame, baselines_df: pd.DataFrame, metric: str, filename: str, title: str):
    """
    Generate a faceted bar plot comparing model performance for a task.
    Includes baseline performance as a reference.
    """
    logging.info(f"Generating detailed model comparison plot for {metric}: {title}")
    
    plot_df = df.copy()
    if plot_df.empty:
        logging.warning("No embedding model data available for the detailed comparison plot.")
        return

    plot_df['RepresentationLabel'] = plot_df['Representation'].map(REP_LABELS)
    plot_df['PromptLabel'] = plot_df['Prompt'].map(PROMPT_LABELS)

    rep_cats = [REP_LABELS[r] for r in REPRESENTATIONS if r in plot_df['Representation'].unique()]
    prompt_cats = [PROMPT_LABELS[p] for p in PROMPTS if p in plot_df['Prompt'].unique()]
    model_names = sorted(plot_df['Model'].unique())
    
    plot_df['RepresentationLabel'] = pd.Categorical(plot_df['RepresentationLabel'], categories=rep_cats, ordered=True)
    plot_df['PromptLabel'] = pd.Categorical(plot_df['PromptLabel'], categories=prompt_cats, ordered=True)
    plot_df = plot_df.dropna(subset=['PromptLabel', 'RepresentationLabel'])

    palette = sns.color_palette('cubehelix', n_colors=len(model_names))
    color_map = dict(zip(model_names, palette))

    g = sns.catplot(
        data=plot_df, x='PromptLabel', y=metric, hue='Model', col='RepresentationLabel',
        kind='bar', palette=color_map, height=6, aspect=1.2, legend=False,
        col_wrap=3
    )

    baseline_models = {'XGBoost': ('blue', '--'), 'ElasticNet': ('green', ':')}
    
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

        # Add baseline horizontal lines with CIs
        for model_name, (color, style) in baseline_models.items():
            baseline_data = baselines_df[baselines_df['Prompt'] == model_name]
            if not baseline_data.empty:
                score = baseline_data[metric].iloc[0]
                ci_low = baseline_data[f'{metric}_CI_Lower'].iloc[0]
                ci_high = baseline_data[f'{metric}_CI_Upper'].iloc[0]
                ax.axhline(y=score, color=color, linestyle=style, zorder=8)
                ax.axhspan(ci_low, ci_high, color=color, alpha=0.1, zorder=7)

        if ax_df[metric].min() > 0.4:
            ax.set_ylim(0.5, max(1.0, ax_df[f'{metric}_CI_Upper'].max() * 1.05))

    g.fig.suptitle(title, fontsize=18, y=1.03)
    g.set_axis_labels("Prompting Strategy", f'Test Set {metric} (with 95% CI)')
    g.set_titles("Representation: {col_name}")
    g.set_xticklabels(rotation=45, ha='right')

    legend_handles = [mpatches.Patch(color=color_map[name], label=name) for name in model_names]
    for model_name, (color, style) in baseline_models.items():
        if not baselines_df[baselines_df['Prompt'] == model_name].empty:
            legend_handles.append(Line2D([0], [0], color=color, linestyle=style, label=f'{model_name} Baseline (95% CI shaded)'))
            
    g.fig.legend(handles=legend_handles, title="Model", bbox_to_anchor=(1.0, 0.5), loc="center left")
    g.fig.tight_layout(rect=[0, 0, 0.9, 1])

    save_figure(g.fig, filename)

def generate_task_comparison_plot(df: pd.DataFrame, metric: str, filename: str, title: str):
    """Generate a professional bar plot comparing representation performance across all prediction tasks."""
    logging.info(f"Generating {metric} professional comparison plot across tasks: {title}")
    
    set_professional_style()
    
    best_indices = df.groupby(['Task', 'Representation'], observed=True)[metric].idxmax()
    plot_df = df.loc[best_indices].copy()
    
    plot_df['TaskLabel'] = plot_df['Task'].map(TASK_LABELS)
    plot_df['RepresentationLabel'] = plot_df['Representation'].map(REP_LABELS)

    plot_df['TaskLabel'] = pd.Categorical(plot_df['TaskLabel'], categories=TASK_LABELS.values(), ordered=True)
    plot_df['RepresentationLabel'] = pd.Categorical(plot_df['RepresentationLabel'], categories=REP_LABELS.values(), ordered=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bplot = sns.barplot(data=plot_df, x='TaskLabel', y=metric, hue='RepresentationLabel', 
                       palette=REP_COLORS, ax=ax, saturation=0.8, edgecolor='white', linewidth=0.5)
    
    # Professional error bars
    hue_order = [x.get_text() for x in ax.get_legend().get_texts()]
    num_hues = len(hue_order)
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_pos_map = {label: i for i, label in enumerate(x_tick_labels)}
    bar_width = 0.8 / num_hues if num_hues > 0 else 0.8
    
    for _, row in plot_df.iterrows():
        task_label = row['TaskLabel']
        rep_label = row['RepresentationLabel']
        
        if task_label in x_pos_map and rep_label in hue_order:
            x_pos_group = x_pos_map[task_label]
            hue_index = hue_order.index(rep_label)
            
            x_coord = x_pos_group + (hue_index - (num_hues - 1) / 2) * bar_width
            
            y_val = row[metric]
            y_err = [[y_val - row[f'{metric}_CI_Lower']], [row[f'{metric}_CI_Upper'] - y_val]]
            
            ax.errorbar(x=x_coord, y=y_val, yerr=y_err, fmt='none', 
                       c='black', capsize=2, capthick=1, elinewidth=1, zorder=10)

    # Professional styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Prediction Task', fontsize=12, fontweight='medium')
    ax.set_ylabel(f'Test Set {metric} (95% CI)', fontsize=12, fontweight='medium')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Professional legend
    legend = ax.legend(title='Data Representation', frameon=True, fancybox=True, 
                      shadow=True, loc='upper left', bbox_to_anchor=(1.02, 1))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    save_figure(fig, filename)
    
def generate_champion_model_plot(df: pd.DataFrame, metric: str, filename: str, title: str):
    """
<<<<<<< HEAD
    Generate a plot showing the best performing semantic model for each task vs. baseline.
    Now includes a confidence interval for the baseline model marker and shows champion model details.
=======
    Generate a professional plot showing the best performing semantic model for each task vs. baseline.
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
    """
    logging.info(f"Generating professional champion model plot for {metric}: {title}")
    
    set_professional_style()
    
    semantic_df = df[df['Representation'] != 'Baseline']
    if semantic_df.empty:
        logging.warning("No semantic models found for champion plot.")
        return
        
    best_indices = semantic_df.groupby('Task', observed=True)[metric].idxmax()
    champion_df = semantic_df.loc[best_indices].copy()

    champion_df['TaskLabel'] = champion_df['Task'].map(TASK_LABELS)
    champion_df['ChampionShort'] = champion_df.apply(
        lambda r: f"{r['Representation']}-{r['Prompt']} ({r['Model'].split('_')[-1] if '_' in r['Model'] else r['Model']})", axis=1
    )
    
    # Create shorter labels for annotations
    champion_df['ChampionShort'] = champion_df.apply(
        lambda r: f"{r['Representation']}-{r['Prompt']} ({r['Model'].split('_')[-1] if '_' in r['Model'] else r['Model']})", axis=1
    )

    baseline_cols = ['Task', metric, f'{metric}_CI_Lower', f'{metric}_CI_Upper']
    baseline_data = df[df['Prompt'] == 'XGBoost'][baseline_cols].set_index('Task')
    champion_df = champion_df.join(baseline_data.rename(columns={
        metric: 'Baseline_Metric',
        f'{metric}_CI_Lower': 'Baseline_CI_Lower',
        f'{metric}_CI_Upper': 'Baseline_CI_Upper'
    }), on='Task')

<<<<<<< HEAD
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.set_theme(style="whitegrid")
    
    # Use a single color for all bars since we'll annotate with champion details
    bars = sns.barplot(data=champion_df, x=metric, y='TaskLabel', 
                       palette=['skyblue'], ax=ax)

    x_err = [champion_df[metric] - champion_df[f'{metric}_CI_Lower'], champion_df[f'{metric}_CI_Upper'] - champion_df[metric]]
    ax.errorbar(x=champion_df[metric], y=np.arange(len(champion_df)), xerr=x_err,
                fmt='none', ecolor='black', capsize=3)
    
    # Add text annotations showing champion model details
    x_min, x_max = ax.get_xlim()
    text_x_pos = x_min + (x_max - x_min) * 0.05  # Position at 5% from left edge
=======
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Professional horizontal bar plot
    bars = ax.barh(np.arange(len(champion_df)), champion_df[metric], 
                   color=PROFESSIONAL_COLORS['primary'], alpha=0.8, 
                   edgecolor='white', linewidth=0.8)

    # Professional error bars
    x_err = [champion_df[metric] - champion_df[f'{metric}_CI_Lower'], 
             champion_df[f'{metric}_CI_Upper'] - champion_df[metric]]
    ax.errorbar(x=champion_df[metric], y=np.arange(len(champion_df)), xerr=x_err,
                fmt='none', ecolor='black', capsize=3, capthick=1, elinewidth=1, zorder=10)
    
    # Professional annotations
    x_min, x_max = ax.get_xlim()
    text_x_pos = x_min + (x_max - x_min) * 0.02
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
    
    for i, (_, row) in enumerate(champion_df.iterrows()):
        champion_text = row['ChampionShort']
        ax.text(text_x_pos, i, champion_text, va='center', ha='left', fontsize=9, 
<<<<<<< HEAD
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                zorder=10)
    
    y_coords_map = {label.get_text(): i for i, label in enumerate(ax.get_yticklabels())}
    for _, row in champion_df.iterrows():
        if pd.notna(row['Baseline_Metric']) and row['TaskLabel'] in y_coords_map:
            y_coord = y_coords_map[row['TaskLabel']]
            
=======
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.9), zorder=10)
    
    # Professional baseline markers
    for i, (_, row) in enumerate(champion_df.iterrows()):
        if pd.notna(row['Baseline_Metric']):
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
            lower_err = row['Baseline_Metric'] - row['Baseline_CI_Lower']
            upper_err = row['Baseline_CI_Upper'] - row['Baseline_Metric']
            ax.errorbar(x=[row['Baseline_Metric']], y=[i], xerr=[[lower_err], [upper_err]],
                        fmt='D', color=PROFESSIONAL_COLORS['success'], markersize=6, 
                        capsize=4, capthick=1, elinewidth=1, zorder=12, alpha=0.9)

<<<<<<< HEAD
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(f'Best Test Set {metric} (with 95% CI)', fontsize=12)
    ax.set_ylabel('Prediction Task', fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker='D', color='w', label='XGBoost Baseline (with 95% CI)',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], color='skyblue', linewidth=6, label='Champion Semantic Model')
    ]
    ax.legend(handles=legend_elements, title="Model Type", loc='lower right')
=======
    # Professional styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'Test Set {metric} (95% CI)', fontsize=12, fontweight='medium')
    ax.set_ylabel('Prediction Task', fontsize=12, fontweight='medium')
    
    ax.set_yticks(np.arange(len(champion_df)))
    ax.set_yticklabels(champion_df['TaskLabel'], fontsize=10)

    # Professional legend
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', label='XGBoost Baseline (95% CI)',
               markerfacecolor=PROFESSIONAL_COLORS['success'], markersize=8),
        Line2D([0], [0], color=PROFESSIONAL_COLORS['primary'], linewidth=6, 
               label='Champion Semantic Model', alpha=0.8)
    ]
    legend = ax.legend(handles=legend_elements, title="Model Type", loc='lower right',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130

    plt.tight_layout()
    save_figure(fig, filename)

def generate_task_comparison_by_model_plot(df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str, filename: str, title: str):
    """
    Generate a professional faceted bar plot comparing task performance for each embedding model.
    """
    logging.info(f"Generating professional per-model task comparison plot for {metric}: {title}")
    
    set_professional_style()
    
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

    # Professional color palette
    palette = sns.color_palette('viridis', n_colors=len(rep_cats))
    color_map = dict(zip(rep_cats, palette))

    g = sns.catplot(
        data=plot_df, x='TaskLabel', y=metric, hue='RepresentationLabel', col='Model',
        kind='bar', palette=color_map, height=5, aspect=1.5, legend=False,
        col_wrap=2, sharey=False, col_order=model_names,
        hue_order=rep_cats, saturation=0.8
    )

    # Professional baseline styles
    baseline_styles = {
        'XGBoost': {'marker': 'D', 'color': PROFESSIONAL_COLORS['success'], 'label': 'XGBoost Baseline'},
        'ElasticNet': {'marker': 's', 'color': PROFESSIONAL_COLORS['secondary'], 'label': 'ElasticNet Baseline'}
    }
    task_key_map = {v: k for k, v in TASK_LABELS.items()}

    for model_name, ax in zip(g.col_names, g.axes.flat):
        if not ax.patches: continue
        
        ax_df = plot_df[plot_df['Model'] == model_name]
        if ax_df.empty: continue

        hue_order = rep_cats
        num_hues = len(hue_order)
        bar_width = 0.8 / num_hues if num_hues > 0 else 0.8
        x_labels = task_cats
        x_pos_map = {label: i for i, label in enumerate(x_labels)}

        # Professional error bars for embedding models
        for _, row in ax_df.iterrows():
            task_label, rep_label = row['TaskLabel'], row['RepresentationLabel']
            if task_label in x_pos_map and rep_label in hue_order:
                x_pos_group = x_pos_map[task_label]
                hue_index = hue_order.index(rep_label)
                bar_offset = (hue_index - (num_hues - 1) / 2) * bar_width
                x_coord = x_pos_group + bar_offset
                
                y_val = row[metric]
                y_err = [[y_val - row[f'{metric}_CI_Lower']], [row[f'{metric}_CI_Upper'] - y_val]]
                ax.errorbar(x=x_coord, y=y_val, yerr=y_err, fmt='none', 
                           c='black', capsize=2, capthick=1, elinewidth=1, zorder=10)

        # Professional baseline markers
        for task_label, x_pos in x_pos_map.items():
            task_key = task_key_map.get(task_label)
            if not task_key: continue
            
            task_baselines = baseline_df[baseline_df['Task'] == task_key]
            if task_baselines.empty: continue
            
            for b_model_name, style in baseline_styles.items():
                baseline_data = task_baselines[task_baselines['Prompt'] == b_model_name]
                if not baseline_data.empty:
                    row = baseline_data.iloc[0]
                    y_val = row[metric]
                    y_err_data = [[y_val - row[f'{metric}_CI_Lower']], [row[f'{metric}_CI_Upper'] - y_val]]
                    ax.errorbar(x=x_pos, y=y_val, yerr=y_err_data, fmt=style['marker'],
                                color=style['color'], markersize=6, capsize=3,
                                elinewidth=1.2, zorder=12, markeredgewidth=1,
                                markerfacecolor='none', alpha=0.9)

        if not ax_df.empty and ax_df[metric].min() > 0.4:
            y_max_ci = ax_df[f'{metric}_CI_Upper'].max()
            ax.set_ylim(0.5, 1)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    g.fig.suptitle(title, fontsize=16, fontweight='bold', y=1.03)
    g.set_axis_labels("Prediction Task", f'Test Set {metric} (95% CI)')
    g.set_titles("Model: {col_name}")
    
    # Professional combined legend
    rep_legend_handles = [mpatches.Patch(color=color_map[name], label=name) for name in rep_cats]
    legend1 = g.fig.legend(handles=rep_legend_handles, title="Representation",
                          bbox_to_anchor=(1.02, 0.6), loc="center left", 
                          frameon=True, fancybox=True, shadow=True)

    baseline_legend_handles = [
        Line2D([0], [0], marker=style['marker'], color='w', label=style['label'],
               markerfacecolor='none', markeredgecolor=style['color'], 
               markeredgewidth=1.5, markersize=7)
        for _, style in baseline_styles.items()
    ]
    g.fig.add_artist(legend1)
    g.fig.legend(handles=baseline_legend_handles, title="Baselines",
                bbox_to_anchor=(1.02, 0.35), loc="center left", 
                frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    save_figure(g.fig, filename)

# --- Table Generation ---

def generate_results_table(df: pd.DataFrame, filename: str) -> None:
    """Generate and save a formatted Markdown table of all results."""
    logging.info("Generating professional Markdown results table...")
    
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
    """Main function to run the analysis and generate professional outputs."""
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

    # --- Generate Professional Per-Task Plots ---
    logging.info("Generating professional per-task performance plots...")
    tasks = summary_df['Task'].unique()
    for task in tasks:
        task_df_summary = summary_df[summary_df['Task'] == task]
        task_label = TASK_LABELS.get(task, task)
        
        generate_representation_barplot(
            task_df_summary, 'AUROC', f'figure_2_auroc_representation_barplot_{task}.pdf',
            f'Figure 2: AUROC by Representation for {task_label}'
        )
        
        generate_performance_lift_plot(
            task_df_summary, 'AUROC', f'figure_3_auroc_lift_{task}.pdf',
            f'Figure 3: AUROC Improvement Over Best Baseline for {task_label}'
        )

        # Professional detailed model comparison
        task_df_full_embeddings = embedding_df_full[embedding_df_full['Task'] == task]
        task_df_baselines = baseline_df_full[baseline_df_full['Task'] == task]
        generate_model_comparison_plot(
            task_df_full_embeddings, task_df_baselines, 'AUROC', f'figure_6_auroc_model_comparison_{task}.pdf',
            f'Figure 6: Detailed Model Comparison for {task_label} with Baselines'
        )

    # --- Generate Professional Cross-Task Summary Plots ---
    logging.info("Generating professional cross-task summary plots...")
    generate_task_comparison_plot(
        summary_df, 'AUROC', 'figure_4_task_comparison_auroc.pdf',
        'Figure 4: Best Model AUROC by Representation and Task'
    )
    generate_champion_model_plot(
        summary_df, 'AUROC', 'figure_5_champion_models_auroc.pdf',
        'Figure 5: Champion Semantic Model vs. XGBoost Baseline by AUROC'
    )
    
    # --- Generate Professional Per-Model Task Comparison ---
    generate_task_comparison_by_model_plot(
        embedding_df_full, 
        baseline_df_full,
        'AUROC', 
        'figure_7_task_comparison_by_model_auroc_with_baselines.pdf',
        'Figure 7: Comparing Task Performance for each Embedding Model with Baselines'
    )
    
    # --- Generate Professional Table ---
    generate_results_table(full_df, 'table_1_full_results.md')
    
    logging.info("All professional manuscript figures and tables have been generated successfully.")

if __name__ == "__main__":
    main()