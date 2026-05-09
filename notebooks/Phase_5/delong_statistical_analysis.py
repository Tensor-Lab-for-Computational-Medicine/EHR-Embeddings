"""
DeLong's Test for Statistical Significance of AUROC Differences

This script performs DeLong's test to compare AUROC scores across:
- Formats (F1, F2, F3)
- Embedding models (text-embedding-004, text-embedding-005, etc.)
- Prompts (P0, P1, P2, P3, P4, P5)

Usage:
    python delong_statistical_analysis.py

Output:
    - Pairwise comparison tables with significance markers
    - CSV files with detailed p-values
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DeLong's Test Implementation
# =============================================================================

def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong's test.
    
    Args:
        x: 1D array of values
        
    Returns:
        Midranks array
    """
    n = len(x)
    sorted_idx = np.argsort(x)
    ranks = np.empty(n, dtype=float)
    
    i = 0
    while i < n:
        j = i
        # Find ties
        while j < n - 1 and x[sorted_idx[j]] == x[sorted_idx[j + 1]]:
            j += 1
        # Average rank for ties
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    
    return ranks


def fast_delong_variance(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute AUC and its variance using the fast DeLong algorithm.
    
    Args:
        predictions: Array of shape (n_classifiers, n_samples) with predicted probabilities
        labels: 1D array of binary labels (0 or 1)
        
    Returns:
        Tuple of (AUC array, covariance matrix)
    """
    # Separate positive and negative samples
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    m = pos_mask.sum()  # Number of positives
    n = neg_mask.sum()  # Number of negatives
    
    if m == 0 or n == 0:
        raise ValueError("Both positive and negative samples are required")
    
    k = predictions.shape[0]  # Number of classifiers
    
    # Compute AUCs using the Mann-Whitney U statistic
    aucs = np.zeros(k)
    for idx in range(k):
        pred = predictions[idx]
        pos_pred = pred[pos_mask]
        neg_pred = pred[neg_mask]
        aucs[idx] = np.mean(np.add.outer(pos_pred, -neg_pred) > 0) + \
                    0.5 * np.mean(np.add.outer(pos_pred, -neg_pred) == 0)
    
    # Compute structural components for variance estimation
    # V10: variance component from positive samples
    # V01: variance component from negative samples
    
    V10 = np.zeros((k, m))  # Shape (k, m)
    V01 = np.zeros((k, n))  # Shape (k, n)
    
    for idx in range(k):
        pred = predictions[idx]
        pos_pred = pred[pos_mask]
        neg_pred = pred[neg_mask]
        
        # Proportion of negatives less than each positive
        for i, p in enumerate(pos_pred):
            V10[idx, i] = np.mean(neg_pred < p) + 0.5 * np.mean(neg_pred == p)
        
        # Proportion of positives greater than each negative
        for j, q in enumerate(neg_pred):
            V01[idx, j] = np.mean(pos_pred > q) + 0.5 * np.mean(pos_pred == q)
    
    # Compute covariance matrix
    S10 = np.cov(V10) if k > 1 else np.var(V10, ddof=1).reshape(1, 1)
    S01 = np.cov(V01) if k > 1 else np.var(V01, ddof=1).reshape(1, 1)
    
    # Total covariance
    S = S10 / m + S01 / n
    
    return aucs, S


def delong_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple[float, float, float, float]:
    """Perform DeLong's test comparing two AUROC values.
    
    Args:
        y_true: Ground truth binary labels
        y_pred1: Predicted probabilities from model 1
        y_pred2: Predicted probabilities from model 2
        
    Returns:
        Tuple of (auc1, auc2, z_statistic, p_value)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred1 = np.asarray(y_pred1).ravel()
    y_pred2 = np.asarray(y_pred2).ravel()
    
    # Stack predictions
    predictions = np.vstack([y_pred1, y_pred2])
    
    # Compute AUCs and covariance
    aucs, cov_matrix = fast_delong_variance(predictions, y_true)
    
    auc1, auc2 = aucs[0], aucs[1]
    
    # Variance of the difference
    var_diff = cov_matrix[0, 0] + cov_matrix[1, 1] - 2 * cov_matrix[0, 1]
    
    if var_diff <= 0:
        # Degenerate case
        return auc1, auc2, 0.0, 1.0
    
    # Z-statistic
    z = (auc1 - auc2) / np.sqrt(var_diff)
    
    # Two-tailed p-value
    p_value = 2 * stats.norm.sf(abs(z))
    
    return auc1, auc2, z, p_value


def significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


# =============================================================================
# Data Loading
# =============================================================================

def load_results_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load a single results pickle file."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {filepath}: {e}")
        return None


def discover_embedding_models(base_dir: Path) -> List[str]:
    """Discover all embedding model directories."""
    models = []
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.endswith('.zip'):
            models.append(item.name)
    return sorted(models)


def discover_tasks(model_dir: Path) -> List[str]:
    """Discover all task directories for an embedding model."""
    tasks = []
    for item in model_dir.iterdir():
        if item.is_dir():
            # Check if it contains results files
            results_files = list(item.glob('results_*.pkl'))
            if results_files:
                tasks.append(item.name)
    return sorted(tasks)


def load_all_results(base_dir: Path) -> pd.DataFrame:
    """Load all results into a unified DataFrame.
    
    Returns DataFrame with columns:
        - embedding_model, task, format, prompt, auroc, y_true, y_pred_proba
    """
    records = []
    
    embedding_models = discover_embedding_models(base_dir)
    print(f"Found {len(embedding_models)} embedding models: {embedding_models}")
    
    for model_name in embedding_models:
        model_dir = base_dir / model_name
        tasks = discover_tasks(model_dir)
        
        if not tasks:
            print(f"  Skipping {model_name}: no completed tasks found")
            continue
            
        print(f"\n  Loading {model_name} with tasks: {tasks}")
        
        for task in tasks:
            task_dir = model_dir / task
            results_files = list(task_dir.glob('results_F*.pkl'))
            
            for rf in results_files:
                # Parse format and prompt from filename: results_F{format}_P{prompt}.pkl
                name = rf.stem  # e.g., "results_F1_P0"
                parts = name.replace('results_', '').split('_')
                if len(parts) >= 2:
                    fmt = parts[0]  # F1, F2, F3
                    prompt = parts[1]  # P0, P1, etc.
                else:
                    continue
                
                result = load_results_file(rf)
                if result is None:
                    continue
                
                # Extract predictions
                y_true = result.get('y_true')
                y_pred = result.get('y_pred_proba')
                
                # Try alternate locations
                if y_true is None and 'full_evaluation' in result:
                    y_true = result['full_evaluation'].get('y_true')
                    y_pred = result['full_evaluation'].get('y_pred_proba')
                
                # Get AUROC
                auroc = None
                if 'auroc' in result:
                    auroc_data = result['auroc']
                    if isinstance(auroc_data, dict):
                        auroc = auroc_data.get('point_estimate')
                    else:
                        auroc = auroc_data
                
                if y_true is not None and y_pred is not None:
                    records.append({
                        'embedding_model': model_name,
                        'task': task,
                        'format': fmt,
                        'prompt': prompt,
                        'auroc': auroc,
                        'y_true': np.asarray(y_true),
                        'y_pred_proba': np.asarray(y_pred),
                        'filepath': str(rf)
                    })
    
    df = pd.DataFrame(records)
    print(f"\nLoaded {len(df)} result records")
    return df


# =============================================================================
# Statistical Comparison Functions
# =============================================================================

def compare_pairwise(df: pd.DataFrame, group_col: str, 
                     fixed_cols: Dict[str, Any] = None) -> pd.DataFrame:
    """Perform pairwise DeLong's tests for a grouping variable.
    
    Args:
        df: DataFrame with results
        group_col: Column to group by for comparison (e.g., 'format', 'prompt', 'embedding_model')
        fixed_cols: Dict of column:value pairs to filter by
        
    Returns:
        DataFrame with pairwise comparison results
    """
    # Apply fixed column filters
    subset = df.copy()
    if fixed_cols:
        for col, val in fixed_cols.items():
            subset = subset[subset[col] == val]
    
    if len(subset) < 2:
        return pd.DataFrame()
    
    # Get unique groups
    groups = sorted(subset[group_col].unique())
    
    comparisons = []
    for g1, g2 in combinations(groups, 2):
        row1 = subset[subset[group_col] == g1]
        row2 = subset[subset[group_col] == g2]
        
        if len(row1) == 0 or len(row2) == 0:
            continue
        
        # Use first matching row (should be unique given fixed_cols)
        r1 = row1.iloc[0]
        r2 = row2.iloc[0]
        
        try:
            auc1, auc2, z, p = delong_test(r1['y_true'], r1['y_pred_proba'], r2['y_pred_proba'])
            comparisons.append({
                'group1': g1,
                'group2': g2,
                'auroc1': auc1,
                'auroc2': auc2,
                'diff': auc1 - auc2,
                'z_statistic': z,
                'p_value': p,
                'significance': significance_marker(p)
            })
        except Exception as e:
            print(f"  Warning: Could not compare {g1} vs {g2}: {e}")
    
    return pd.DataFrame(comparisons)


def format_comparison_table(df: pd.DataFrame, tasks: List[str], 
                           embedding_model: str, prompt: str) -> pd.DataFrame:
    """Create a format comparison table (F1 vs F2 vs F3) for a specific model/prompt."""
    formats = ['F1', 'F2', 'F3']
    results = []
    
    for task in tasks:
        subset = df[(df['task'] == task) & 
                   (df['embedding_model'] == embedding_model) & 
                   (df['prompt'] == prompt)]
        
        if len(subset) < 2:
            continue
        
        row = {'task': task}
        
        # Get AUROCs for each format
        for fmt in formats:
            fmt_row = subset[subset['format'] == fmt]
            if len(fmt_row) > 0:
                row[f'{fmt}_auroc'] = fmt_row.iloc[0]['auroc']
        
        # Pairwise comparisons
        for f1, f2 in combinations(formats, 2):
            r1 = subset[subset['format'] == f1]
            r2 = subset[subset['format'] == f2]
            
            if len(r1) > 0 and len(r2) > 0:
                try:
                    _, _, _, p = delong_test(
                        r1.iloc[0]['y_true'], 
                        r1.iloc[0]['y_pred_proba'], 
                        r2.iloc[0]['y_pred_proba']
                    )
                    row[f'{f1}_vs_{f2}_p'] = p
                    row[f'{f1}_vs_{f2}_sig'] = significance_marker(p)
                except:
                    pass
        
        results.append(row)
    
    return pd.DataFrame(results)


def prompt_comparison_table(df: pd.DataFrame, tasks: List[str],
                           embedding_model: str, fmt: str) -> pd.DataFrame:
    """Create a prompt comparison table (P0-P5) for a specific model/format."""
    prompts = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
    results = []
    
    for task in tasks:
        subset = df[(df['task'] == task) & 
                   (df['embedding_model'] == embedding_model) & 
                   (df['format'] == fmt)]
        
        if len(subset) < 2:
            continue
        
        row = {'task': task}
        
        # Get AUROCs for each prompt
        for prompt in prompts:
            prompt_row = subset[subset['prompt'] == prompt]
            if len(prompt_row) > 0:
                row[f'{prompt}_auroc'] = prompt_row.iloc[0]['auroc']
        
        # Pairwise comparisons
        for p1, p2 in combinations(prompts, 2):
            r1 = subset[subset['prompt'] == p1]
            r2 = subset[subset['prompt'] == p2]
            
            if len(r1) > 0 and len(r2) > 0:
                try:
                    _, _, _, p = delong_test(
                        r1.iloc[0]['y_true'], 
                        r1.iloc[0]['y_pred_proba'], 
                        r2.iloc[0]['y_pred_proba']
                    )
                    row[f'{p1}_vs_{p2}_p'] = p
                    row[f'{p1}_vs_{p2}_sig'] = significance_marker(p)
                except:
                    pass
        
        results.append(row)
    
    return pd.DataFrame(results)


def embedding_model_comparison(df: pd.DataFrame, task: str, 
                               fmt: str, prompt: str) -> pd.DataFrame:
    """Compare all embedding models for a specific task/format/prompt."""
    subset = df[(df['task'] == task) & 
               (df['format'] == fmt) & 
               (df['prompt'] == prompt)]
    
    if len(subset) < 2:
        return pd.DataFrame()
    
    models = sorted(subset['embedding_model'].unique())
    comparisons = []
    
    for m1, m2 in combinations(models, 2):
        r1 = subset[subset['embedding_model'] == m1]
        r2 = subset[subset['embedding_model'] == m2]
        
        if len(r1) > 0 and len(r2) > 0:
            try:
                auc1, auc2, z, p = delong_test(
                    r1.iloc[0]['y_true'],
                    r1.iloc[0]['y_pred_proba'],
                    r2.iloc[0]['y_pred_proba']
                )
                comparisons.append({
                    'model1': m1,
                    'model2': m2,
                    'auroc1': auc1,
                    'auroc2': auc2,
                    'diff': auc1 - auc2,
                    'z_statistic': z,
                    'p_value': p,
                    'significance': significance_marker(p)
                })
            except Exception as e:
                print(f"  Warning: Could not compare {m1} vs {m2}: {e}")
    
    return pd.DataFrame(comparisons)


# =============================================================================
# Champion Model Analysis (From Table 1 in the image)
# =============================================================================

# Task name mappings
TASK_MAPPING = {
    'readmission_30': '30-Day Readmission',
    'mort_hosp': 'Hospital Mortality',
    'los_3': 'Length-of-Stay > 3 Days',
    'los_7': 'Length-of-Stay > 7 Days',
    'intervention_vent': 'Mechanical Ventilation',
    'intervention_vaso': 'Vasopressor Administration'
}

# Champion semantic models from Table 1
CHAMPION_SEMANTIC = {
    'readmission_30': ('F1', 'P0', 'text-embedding-005'),
    'mort_hosp': ('F3', 'P5', 'text-embedding-004'),
    'los_3': ('F3', 'P1', 'text-embedding-004'),
    'los_7': ('F3', 'P2', 'text-embedding-005'),
    'intervention_vent': ('F3', 'P0', 'text-embedding-004'),
    'intervention_vaso': ('F3', 'P2', 'text-embedding-004')
}


def analyze_champion_model(df: pd.DataFrame, task: str, 
                          champion_format: str, champion_prompt: str, 
                          champion_model: str) -> Dict:
    """Analyze statistical significance for a champion model.
    
    Compares the champion against all other:
    - Formats (with same model and prompt)
    - Prompts (with same model and format)
    - Embedding models (with same format and prompt)
    """
    results = {
        'task': task,
        'champion': f"{champion_format}_{champion_prompt} ({champion_model})",
        'format_comparisons': [],
        'prompt_comparisons': [],
        'model_comparisons': []
    }
    
    # Get champion data
    champion_data = df[(df['task'] == task) & 
                       (df['format'] == champion_format) & 
                       (df['prompt'] == champion_prompt) & 
                       (df['embedding_model'] == champion_model)]
    
    if len(champion_data) == 0:
        print(f"  Warning: Champion model not found for {task}")
        return results
    
    champion = champion_data.iloc[0]
    
    # Format comparisons (same model/prompt, different formats)
    formats = ['F1', 'F2', 'F3']
    for fmt in formats:
        if fmt == champion_format:
            continue
        
        other = df[(df['task'] == task) & 
                  (df['format'] == fmt) & 
                  (df['prompt'] == champion_prompt) & 
                  (df['embedding_model'] == champion_model)]
        
        if len(other) > 0:
            other = other.iloc[0]
            try:
                auc1, auc2, z, p = delong_test(
                    champion['y_true'], champion['y_pred_proba'], other['y_pred_proba']
                )
                results['format_comparisons'].append({
                    'comparison': f"{champion_format} vs {fmt}",
                    'champion_auroc': auc1,
                    'other_auroc': auc2,
                    'diff': auc1 - auc2,
                    'p_value': p,
                    'significance': significance_marker(p)
                })
            except:
                pass
    
    # Prompt comparisons (same model/format, different prompts)
    prompts = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
    for prompt in prompts:
        if prompt == champion_prompt:
            continue
        
        other = df[(df['task'] == task) & 
                  (df['format'] == champion_format) & 
                  (df['prompt'] == prompt) & 
                  (df['embedding_model'] == champion_model)]
        
        if len(other) > 0:
            other = other.iloc[0]
            try:
                auc1, auc2, z, p = delong_test(
                    champion['y_true'], champion['y_pred_proba'], other['y_pred_proba']
                )
                results['prompt_comparisons'].append({
                    'comparison': f"{champion_prompt} vs {prompt}",
                    'champion_auroc': auc1,
                    'other_auroc': auc2,
                    'diff': auc1 - auc2,
                    'p_value': p,
                    'significance': significance_marker(p)
                })
            except:
                pass
    
    # Embedding model comparisons (same format/prompt, different models)
    models = df[(df['task'] == task) & 
               (df['format'] == champion_format) & 
               (df['prompt'] == champion_prompt)]['embedding_model'].unique()
    
    for model in models:
        if model == champion_model:
            continue
        
        other = df[(df['task'] == task) & 
                  (df['format'] == champion_format) & 
                  (df['prompt'] == champion_prompt) & 
                  (df['embedding_model'] == model)]
        
        if len(other) > 0:
            other = other.iloc[0]
            try:
                auc1, auc2, z, p = delong_test(
                    champion['y_true'], champion['y_pred_proba'], other['y_pred_proba']
                )
                results['model_comparisons'].append({
                    'comparison': f"{champion_model} vs {model}",
                    'champion_auroc': auc1,
                    'other_auroc': auc2,
                    'diff': auc1 - auc2,
                    'p_value': p,
                    'significance': significance_marker(p)
                })
            except:
                pass
    
    return results


def create_comprehensive_comparison_matrix(df: pd.DataFrame, task: str, 
                                           comparison_type: str,
                                           fixed_model: str = None,
                                           fixed_format: str = None,
                                           fixed_prompt: str = None) -> pd.DataFrame:
    """Create a full pairwise comparison matrix.
    
    Args:
        df: Results DataFrame
        task: Task name
        comparison_type: 'format', 'prompt', or 'model'
        fixed_*: Fixed values for non-compared dimensions
    """
    subset = df[df['task'] == task].copy()
    
    if fixed_model:
        subset = subset[subset['embedding_model'] == fixed_model]
    if fixed_format:
        subset = subset[subset['format'] == fixed_format]
    if fixed_prompt:
        subset = subset[subset['prompt'] == fixed_prompt]
    
    if comparison_type == 'format':
        items = sorted(subset['format'].unique())
        group_col = 'format'
    elif comparison_type == 'prompt':
        items = sorted(subset['prompt'].unique())
        group_col = 'prompt'
    else:  # model
        items = sorted(subset['embedding_model'].unique())
        group_col = 'embedding_model'
    
    # Create matrix
    n = len(items)
    auroc_matrix = pd.DataFrame(index=items, columns=items, dtype=float)
    pvalue_matrix = pd.DataFrame(index=items, columns=items, dtype=float)
    
    # Fill diagonal with AUROCs
    for item in items:
        row = subset[subset[group_col] == item]
        if len(row) > 0:
            auroc_matrix.loc[item, item] = row.iloc[0]['auroc']
    
    # Fill off-diagonal with comparison results
    for i1, i2 in combinations(items, 2):
        r1 = subset[subset[group_col] == i1]
        r2 = subset[subset[group_col] == i2]
        
        if len(r1) > 0 and len(r2) > 0:
            try:
                auc1, auc2, z, p = delong_test(
                    r1.iloc[0]['y_true'],
                    r1.iloc[0]['y_pred_proba'],
                    r2.iloc[0]['y_pred_proba']
                )
                # Store p-value in both directions
                pvalue_matrix.loc[i1, i2] = p
                pvalue_matrix.loc[i2, i1] = p
                
                # Store AUROC difference
                auroc_matrix.loc[i1, i2] = auc1 - auc2
                auroc_matrix.loc[i2, i1] = auc2 - auc1
            except:
                pass
    
    return auroc_matrix, pvalue_matrix


# =============================================================================
# Output Formatting
# =============================================================================

def format_matrix_with_significance(auroc_matrix: pd.DataFrame, 
                                   pvalue_matrix: pd.DataFrame) -> pd.DataFrame:
    """Format comparison matrix with significance markers."""
    result = auroc_matrix.copy().astype(str)
    
    for i in result.index:
        for j in result.columns:
            if i == j:
                # Diagonal: show AUROC
                val = auroc_matrix.loc[i, j]
                if pd.notna(val):
                    result.loc[i, j] = f"{val:.4f}"
                else:
                    result.loc[i, j] = "-"
            else:
                # Off-diagonal: show difference with significance
                diff = auroc_matrix.loc[i, j]
                p = pvalue_matrix.loc[i, j]
                
                if pd.notna(diff) and pd.notna(p):
                    sig = significance_marker(p)
                    result.loc[i, j] = f"{diff:+.4f}{sig}"
                else:
                    result.loc[i, j] = "-"
    
    return result


def print_champion_analysis(results: Dict):
    """Print formatted champion analysis results."""
    print(f"\n{'='*80}")
    print(f"CHAMPION MODEL ANALYSIS: {results['task']}")
    print(f"Champion: {results['champion']}")
    print('='*80)
    
    # Format comparisons
    if results['format_comparisons']:
        print("\n--- FORMAT COMPARISONS (Champion Format vs Others) ---")
        df = pd.DataFrame(results['format_comparisons'])
        print(df.to_string(index=False))
    
    # Prompt comparisons
    if results['prompt_comparisons']:
        print("\n--- PROMPT COMPARISONS (Champion Prompt vs Others) ---")
        df = pd.DataFrame(results['prompt_comparisons'])
        print(df.to_string(index=False))
    
    # Model comparisons
    if results['model_comparisons']:
        print("\n--- EMBEDDING MODEL COMPARISONS (Champion Model vs Others) ---")
        df = pd.DataFrame(results['model_comparisons'])
        print(df.to_string(index=False))


# =============================================================================
# Full Comparison Tables Using Complete Model (text-embedding-large-exp-03-07)
# =============================================================================

def generate_full_format_comparisons(df: pd.DataFrame, output_dir: Path, 
                                     reference_model: str = 'text-embedding-large-exp-03-07'):
    """Generate comprehensive format comparison tables for all tasks and prompts."""
    tasks = sorted(df['task'].unique())
    prompts = sorted(df['prompt'].unique())
    formats = ['F1', 'F2', 'F3']
    
    all_comparisons = []
    
    for task in tasks:
        for prompt in prompts:
            subset = df[(df['task'] == task) & 
                       (df['embedding_model'] == reference_model) & 
                       (df['prompt'] == prompt)]
            
            if len(subset) < 2:
                continue
            
            for f1, f2 in combinations(formats, 2):
                r1 = subset[subset['format'] == f1]
                r2 = subset[subset['format'] == f2]
                
                if len(r1) > 0 and len(r2) > 0:
                    try:
                        auc1, auc2, z, p = delong_test(
                            r1.iloc[0]['y_true'],
                            r1.iloc[0]['y_pred_proba'],
                            r2.iloc[0]['y_pred_proba']
                        )
                        all_comparisons.append({
                            'Task': TASK_MAPPING.get(task, task),
                            'Prompt': prompt,
                            'Format 1': f1,
                            'Format 2': f2,
                            f'{f1} AUROC': auc1,
                            f'{f2} AUROC': auc2,
                            'Difference': auc1 - auc2,
                            'P-Value': p,
                            'Significance': significance_marker(p)
                        })
                    except Exception as e:
                        pass
    
    result_df = pd.DataFrame(all_comparisons)
    if not result_df.empty:
        result_df.to_csv(output_dir / 'format_comparisons_all.csv', index=False)
    return result_df


def generate_full_prompt_comparisons(df: pd.DataFrame, output_dir: Path,
                                     reference_model: str = 'text-embedding-large-exp-03-07'):
    """Generate comprehensive prompt comparison tables for all tasks and formats."""
    tasks = sorted(df['task'].unique())
    prompts = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
    formats = ['F1', 'F2', 'F3']
    
    all_comparisons = []
    
    for task in tasks:
        for fmt in formats:
            subset = df[(df['task'] == task) & 
                       (df['embedding_model'] == reference_model) & 
                       (df['format'] == fmt)]
            
            if len(subset) < 2:
                continue
            
            for p1, p2 in combinations(prompts, 2):
                r1 = subset[subset['prompt'] == p1]
                r2 = subset[subset['prompt'] == p2]
                
                if len(r1) > 0 and len(r2) > 0:
                    try:
                        auc1, auc2, z, p = delong_test(
                            r1.iloc[0]['y_true'],
                            r1.iloc[0]['y_pred_proba'],
                            r2.iloc[0]['y_pred_proba']
                        )
                        all_comparisons.append({
                            'Task': TASK_MAPPING.get(task, task),
                            'Format': fmt,
                            'Prompt 1': p1,
                            'Prompt 2': p2,
                            f'{p1} AUROC': auc1,
                            f'{p2} AUROC': auc2,
                            'Difference': auc1 - auc2,
                            'P-Value': p,
                            'Significance': significance_marker(p)
                        })
                    except Exception as e:
                        pass
    
    result_df = pd.DataFrame(all_comparisons)
    if not result_df.empty:
        result_df.to_csv(output_dir / 'prompt_comparisons_all.csv', index=False)
    return result_df


def generate_latex_table(df: pd.DataFrame, comparison_type: str, output_dir: Path):
    """Generate LaTeX-formatted table for manuscript."""
    if comparison_type == 'format':
        # Create a matrix showing format comparisons per task
        tasks = df['Task'].unique()
        
        latex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Format Comparison (DeLong's Test)}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"Task & F1 vs F2 & F1 vs F3 & F2 vs F3 \\",
            r"\hline"
        ]
        
        for task in sorted(tasks):
            task_data = df[df['Task'] == task]
            row = [task]
            
            for f1, f2 in [('F1', 'F2'), ('F1', 'F3'), ('F2', 'F3')]:
                comp = task_data[(task_data['Format 1'] == f1) & (task_data['Format 2'] == f2)]
                if len(comp) > 0:
                    # Average across prompts
                    avg_p = comp['P-Value'].mean()
                    sig = significance_marker(avg_p)
                    avg_diff = comp['Difference'].mean()
                    row.append(f"${avg_diff:+.3f}${sig}")
                else:
                    row.append("-")
            
            latex_lines.append(" & ".join(row) + r" \\")
        
        latex_lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        with open(output_dir / 'format_comparison_latex.tex', 'w') as f:
            f.write('\n'.join(latex_lines))


def create_pairwise_matrix_table(df: pd.DataFrame, task: str, model: str, 
                                 comparison_type: str, fixed_value: str = None) -> str:
    """Create a formatted pairwise comparison matrix as a string table."""
    subset = df[(df['task'] == task) & (df['embedding_model'] == model)]
    
    if comparison_type == 'format':
        items = ['F1', 'F2', 'F3']
        if fixed_value:
            subset = subset[subset['prompt'] == fixed_value]
        group_col = 'format'
    elif comparison_type == 'prompt':
        items = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
        if fixed_value:
            subset = subset[subset['format'] == fixed_value]
        group_col = 'prompt'
    else:
        return ""
    
    n = len(items)
    
    # Build header
    lines = []
    header = f"{'':8}" + "".join(f"{item:>10}" for item in items)
    lines.append(header)
    lines.append("-" * len(header))
    
    for i, item1 in enumerate(items):
        row = f"{item1:8}"
        for j, item2 in enumerate(items):
            if i == j:
                # Diagonal: show AUROC
                r = subset[subset[group_col] == item1]
                if len(r) > 0:
                    row += f"{r.iloc[0]['auroc']:10.4f}"
                else:
                    row += f"{'-':>10}"
            else:
                # Off-diagonal: show p-value with significance
                r1 = subset[subset[group_col] == item1]
                r2 = subset[subset[group_col] == item2]
                
                if len(r1) > 0 and len(r2) > 0:
                    try:
                        _, _, _, p = delong_test(
                            r1.iloc[0]['y_true'],
                            r1.iloc[0]['y_pred_proba'],
                            r2.iloc[0]['y_pred_proba']
                        )
                        diff = r1.iloc[0]['auroc'] - r2.iloc[0]['auroc']
                        sig = significance_marker(p)
                        row += f"{diff:+.4f}{sig:>4}"[:10]
                    except:
                        row += f"{'-':>10}"
                else:
                    row += f"{'-':>10}"
        lines.append(row)
    
    return '\n'.join(lines)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    # Configuration
    BASE_DIR = Path(__file__).parent / 'embedding_model_results'
    OUTPUT_DIR = Path(__file__).parent / 'statistical_analysis_output'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("DeLong's Test for Statistical Significance of AUROC Differences")
    print("="*80)
    
    # Load all results
    print("\nLoading results from all embedding models...")
    df = load_all_results(BASE_DIR)
    
    if len(df) == 0:
        print("ERROR: No results found!")
        return
    
    # Summary
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"Embedding models with predictions: {sorted(df['embedding_model'].unique())}")
    print(f"Tasks: {sorted(df['task'].unique())}")
    print(f"Formats: {sorted(df['format'].unique())}")
    print(f"Prompts: {sorted(df['prompt'].unique())}")
    print(f"Total records: {len(df)}")
    
    tasks = sorted(df['task'].unique())
    
    # Identify reference model (the one with most complete data)
    model_counts = df['embedding_model'].value_counts()
    reference_model = model_counts.idxmax()
    print(f"\nReference model (most complete data): {reference_model} ({model_counts[reference_model]} records)")
    
    # ==========================================================================
    # 1. Champion Model Analysis (from Table 1)
    # ==========================================================================
    print("\n" + "="*80)
    print("SECTION 1: CHAMPION MODEL STATISTICAL SIGNIFICANCE")
    print("="*80)
    
    all_champion_results = []
    for task, (fmt, prompt, model) in CHAMPION_SEMANTIC.items():
        if task in tasks:
            results = analyze_champion_model(df, task, fmt, prompt, model)
            all_champion_results.append(results)
            print_champion_analysis(results)
    
    # Save champion analysis
    champion_df = []
    for res in all_champion_results:
        task_name = TASK_MAPPING.get(res['task'], res['task'])
        
        for comp in res['format_comparisons']:
            champion_df.append({
                'Task': task_name,
                'Comparison Type': 'Format',
                'Comparison': comp['comparison'],
                'Champion AUROC': comp['champion_auroc'],
                'Other AUROC': comp['other_auroc'],
                'Difference': comp['diff'],
                'P-Value': comp['p_value'],
                'Significance': comp['significance']
            })
        
        for comp in res['prompt_comparisons']:
            champion_df.append({
                'Task': task_name,
                'Comparison Type': 'Prompt',
                'Comparison': comp['comparison'],
                'Champion AUROC': comp['champion_auroc'],
                'Other AUROC': comp['other_auroc'],
                'Difference': comp['diff'],
                'P-Value': comp['p_value'],
                'Significance': comp['significance']
            })
        
        for comp in res['model_comparisons']:
            champion_df.append({
                'Task': task_name,
                'Comparison Type': 'Embedding Model',
                'Comparison': comp['comparison'],
                'Champion AUROC': comp['champion_auroc'],
                'Other AUROC': comp['other_auroc'],
                'Difference': comp['diff'],
                'P-Value': comp['p_value'],
                'Significance': comp['significance']
            })
    
    if champion_df:
        champion_results_df = pd.DataFrame(champion_df)
        champion_results_df.to_csv(OUTPUT_DIR / 'champion_model_comparisons.csv', index=False)
        print(f"\nChampion analysis saved to: {OUTPUT_DIR / 'champion_model_comparisons.csv'}")
    
    # ==========================================================================
    # 2. Comprehensive Format Comparisons (Using Reference Model)
    # ==========================================================================
    print("\n" + "="*80)
    print(f"SECTION 2: FORMAT COMPARISON MATRICES (F1 vs F2 vs F3)")
    print(f"Using reference model: {reference_model}")
    print("="*80)
    
    format_comparisons = generate_full_format_comparisons(df, OUTPUT_DIR, reference_model)
    
    # Print summary per task
    for task in tasks:
        task_name = TASK_MAPPING.get(task, task)
        
        # Use P0 as representative prompt
        print(f"\n{task_name} (Prompt P0):")
        matrix_str = create_pairwise_matrix_table(df, task, reference_model, 'format', 'P0')
        if matrix_str:
            print(matrix_str)
    
    if not format_comparisons.empty:
        print(f"\nFull format comparisons saved to: {OUTPUT_DIR / 'format_comparisons_all.csv'}")
    
    # ==========================================================================
    # 3. Comprehensive Prompt Comparisons (Using Reference Model)
    # ==========================================================================
    print("\n" + "="*80)
    print(f"SECTION 3: PROMPT COMPARISON MATRICES (P0-P5)")
    print(f"Using reference model: {reference_model}")
    print("="*80)
    
    prompt_comparisons = generate_full_prompt_comparisons(df, OUTPUT_DIR, reference_model)
    
    # Print summary per task
    for task in tasks:
        task_name = TASK_MAPPING.get(task, task)
        
        # Use F3 as representative format (best performing in most tasks)
        print(f"\n{task_name} (Format F3):")
        matrix_str = create_pairwise_matrix_table(df, task, reference_model, 'prompt', 'F3')
        if matrix_str:
            print(matrix_str)
    
    if not prompt_comparisons.empty:
        print(f"\nFull prompt comparisons saved to: {OUTPUT_DIR / 'prompt_comparisons_all.csv'}")
    
    # ==========================================================================
    # 4. Embedding Model Comparisons
    # ==========================================================================
    print("\n" + "="*80)
    print("SECTION 4: EMBEDDING MODEL COMPARISON MATRICES")
    print("="*80)
    
    all_model_comparisons = []
    for task in tasks:
        for fmt in ['F1', 'F2', 'F3']:
            for prompt in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']:
                comp_df = embedding_model_comparison(df, task, fmt, prompt)
                if not comp_df.empty:
                    comp_df['Task'] = TASK_MAPPING.get(task, task)
                    comp_df['Format'] = fmt
                    comp_df['Prompt'] = prompt
                    all_model_comparisons.append(comp_df)
    
    if all_model_comparisons:
        model_comp_df = pd.concat(all_model_comparisons, ignore_index=True)
        model_comp_df.to_csv(OUTPUT_DIR / 'model_comparisons_all.csv', index=False)
        
        # Print summary
        print("\nSignificant model comparisons (p < 0.05):")
        sig_comps = model_comp_df[model_comp_df['p_value'] < 0.05]
        if not sig_comps.empty:
            print(sig_comps[['Task', 'Format', 'Prompt', 'model1', 'model2', 'diff', 'p_value', 'significance']].to_string(index=False))
        else:
            print("  No significant differences found.")
        
        print(f"\nFull model comparisons saved to: {OUTPUT_DIR / 'model_comparisons_all.csv'}")
    
    # ==========================================================================
    # 5. Comprehensive Summary Table
    # ==========================================================================
    print("\n" + "="*80)
    print("SECTION 5: COMPREHENSIVE AUROC SUMMARY TABLE")
    print("="*80)
    
    # Create a pivot table of all results
    summary_records = []
    for _, row in df.iterrows():
        summary_records.append({
            'Task': TASK_MAPPING.get(row['task'], row['task']),
            'Embedding Model': row['embedding_model'],
            'Format': row['format'],
            'Prompt': row['prompt'],
            'AUROC': row['auroc'],
            'Arm': f"{row['format']}_{row['prompt']}"
        })
    
    summary_df = pd.DataFrame(summary_records)
    
    # Pivot by embedding model for easy comparison
    for task in tasks:
        task_name = TASK_MAPPING.get(task, task)
        task_data = summary_df[summary_df['Task'] == task_name]
        
        if len(task_data) > 0:
            pivot = task_data.pivot_table(
                index=['Format', 'Prompt'], 
                columns='Embedding Model', 
                values='AUROC'
            )
            
            print(f"\n{task_name}:")
            print(pivot.round(4).to_string())
            
            # Save to CSV
            pivot.to_csv(OUTPUT_DIR / f'auroc_summary_{task}.csv')
    
    # ==========================================================================
    # 6. Summary Statistics
    # ==========================================================================
    print("\n" + "="*80)
    print("SECTION 6: SUMMARY STATISTICS")
    print("="*80)
    
    # Count significant comparisons
    if not format_comparisons.empty:
        fmt_sig = format_comparisons['Significance'].value_counts()
        print(f"\nFormat comparisons significance distribution:")
        print(f"  *   (p < 0.05):  {fmt_sig.get('*', 0)}")
        print(f"  **  (p < 0.01):  {fmt_sig.get('**', 0)}")
        print(f"  *** (p < 0.001): {fmt_sig.get('***', 0)}")
        print(f"  Not significant: {fmt_sig.get('', 0)}")
    
    if not prompt_comparisons.empty:
        prm_sig = prompt_comparisons['Significance'].value_counts()
        print(f"\nPrompt comparisons significance distribution:")
        print(f"  *   (p < 0.05):  {prm_sig.get('*', 0)}")
        print(f"  **  (p < 0.01):  {prm_sig.get('**', 0)}")
        print(f"  *** (p < 0.001): {prm_sig.get('***', 0)}")
        print(f"  Not significant: {prm_sig.get('', 0)}")
    
    print("\n" + "="*80)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("="*80)
    
    return df


if __name__ == '__main__':
    main()

