import pandas as pd
import numpy as np
import os
import pickle
import argparse
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

import pysubgroup as ps

# Set up plotting style
try:
    plt.style.use('seaborn-darkgrid')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")

# =============================================================================
# DATA LOADING (minimal)
# =============================================================================

# =============================================================================
# SUBGROUP DISCOVERY
# =============================================================================

def _jaccard(a, b):
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union

def run_subgroup_discovery_analysis(X_features, target, analysis_name, config):
    
    # Ensure features and target have same index
    common_index = X_features.index.intersection(target.index)
    X_analysis = X_features.loc[common_index].copy()
    y_analysis = target.loc[common_index].copy()
    
    # Check for sufficient samples
    n_positive = y_analysis.sum()
    n_negative = len(y_analysis) - n_positive
    
    MIN_SAMPLES_PER_CLASS = 10
    if n_positive < MIN_SAMPLES_PER_CLASS or n_negative < MIN_SAMPLES_PER_CLASS:
        return pd.DataFrame()
    
    # Create dataset for pysubgroup
    data = X_analysis.copy()
    data['target'] = y_analysis.values
    
    # Create binary target
    target_column = ps.BinaryTarget('target', 1)
    searchspace = ps.create_selectors(data, ignore=['target'])
    max_candidates = getattr(config, 'SUBGROUP_MAX_CANDIDATES', 20)
    task = ps.SubgroupDiscoveryTask(
        data,
        target_column,
        searchspace,
        result_set_size=max_candidates,
        depth=config.SUBGROUP_MAX_DEPTH,
        qf=ps.WRAccQF(),
        min_quality=0.001,
    )
    result = ps.BeamSearch(beam_width=max_candidates).execute(task)
    results_data = []
    if hasattr(result, 'to_dataframe'):
        result_df = result.to_dataframe()
        result_df = result_df[result_df['relative_size_sg'] >= config.SUBGROUP_MIN_SUPPORT]
        result_df = result_df[(result_df['quality'] >= getattr(config, 'SUBGROUP_MIN_QUALITY', 0.0)) |
                              (result_df['lift'] >= getattr(config, 'SUBGROUP_MIN_LIFT', 0.0))]
        result_df = result_df.drop_duplicates(subset=['size_sg','positives_sg','target_share_sg']).reset_index(drop=True)

        # Redundancy filter by Jaccard similarity of coverage
        j_thresh = getattr(config, 'SUBGROUP_JACCARD_MAX', 0.8)
        # Precompute coverage index sets
        cover_sets = []
        for _, r in result_df.iterrows():
            sg = r['subgroup']
            try:
                mask = sg.covers(data)
            except Exception:
                mask = sg.subgroup_description.covers(data)
            cover_sets.append(set(data.index[mask]))
        # Keep best-quality non-redundant subgroups
        order = list(result_df.sort_values('quality', ascending=False).index)
        kept_idx, kept_sets = [], []
        for i in order:
            cs = cover_sets[i]
            if all(_jaccard(cs, ks) <= j_thresh for ks in kept_sets):
                kept_idx.append(i)
                kept_sets.append(cs)
        result_df = result_df.loc[kept_idx].reset_index(drop=True)

        for idx in range(len(result_df)):
            row = result_df.iloc[idx]
            sg = row['subgroup']
            try:
                mask = sg.covers(data)
            except Exception:
                mask = sg.subgroup_description.covers(data)
            member_idx = data.index[mask]
            results_data.append({
                'rank': idx + 1,
                'rule': str(row['subgroup']),
                'quality_WRAcc': row['quality'],
                'coverage': int(row['size_sg']),
                'coverage_pct': round(row['relative_size_sg'] * 100, 1),
                'n_positives': int(row['positives_sg']),
                'target_share': round(row['target_share_sg'] * 100, 1),
                'baseline_rate': round(row['target_share_dataset'] * 100, 1),
                'lift': round(row['lift'], 2),
                'members': "|".join(map(str, member_idx.tolist())),
            })
    return pd.DataFrame(results_data)

# Removed univariate fallback to keep script concise and deterministic

def analyze_differential_failures(cohorts_idx, X_test_num, config):
    X_features = X_test_num.copy()
    analyses = getattr(config, 'ANALYSES', [
        ('SM_miss', 'FN_SM', 'TP_concordant', 'SM False Negatives vs Concordant True Positives'),
        ('SM_false_alarm', 'FP_SM', 'TN_concordant', 'SM False Positives vs Concordant True Negatives'),
        ('NM_miss', 'FN_NM', 'TP_concordant', 'NM False Negatives vs Concordant True Positives'),
        ('NM_false_alarm', 'FP_NM', 'TN_concordant', 'NM False Positives vs Concordant True Negatives')
    ])
    all_results = {}
    for analysis_key, error_cohort, success_cohort, title in analyses:
        error_idx = cohorts_idx.get(error_cohort, pd.Index([]))
        success_idx = cohorts_idx.get(success_cohort, pd.Index([]))
        pop_index = error_idx.union(success_idx)
        X_population = X_features.loc[X_features.index.isin(pop_index)]
        target = pd.Series(0, index=X_population.index)
        target.loc[target.index.isin(error_idx)] = 1
        results_df = run_subgroup_discovery_analysis(
            X_features=X_population,
            target=target,
            analysis_name=title,
            config=config,
        )
        all_results[analysis_key] = {
            'title': title,
            'results': results_df,
            'error_cohort': error_cohort,
            'success_cohort': success_cohort,
            'error_count': len(error_idx),
            'success_count': len(success_idx),
        }
    return all_results

##

# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_feature_distributions(subgroup_results, cohorts_idx, X_test_num, config):
    """Create distribution plots for top features from each analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    analysis_order = ['SM_miss', 'SM_false_alarm', 'NM_miss', 'NM_false_alarm']    
    
    for idx, analysis_key in enumerate(analysis_order):
        ax = axes[idx]
        
        if analysis_key not in subgroup_results:
            ax.text(0.5, 0.5, 'No results', ha='center', va='center')
            ax.set_title(f"{analysis_key} Analysis")
            continue
        
        analysis = subgroup_results[analysis_key]
        results_df = analysis['results']
        
        if results_df.empty:
            ax.text(0.5, 0.5, 'No patterns', ha='center', va='center')
            ax.set_title(analysis['title'])
            continue
        
        # Get the top rule and extract the first feature
        top_rule = results_df.iloc[0]['rule']
        
        # Extract feature name from rule (simple parsing)
        feature_name = None
        for col in X_test_num.columns:
            if col in top_rule:
                feature_name = col
                break
        
        if feature_name is None:
            ax.text(0.5, 0.5, f'Top rule:\n{top_rule[:50]}...', ha='center', va='center', fontsize=9)
            ax.set_title(analysis['title'])
            continue
        
        # Get data for the cohorts
        error_idx = cohorts_idx.get(analysis['error_cohort'], pd.Index([]))
        success_idx = cohorts_idx.get(analysis['success_cohort'], pd.Index([]))
        error_data = X_test_num.loc[X_test_num.index.isin(error_idx), feature_name].dropna()
        success_data = X_test_num.loc[X_test_num.index.isin(success_idx), feature_name].dropna()
        
        # Create violin plot
        plot_data = pd.DataFrame({
            'Value': np.concatenate([error_data, success_data]),
            'Cohort': ['Error'] * len(error_data) + ['Success'] * len(success_data)
        })
        
        sns.violinplot(data=plot_data, x='Cohort', y='Value', ax=ax, palette=['red', 'green'])
        ax.set_title(f"{analysis['title']}\nTop Feature: {feature_name}", fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel(feature_name)
        
        # Add statistical annotation
        _, p_value = mannwhitneyu(error_data, success_data, alternative='two-sided')
        ax.text(0.5, 0.95, f'p={p_value:.3f}', transform=ax.transAxes, 
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'h2b_feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# REPORTING
# =============================================================================

def create_summary_table(subgroup_results, config):
    """Create summary table of discovered subgroups."""
    summary_data = []
    
    for analysis_key, analysis in subgroup_results.items():
        results_df = analysis['results']
        
        if results_df.empty:
            continue
        
        # Include all discovered subgroups (post-filtered)
        for idx in range(len(results_df)):
            row = results_df.iloc[idx]
            
            summary_data.append({
                'Analysis': analysis['title'],
                'Rank': row['rank'],
                'Rule': row['rule'][:100],
                'WRAcc': row.get('quality_WRAcc', row.get('lift', 0)),
                'Coverage': f"{row['coverage']} ({row['coverage_pct']}%)",
                'Target_Share': f"{row.get('target_share', 0):.1f}%",
                'Lift': row.get('lift', 0)
            })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def create_detailed_report(subgroup_results, cohorts_idx, config):
    """Create detailed H2b analysis report."""
    report = []
    report.append("="*80)
    report.append("H2b ANALYSIS REPORT: DIFFERENTIAL COHORT PROFILES")
    report.append("="*80)
    report.append("")
    
    report.append("METHOD: Subgroup Discovery with pysubgroup")
    report.append(f"  - Max depth: {config.SUBGROUP_MAX_DEPTH}")
    report.append(f"  - Min support: {config.SUBGROUP_MIN_SUPPORT*100:.0f}%")
    report.append(f"  - Quality measure: WRAcc")
    report.append("")
    
    # Cohort sizes
    report.append("COHORT SIZES")
    report.append("-"*40)
    for name, idx_vals in cohorts_idx.items():
        report.append(f"  {name}: {len(idx_vals)} patients")
    report.append("")
    
    # Results for each analysis
    analysis_order = [
        ('SM_miss', 'SM False Negatives vs Concordant True Positives'),
        ('SM_false_alarm', 'SM False Positives vs Concordant True Negatives'),
        ('NM_miss', 'NM False Negatives vs Concordant True Positives'),
        ('NM_false_alarm', 'NM False Positives vs Concordant True Negatives')
    ]
    
    for analysis_key, title in analysis_order:
        report.append("="*80)
        report.append(f"ANALYSIS: {title}")
        report.append("="*80)
        
        if analysis_key not in subgroup_results:
            report.append("No results available")
            report.append("")
            continue
        
        analysis = subgroup_results[analysis_key]
        results_df = analysis['results']
        
        report.append(f"Error cohort size: {analysis['error_count']}")
        report.append(f"Success cohort size: {analysis['success_count']}")
        report.append("")
        
        if results_df.empty:
            report.append("No significant patterns discovered")
        else:
            report.append("TOP DISCOVERED PATTERNS:")
            report.append("-"*40)
            
            for idx in range(len(results_df)):
                row = results_df.iloc[idx]
                report.append(f"\nPattern #{row['rank']}:")
                report.append(f"  Rule: {row['rule']}")
                report.append(f"  WRAcc: {row.get('quality_WRAcc', 'N/A')}")
                report.append(f"  Coverage: {row['coverage']} patients ({row['coverage_pct']}%)")
                report.append(f"  Target share: {row.get('target_share', 0):.1f}%")
                report.append(f"  Lift: {row.get('lift', 0):.2f}x")
                
        
        report.append("")
    
    # Summary
    report.append("="*80)
    report.append("SUMMARY")
    report.append("="*80)
    report.append("See discovered patterns above; interpret strength, lift, and clinical meaning.")
    
    report_text = "\n".join(report)
    
    # Save report
    with open(os.path.join(config.OUTPUT_DIR, 'h2b_detailed_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def _load_config(config_file: str = None):
    if config_file:
        if not os.path.isabs(config_file):
            config_file = os.path.abspath(config_file)
        spec = importlib.util.spec_from_file_location('dynamic_config_h2', config_file)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod.ConfigH2()
    else:
        from config_h2_readmin30 import ConfigH2 as _Cfg
        return _Cfg()


def main():
    parser = argparse.ArgumentParser(description='H2b analysis (reusable)')
    parser.add_argument('--config_file', type=str, default=None, help='Path to ConfigH2 .py file (e.g., config_h2_readmin30.py). Defaults to morthosp config if omitted.')
    args = parser.parse_args()

    config = _load_config(args.config_file)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Strictly rely on H2a artifact (no fallbacks)
    artifact_path = os.path.join(config.H2A_OUTPUT_DIR, 'h2a_to_h2b_artifact.pkl')
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Missing H2a artifact: {artifact_path}")
    with open(artifact_path, 'rb') as f:
        artifact = pickle.load(f)

    # Required fields from artifact
    required_keys = ['y_true', 'cohorts_by_pos']
    if any(k not in artifact for k in required_keys):
        raise KeyError(f"H2a artifact missing keys: {required_keys}")

    # Load numeric test features used for subgroup discovery
    with open(config.X_TEST_NUM_PATH, 'rb') as f:
        X_test_num = pickle.load(f)
    if not isinstance(X_test_num, pd.DataFrame):
        X_test_num = pd.DataFrame(X_test_num)
    # Revert standardized features to original clinical scale for interpretability
    with open(r'D:\Projects\EHR Embeddings\notebooks\Phase 1 and 2\phase_1_outputs\scaler.pkl', 'rb') as f:
        _sc = pickle.load(f)
    if hasattr(_sc, 'feature_names_in_'):
        X_test_num[_sc.feature_names_in_] = _sc.inverse_transform(X_test_num[_sc.feature_names_in_])
    else:
        X_test_num = pd.DataFrame(_sc.inverse_transform(X_test_num), columns=X_test_num.columns, index=X_test_num.index)

    # Build index lists from artifact positions aligned to X_test_num
    full_index = pd.Index(X_test_num.index)
    cohorts_idx = {name: full_index.take(np.asarray(idx_list, dtype=int)) for name, idx_list in artifact['cohorts_by_pos'].items()}

    results = analyze_differential_failures(cohorts_idx, X_test_num, config)
    plot_feature_distributions(results, cohorts_idx, X_test_num, config)

    summary_df = create_summary_table(results, config)
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'h2b_summary_table.csv'), index=False)
        print(summary_df.to_string(index=False))

    report = create_detailed_report(results, cohorts_idx, config)
    print("\n" + report)

    for key, analysis in results.items():
        if not analysis['results'].empty:
            analysis['results'].to_csv(os.path.join(config.OUTPUT_DIR, f'h2b_patterns_{key}.csv'), index=False)

    # Export vetted subgroups for Phase V (final_subgroups.csv)
    final_rows = [
        {'analysis_key': k, 'rank': int(r['rank']), 'rule_str': r['rule'], 'members': r.get('members', '')}
        for k, a in results.items() if not a['results'].empty
        for _, r in a['results'].iterrows()
    ]
    if final_rows:
        pd.DataFrame(final_rows).to_csv(os.path.join(config.OUTPUT_DIR, 'final_subgroups.csv'), index=False)

    with open(os.path.join(config.OUTPUT_DIR, 'h2b_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()