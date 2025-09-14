import pandas as pd
import numpy as np
import os
import pickle
import argparse
import importlib.util
import re

import pysubgroup as ps


def _jaccard(a, b):
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union

def _normalize_rule_case(rule_str: str) -> str:
    s = str(rule_str)
    s = re.sub(r"\band\b", "AND", s, flags=re.IGNORECASE)
    s = re.sub(r"\bor\b", "OR", s, flags=re.IGNORECASE)
    return s

 

def _get_phenotype_types(config) -> dict:
    mapping = {}
    try:
        csv_path = getattr(config, 'PHENOTYPE_RULES_CSV', None)
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if {'phenotype_name', 'phenotype_type'}.issubset(df.columns):
                mapping = {str(r['phenotype_name']).strip(): str(r['phenotype_type']).strip().lower() for _, r in df.iterrows()}
    except Exception:
        pass
    return mapping


_BASELINE_CATEGORIES = {
    'normal', 'none', 'stage_0', 'stable', 'normotension'
}


def _infer_col_type(series: pd.Series, declared_types: dict) -> str:
    name = series.name
    t = declared_types.get(name)
    if t:
        return t
    if series.dtype == bool:
        return 'binary'
    if pd.api.types.is_numeric_dtype(series):
        return 'continuous'
    return 'categorical'


def _build_restricted_searchspace(data: pd.DataFrame, config):
    declared_types = _get_phenotype_types(config)
    all_selectors = ps.create_selectors(data, ignore=['target'])
    restricted = []
    for sel in all_selectors:
        try:
            s = str(sel)
            # Extract column name conservatively before '==' or ':'
            col = s.split('==', 1)[0].split(':', 1)[0].strip()
            if col == 'target' or col not in data.columns:
                continue
            col_type = _infer_col_type(data[col], declared_types)

            # Numeric selectors typically render like "col: [a:b["; keep if column is continuous
            if (('[' in s and ':' in s) or ('>' in s) or ('<' in s)):
                if col_type == 'continuous':
                    restricted.append(sel)
                continue

            # Equality selectors: "col==value"
            if '==' in s:
                rhs = s.split('==', 1)[1].strip()
                rhs_clean = rhs.strip().strip('"\'')
                if col_type == 'binary':
                    # Only keep TRUE condition
                    if rhs_clean.lower() in {'true', '1', 'yes'}:
                        restricted.append(sel)
                elif col_type == 'categorical':
                    if rhs_clean.strip().lower() not in _BASELINE_CATEGORIES:
                        restricted.append(sel)
                # Ignore equality for continuous
            # Otherwise skip
        except Exception:
            # On any parsing issues, skip the selector to keep the space conservative
            continue
    return restricted


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
    # Build restricted searchspace from clinically significant conditions
    searchspace = _build_restricted_searchspace(data, config)
    if not searchspace:
        return pd.DataFrame()
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
                'rule': _normalize_rule_case(str(row['subgroup'])),
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
# REPORTING (concise)
# =============================================================================

def create_detailed_report(subgroup_results, cohorts_idx, config, depth: int):
    """Write concise per-depth report (patterns only)."""
    report = []
    report.append("="*80)
    report.append(f"H2b ANALYSIS REPORT: DIFFERENTIAL COHORT PROFILES (max_depth={depth})")
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
        # Dataset baseline rate for this analysis population (same across rows)
        if not results_df.empty and 'baseline_rate' in results_df.columns:
            report.append(f"Dataset baseline rate: {results_df.iloc[0]['baseline_rate']:.1f}%")
        report.append("")
        
        if results_df.empty:
            report.append("No significant patterns discovered")
        else:
            report.append("TOP DISCOVERED PATTERNS:")
            report.append("-"*40)
            
            for idx in range(len(results_df)):
                row = results_df.iloc[idx]
                report.append(f"\nPattern #{row['rank']}:")
                report.append(f"  Rule: {_normalize_rule_case(row['rule'])}")
                report.append(f"  WRAcc: {row.get('quality_WRAcc', 'N/A')}")
                report.append(f"  Coverage: {row['coverage']} patients ({row['coverage_pct']}%)")
                report.append(f"  Target share: {row.get('target_share', 0):.1f}%  |  Baseline: {row.get('baseline_rate', 0):.1f}%")
                report.append(f"  Lift: {row.get('lift', 0):.2f}x")
                
        
        report.append("")
    
    # Summary
    report.append("="*80)
    report.append("SUMMARY")
    report.append("="*80)
    report.append("See discovered patterns above; interpret strength, lift, and clinical meaning.")
    
    report_text = "\n".join(report)
    
    # Save report
    with open(os.path.join(config.OUTPUT_DIR, f'h2b_detailed_report_depth_{depth}.txt'), 'w', encoding='utf-8') as f:
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
    parser.add_argument('--max_depth', type=int, default=None, help='Override subgroup discovery max_depth (single run).')
    parser.add_argument('--depths', type=str, default=None, help='Comma-separated list of depths to sweep (e.g., 2,3,4,5). Overrides --max_depth if provided.')
    args = parser.parse_args()

    config = _load_config(args.config_file)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Resolve depth sweep
    if args.depths:
        depths = [int(d.strip()) for d in args.depths.split(',') if d.strip()]
    elif args.max_depth is not None:
        depths = [int(args.max_depth)]
    else:
        depths = [int(getattr(config, 'SUBGROUP_MAX_DEPTH', 3))]

    # Load once: artifact and interpretable features
    artifact_path = os.path.join(config.H2A_OUTPUT_DIR, 'h2a_to_h2b_artifact.pkl')
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Missing H2a artifact: {artifact_path}")
    with open(artifact_path, 'rb') as f:
        artifact = pickle.load(f)

    required_keys = ['y_true', 'cohorts_by_pos']
    if any(k not in artifact for k in required_keys):
        raise KeyError(f"H2a artifact missing keys: {required_keys}")

    # Prefer engineered phenotypes if available; fallback to numeric
    X_test_pheno = None
    if hasattr(config, 'X_TEST_PHENOS_PATH') and os.path.exists(config.X_TEST_PHENOS_PATH):
        try:
            X_test_pheno = pd.read_pickle(config.X_TEST_PHENOS_PATH)
        except Exception:
            X_test_pheno = None
    if X_test_pheno is not None and isinstance(X_test_pheno, pd.DataFrame) and not X_test_pheno.empty:
        X_features = X_test_pheno
    else:
        with open(config.X_TEST_NUM_PATH, 'rb') as f:
            X_test_num = pickle.load(f)
        if not isinstance(X_test_num, pd.DataFrame):
            X_test_num = pd.DataFrame(X_test_num)
        # Inverse transform using configured scaler
        try:
            with open(getattr(config, 'SCALER_PATH'), 'rb') as f:
                _sc = pickle.load(f)
            if hasattr(_sc, 'feature_names_in_'):
                X_test_num[_sc.feature_names_in_] = _sc.inverse_transform(X_test_num[_sc.feature_names_in_])
            else:
                X_test_num = pd.DataFrame(_sc.inverse_transform(X_test_num), columns=X_test_num.columns, index=X_test_num.index)
        except Exception:
            pass
        X_features = X_test_num

    full_index = pd.Index(X_features.index)
    cohorts_idx = {name: full_index.take(np.asarray(idx_list, dtype=int)) for name, idx_list in artifact['cohorts_by_pos'].items()}

    # Sweep depths; write each run to phase_iv/depth_{d}/
    base_output = config.OUTPUT_DIR
    multi = len(depths) > 1
    for d in depths:
        config.SUBGROUP_MAX_DEPTH = d
        run_dir = os.path.join(base_output, 'phase_iv', f'depth_{d}') if multi else base_output
        config.OUTPUT_DIR = run_dir
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        results = analyze_differential_failures(cohorts_idx, X_features, config)

        # Minimal console output per analysis
        for key, meta in results.items():
            n = 0 if meta['results'].empty else meta['results'].shape[0]
            print(f"[depth={d}] {meta['title']}: {n} patterns kept")

        # Write patterns per analysis (no per-depth archetypes)
        for key, analysis in results.items():
            df = analysis['results']
            if not df.empty:
                df.to_csv(os.path.join(config.OUTPUT_DIR, f'h2b_patterns_{key}_depth_{d}.csv'), index=False)

        # Write per-depth final_subgroups with metrics for cross-depth selection
        rows = []
        for key, analysis in results.items():
            df = analysis['results']
            if df is None or df.empty:
                continue
            rows.extend([
                {
                    'analysis_key': key,
                    'rank': int(r['rank']),
                    'rule_str': r['rule'],
                    'coverage': int(r.get('coverage', 0)),
                    'coverage_pct': float(r.get('coverage_pct', 0.0)),
                    'lift': float(r.get('lift', 0.0)),
                    'quality_WRAcc': float(r.get('quality_WRAcc', 0.0)),
                    'target_share': float(r.get('target_share', 0.0)),
                    'baseline_rate': float(r.get('baseline_rate', 0.0)),
                    'members': r.get('members', ''),
                    'source_depth': f'depth_{d}',
                }
                for _, r in df.iterrows()
            ])
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(config.OUTPUT_DIR, 'final_subgroups.csv'), index=False)

        # Write concise per-depth report (patterns only)
        create_detailed_report(results, cohorts_idx, config, d)


    # Restore output dir in case the object is reused elsewhere
    config.OUTPUT_DIR = base_output

if __name__ == "__main__":
    main()