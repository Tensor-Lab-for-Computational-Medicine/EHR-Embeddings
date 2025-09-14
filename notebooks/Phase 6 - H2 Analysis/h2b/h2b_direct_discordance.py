import os
import argparse
import importlib.util
import pickle
import pandas as pd
import numpy as np
import pysubgroup as ps
import re


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


def _jaccard(a, b):
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

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
            col = s.split('==', 1)[0].split(':', 1)[0].strip()
            if col == 'target' or col not in data.columns:
                continue
            col_type = _infer_col_type(data[col], declared_types)

            if (('[' in s and ':' in s) or ('>' in s) or ('<' in s)):
                if col_type == 'continuous':
                    restricted.append(sel)
                continue

            if '==' in s:
                rhs = s.split('==', 1)[1].strip()
                rhs_clean = rhs.strip().strip('"\'')
                if col_type == 'binary':
                    if rhs_clean.lower() in {'true', '1', 'yes'}:
                        restricted.append(sel)
                elif col_type == 'categorical':
                    if rhs_clean.strip().lower() not in _BASELINE_CATEGORIES:
                        restricted.append(sel)
        except Exception:
            continue
    return restricted


def run_subgroup_discovery(X_features: pd.DataFrame, y_target: pd.Series, max_depth: int, config) -> pd.DataFrame:
    common_index = X_features.index.intersection(y_target.index)
    X = X_features.loc[common_index]
    y = y_target.loc[common_index].astype(int)
    if y.sum() < 10 or (len(y) - int(y.sum())) < 10:
        return pd.DataFrame()
    data = X.copy()
    data['target'] = y.values
    target = ps.BinaryTarget('target', 1)
    searchspace = _build_restricted_searchspace(data, config)
    if not searchspace:
        return pd.DataFrame()
    task = ps.SubgroupDiscoveryTask(
        data,
        target,
        searchspace,
        result_set_size=getattr(config, 'SUBGROUP_MAX_CANDIDATES', 200),
        depth=max_depth,
        qf=ps.WRAccQF(),
        min_quality=0.001,
    )
    result = ps.BeamSearch(beam_width=getattr(config, 'SUBGROUP_MAX_CANDIDATES', 200)).execute(task)
    if not hasattr(result, 'to_dataframe'):
        return pd.DataFrame()
    df = result.to_dataframe()
    if df.empty:
        return pd.DataFrame()
    df = df[df['relative_size_sg'] >= getattr(config, 'SUBGROUP_MIN_SUPPORT', 0.01)]
    min_q = getattr(config, 'SUBGROUP_MIN_QUALITY', 0.0)
    min_l = getattr(config, 'SUBGROUP_MIN_LIFT', 0.0)
    if min_q > 0 or min_l > 0:
        df = df[(df['quality'] >= min_q) | (df['lift'] >= min_l)]
    if df.empty:
        return pd.DataFrame()
    # Redundancy filter
    j_thresh = getattr(config, 'SUBGROUP_JACCARD_MAX', 0.8)
    cover_sets = []
    for _, r in df.iterrows():
        sg = r['subgroup']
        try:
            mask = sg.covers(data)
        except Exception:
            mask = sg.subgroup_description.covers(data)
        cover_sets.append(set(data.index[mask]))
    kept_idx, kept_sets = [], []
    for i in list(df.sort_values('quality', ascending=False).index):
        cs = cover_sets[i]
        if all(_jaccard(cs, ks) <= j_thresh for ks in kept_sets):
            kept_idx.append(i)
            kept_sets.append(cs)
    df = df.loc[kept_idx].reset_index(drop=True)
    rows = []
    for idx in range(len(df)):
        r = df.iloc[idx]
        sg = r['subgroup']
        try:
            mask = sg.covers(data)
        except Exception:
            mask = sg.subgroup_description.covers(data)
        member_idx = data.index[mask]
        rows.append({
            'rank': idx + 1,
            'rule': _normalize_rule_case(str(r['subgroup'])),
            'quality_WRAcc': r['quality'],
            'coverage': int(r['size_sg']),
            'coverage_pct': round(r['relative_size_sg'] * 100, 1),
            'n_positives': int(r['positives_sg']),
            'target_share': round(r['target_share_sg'] * 100, 1),
            'baseline_rate': round(r['target_share_dataset'] * 100, 1),
            'lift': round(r['lift'], 2),
            'members': "|".join(map(str, member_idx.tolist())),
        })
    return pd.DataFrame(rows)


def build_discordant_cohorts(artifact, X_index: pd.Index):
    cbp = artifact['cohorts_by_pos']
    idx = {k: X_index.take(np.asarray(v, dtype=int)) for k, v in cbp.items()}
    # In this codebase, these four sets are already the discordant cases by definition
    cohorts = {
        'SM_Win_on_Deaths': idx.get('FN_NM', pd.Index([])),
        'NM_Win_on_Deaths': idx.get('FN_SM', pd.Index([])),
        'SM_Win_on_Survivors': idx.get('FP_NM', pd.Index([])),
        'NM_Win_on_Survivors': idx.get('FP_SM', pd.Index([])),
    }
    return cohorts


def write_report(
    deaths_sm_df: pd.DataFrame,
    deaths_nm_df: pd.DataFrame,
    surv_sm_df: pd.DataFrame,
    surv_nm_df: pd.DataFrame,
    cohorts: dict,
    out_path: str,
    depth: int,
):
    lines = []
    lines.append('=' * 80)
    lines.append(f'PHASE IV-B: DIRECT DISCORDANCE CHARACTERIZATION (max_depth={depth})')
    lines.append('=' * 80)
    lines.append('')
    lines.append('COHORT SIZES (discordant)')
    for k in ['SM_Win_on_Deaths','NM_Win_on_Deaths','SM_Win_on_Survivors','NM_Win_on_Survivors']:
        lines.append(f'  {k}: {len(cohorts.get(k, []))}')
    lines.append('')
    def _section(title, df):
        lines.append('=' * 80)
        lines.append(title)
        lines.append('=' * 80)
        if df.empty:
            lines.append('No significant patterns discovered.')
            lines.append('')
        else:
            for _, r in df.iterrows():
                lines.append(f"Rule #{int(r['rank'])}: {_normalize_rule_case(r['rule'])}")
                lines.append(f"  WRAcc={r['quality_WRAcc']:.4f}  Lift={r['lift']:.2f}x  Coverage={r['coverage']} ({r['coverage_pct']}%)  Target%={r['target_share']:.1f}%  Baseline%={r['baseline_rate']:.1f}%")
            lines.append('')
    _section('Battleground of the Deceased — SM advantage (is_SM_win=1)', deaths_sm_df)
    _section('Battleground of the Deceased — NM advantage (is_SM_win=0)', deaths_nm_df)
    _section('Battleground of the Survivors — SM advantage (is_SM_win=1)', surv_sm_df)
    _section('Battleground of the Survivors — NM advantage (is_SM_win=0)', surv_nm_df)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description='Phase IV-B: Direct Discordance Characterization')
    p.add_argument('--config_file', type=str, default=None, help='Path to ConfigH2 .py (e.g., config_h2_readmin30.py)')
    p.add_argument('--depths', type=str, default='2,3', help='Comma-separated max_depth values')
    args = p.parse_args()

    cfg = _load_config(args.config_file)
    # Ensure exploratory defaults per plan
    if not hasattr(cfg, 'SUBGROUP_MIN_SUPPORT'):
        cfg.SUBGROUP_MIN_SUPPORT = 0.01
    if not hasattr(cfg, 'SUBGROUP_MIN_QUALITY'):
        cfg.SUBGROUP_MIN_QUALITY = 0.05
    if not hasattr(cfg, 'SUBGROUP_MIN_LIFT'):
        cfg.SUBGROUP_MIN_LIFT = 1.5
    if not hasattr(cfg, 'SUBGROUP_MAX_CANDIDATES'):
        cfg.SUBGROUP_MAX_CANDIDATES = 200
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    art_path = os.path.join(cfg.H2A_OUTPUT_DIR, 'h2a_to_h2b_artifact.pkl')
    if not os.path.exists(art_path):
        raise FileNotFoundError(f'Missing H2a artifact: {art_path}')
    with open(art_path, 'rb') as f:
        art = pickle.load(f)

    # Prefer engineered phenotypes if available; fallback to numeric
    X_test_pheno = None
    if hasattr(cfg, 'X_TEST_PHENOS_PATH') and os.path.exists(cfg.X_TEST_PHENOS_PATH):
        try:
            X_test_pheno = pd.read_pickle(cfg.X_TEST_PHENOS_PATH)
        except Exception:
            X_test_pheno = None
    if X_test_pheno is not None and isinstance(X_test_pheno, pd.DataFrame) and not X_test_pheno.empty:
        X_test = X_test_pheno
    else:
        with open(cfg.X_TEST_NUM_PATH, 'rb') as f:
            X_test = pickle.load(f)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        try:
            with open(getattr(cfg, 'SCALER_PATH'), 'rb') as f:
                _sc = pickle.load(f)
            if hasattr(_sc, 'feature_names_in_'):
                X_test[_sc.feature_names_in_] = _sc.inverse_transform(X_test[_sc.feature_names_in_])
            else:
                X_test = pd.DataFrame(_sc.inverse_transform(X_test), columns=X_test.columns, index=X_test.index)
        except Exception:
            pass

    cohorts = build_discordant_cohorts(art, pd.Index(X_test.index))

    # Build Analysis 1 (Deaths) target: is_SM_win=1 for SM_Win_on_Deaths, 0 for NM_Win_on_Deaths
    deaths_pop = cohorts['SM_Win_on_Deaths'].union(cohorts['NM_Win_on_Deaths'])
    deaths_X = X_test.loc[X_test.index.isin(deaths_pop)]
    deaths_y = pd.Series(0, index=deaths_X.index)
    deaths_y.loc[deaths_y.index.isin(cohorts['SM_Win_on_Deaths'])] = 1

    # Analysis 2 (Survivors)
    surv_pop = cohorts['SM_Win_on_Survivors'].union(cohorts['NM_Win_on_Survivors'])
    surv_X = X_test.loc[X_test.index.isin(surv_pop)]
    surv_y = pd.Series(0, index=surv_X.index)
    surv_y.loc[surv_y.index.isin(cohorts['SM_Win_on_Survivors'])] = 1

    depths = [int(x.strip()) for x in args.depths.split(',') if x.strip()]
    base_output = cfg.OUTPUT_DIR
    for d in depths:
        out_dir = os.path.join(base_output, 'phase_ivb', f'depth_{d}')
        os.makedirs(out_dir, exist_ok=True)
        # Discover for SM advantage (target=1) and NM advantage (target=0)
        deaths_sm_df = run_subgroup_discovery(deaths_X, deaths_y, d, cfg)
        deaths_nm_y = 1 - deaths_y
        deaths_nm_df = run_subgroup_discovery(deaths_X, deaths_nm_y, d, cfg)
        surv_sm_df = run_subgroup_discovery(surv_X, surv_y, d, cfg)
        surv_nm_y = 1 - surv_y
        surv_nm_df = run_subgroup_discovery(surv_X, surv_nm_y, d, cfg)

        # Write candidates only (no final cohorts here)
        if not deaths_sm_df.empty:
            deaths_sm_df.to_csv(os.path.join(out_dir, f'ivb_patterns_deaths_SM_depth_{d}.csv'), index=False)
        if not deaths_nm_df.empty:
            deaths_nm_df.to_csv(os.path.join(out_dir, f'ivb_patterns_deaths_NM_depth_{d}.csv'), index=False)
        if not surv_sm_df.empty:
            surv_sm_df.to_csv(os.path.join(out_dir, f'ivb_patterns_survivors_SM_depth_{d}.csv'), index=False)
        if not surv_nm_df.empty:
            surv_nm_df.to_csv(os.path.join(out_dir, f'ivb_patterns_survivors_NM_depth_{d}.csv'), index=False)

        # Detailed report (patterns only; archetypes selected cross-depth elsewhere)
        write_report(
            deaths_sm_df,
            deaths_nm_df,
            surv_sm_df,
            surv_nm_df,
            cohorts,
            os.path.join(out_dir, f'ivb_detailed_report_depth_{d}.txt'),
            d,
        )

        # Minimal console output
        for name, df in [
            ('Deaths-SM', deaths_sm_df), ('Deaths-NM', deaths_nm_df),
            ('Survivors-SM', surv_sm_df), ('Survivors-NM', surv_nm_df)
        ]:
            n = 0 if df is None or df.empty else df.shape[0]
            print(f'[depth={d}] {name}: {n} patterns kept')

        # Write per-depth final_subgroups with metrics for cross-depth selection
        def _collect(df, analysis_key):
            if df is None or df.empty:
                return []
            return [{
                'analysis_key': analysis_key,
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
            } for _, r in df.iterrows()]

        rows = []
        rows += _collect(deaths_sm_df, 'IVB_deaths_SM')
        rows += _collect(deaths_nm_df, 'IVB_deaths_NM')
        rows += _collect(surv_sm_df, 'IVB_survivors_SM')
        rows += _collect(surv_nm_df, 'IVB_survivors_NM')
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'final_subgroups_ivb.csv'), index=False)


if __name__ == '__main__':
    main()