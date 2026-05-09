import os
import argparse
import importlib.util
import pickle
import re
import pandas as pd
import numpy as np
import pysubgroup as ps
import sys


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
            col = s.split('==', 1)[0].split(':', 1)[0].strip()
            if col == 'target' or col not in data.columns:
                continue
            col_type = _infer_col_type(data[col], declared_types)

            # Numeric selectors like "col: [a:b[" or inequalities
            if (('[' in s and ':' in s) or ('>' in s) or ('<' in s)):
                if col_type == 'continuous':
                    restricted.append(sel)
                continue

            # Equality selectors: keep only informative categories
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


def _parse_interval(token: str):
    # token like "[a:b[" or "]a:b]"; treat left-inclusive if '[' else exclusive
    m = re.match(r"^([\[\]])\s*([^:]+)\s*:\s*([^\[\]]+)\s*([\[\]])$", token.strip())
    if not m:
        return None
    left_sym, a, b, right_sym = m.groups()
    def _num(x):
        try:
            return float(x)
        except Exception:
            return None
    return {
        'left_incl': left_sym == '[',
        'right_incl': right_sym == ']',
        'a': _num(a),
        'b': _num(b),
    }


def _evaluate_simple_predicate(df: pd.DataFrame, expr: str) -> pd.Series:
    # Supported forms: col == value, col > x, col >= x, col < x, col <= x, col: [a:b[
    s = expr.strip()
    # Interval
    if ':' in s and ('[' in s or ']' in s):
        col, rest = s.split(':', 1)
        col = col.strip()
        iv = _parse_interval(rest.strip())
        if iv is None or col not in df.columns:
            return pd.Series(False, index=df.index)
        x = pd.to_numeric(df[col], errors='coerce')
        ge = (x >= iv['a']) if iv['left_incl'] else (x > iv['a'])
        le = (x <= iv['b']) if iv['right_incl'] else (x < iv['b'])
        return (ge & le).fillna(False)
    # Equality
    if '==' in s:
        col, val = s.split('==', 1)
        col = col.strip()
        val = val.strip().strip('"').strip("'")
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        series = df[col]
        if series.dtype == bool:
            tgt = str(val).lower() in {'true','1','yes'}
            return (series.astype(bool) == tgt)
        return (series.astype(str) == str(val))
    # Inequalities
    for op in ['>=','<=','>','<']:
        if op in s:
            col, val = s.split(op, 1)
            col = col.strip()
            try:
                valf = float(val.strip())
            except Exception:
                valf = np.nan
            if col not in df.columns:
                return pd.Series(False, index=df.index)
            x = pd.to_numeric(df[col], errors='coerce')
            if op == '>=':
                return (x >= valf).fillna(False)
            if op == '<=':
                return (x <= valf).fillna(False)
            if op == '>':
                return (x > valf).fillna(False)
            if op == '<':
                return (x < valf).fillna(False)
    return pd.Series(False, index=df.index)


def _mask_for_rule(rule_str: str, df: pd.DataFrame) -> pd.Series:
    parts = [p.strip() for p in str(rule_str).split('AND')]
    if not parts:
        return pd.Series(False, index=df.index)
    mask = pd.Series(True, index=df.index)
    for p in parts:
        if not p:
            continue
        mask = mask & _evaluate_simple_predicate(df, p)
    return mask


def _safe_cover_mask(subgroup, data: pd.DataFrame):
    try:
        return subgroup.covers(data)
    except Exception:
        return subgroup.subgroup_description.covers(data)


def _fdr_bh(p_values, alpha=0.05):
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    if m == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    # Benjamini-Hochberg
    prev = 1.0
    for i in range(m - 1, -1, -1):
        q[i] = min(prev, ranked[i] * m / (i + 1))
        prev = q[i]
    # Unsort
    q_full = np.empty_like(q)
    q_full[order] = q
    return q_full


def _compute_lift(mask: pd.Series, y: pd.Series) -> float:
    if mask.sum() == 0:
        return 0.0
    base = float(y.mean()) if y.size else 0.0
    if base <= 0:
        return 0.0
    rate = float(y[mask].mean())
    return rate / base if base > 0 else 0.0


def _augment_and_prune_rules(
    analysis_key: str,
    df_rules: pd.DataFrame,
    X_train: pd.DataFrame,
    train_error_idx: pd.Index,
    train_success_idx: pd.Index,
    X_test: pd.DataFrame,
    test_error_idx: pd.Index,
    test_success_idx: pd.Index,
    alpha: float = 0.05,
):
    if df_rules is None or df_rules.empty:
        return df_rules
    # Build train/test population masks and targets
    pop_tr = train_error_idx.union(train_success_idx)
    pop_ts = test_error_idx.union(test_success_idx)
    Xtr = X_train.loc[X_train.index.isin(pop_tr)]
    Xts = X_test.loc[X_test.index.isin(pop_ts)]
    ytr = pd.Series(0, index=Xtr.index)
    ytr.loc[ytr.index.isin(train_error_idx)] = 1
    yts = pd.Series(0, index=Xts.index)
    yts.loc[yts.index.isin(test_error_idx)] = 1

    pvals = []
    pvals_ts = []
    lifts_tr = []
    lifts_ts = []
    cov_tr = []
    cov_ts = []
    cov_ts_pct = []
    ts_target_share = []
    ts_baseline_rate = []
    wracc_ts = []
    members_ts = []
    n_ts = int(len(Xts))
    for _, r in df_rules.iterrows():
        rule = r.get('rule') or r.get('rule_str')
        mtr = _mask_for_rule(str(rule), Xtr)
        mts = _mask_for_rule(str(rule), Xts)
        # Train contingency
        a = int(((mtr == True) & (ytr == 1)).sum())
        b = int(((mtr == True) & (ytr == 0)).sum())
        c = int(((mtr == False) & (ytr == 1)).sum())
        d = int(((mtr == False) & (ytr == 0)).sum())
        try:
            from scipy.stats import fisher_exact
            _, p = fisher_exact([[a, b], [c, d]], alternative='two-sided')
        except Exception:
            # Fallback: simple heuristic p if scipy missing
            p = 1.0
        pvals.append(float(p))
        # Test contingency
        a_ts = int(((mts == True) & (yts == 1)).sum())
        b_ts = int(((mts == True) & (yts == 0)).sum())
        c_ts = int(((mts == False) & (yts == 1)).sum())
        d_ts = int(((mts == False) & (yts == 0)).sum())
        try:
            from scipy.stats import fisher_exact as _fisher
            _, p_ts = _fisher([[a_ts, b_ts], [c_ts, d_ts]], alternative='two-sided')
        except Exception:
            p_ts = 1.0
        pvals_ts.append(float(p_ts))
        lifts_tr.append(_compute_lift(mtr, ytr))
        lifts_ts.append(_compute_lift(mts, yts))
        cov_tr.append(int(mtr.sum()))
        cov_ts_val = int(mts.sum())
        cov_ts.append(cov_ts_val)
        cov_ts_pct.append(round((100.0 * cov_ts_val / n_ts), 1) if n_ts else 0.0)
        base_rate = float(yts.mean()) if yts.size else 0.0
        rate_in_sg = float(yts[mts].mean()) if cov_ts_val > 0 else 0.0
        ts_target_share.append(round(rate_in_sg * 100.0, 1))
        ts_baseline_rate.append(round(base_rate * 100.0, 1))
        wracc_ts.append(((cov_ts_val / n_ts) * (rate_in_sg - base_rate)) if n_ts else 0.0)
        try:
            members_ts.append("|".join(map(str, Xts.index[mts].tolist())))
        except Exception:
            members_ts.append("")
    qvals = _fdr_bh(pvals, alpha=alpha)
    qvals_ts = _fdr_bh(pvals_ts, alpha=alpha)
    out = df_rules.copy()
    # Overwrite significance with TEST-based values for final pruning/selection
    out['p_value'] = pvals_ts
    out['q_value'] = qvals_ts
    out['lift_train'] = np.round(lifts_tr, 3)
    out['lift_test'] = np.round(lifts_ts, 3)
    out['coverage_train'] = cov_tr
    out['coverage_test'] = cov_ts
    # Overwrite key metrics to reflect TEST population only (unseen battleground)
    out['coverage_pct'] = cov_ts_pct
    out['target_share'] = ts_target_share
    out['baseline_rate'] = ts_baseline_rate
    out['quality_WRAcc'] = wracc_ts
    out['members'] = members_ts
    keep = (out['q_value'] <= alpha) & (out['lift_test'] >= 1.0) & (out['lift_test'] >= 0.7 * out['lift_train'])
    
    out = out.loc[keep].reset_index(drop=True)
    out['lift'], out['coverage'] = out['lift_test'], out['coverage_test']
    return out


def _run_subgroup_discovery(X_features: pd.DataFrame, y_target: pd.Series, max_depth: int, config) -> pd.DataFrame:
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
    max_candidates = int(getattr(config, 'SUBGROUP_MAX_CANDIDATES', 200))
    task = ps.SubgroupDiscoveryTask(
        data,
        target,
        searchspace,
        result_set_size=max_candidates,
        depth=max_depth,
        qf=ps.WRAccQF(),
        min_quality=0.001,
    )
    result = ps.BeamSearch(beam_width=max_candidates).execute(task)
    if not hasattr(result, 'to_dataframe'):
        return pd.DataFrame()
    df = result.to_dataframe()
    if df.empty:
        return pd.DataFrame()
    df = df[df['relative_size_sg'] >= float(getattr(config, 'SUBGROUP_MIN_SUPPORT', 0.01))]
    min_q = float(getattr(config, 'SUBGROUP_MIN_QUALITY', 0.0))
    min_l = float(getattr(config, 'SUBGROUP_MIN_LIFT', 0.0))
    if min_q > 0 or min_l > 0:
        df = df[(df['quality'] >= min_q) | (df['lift'] >= min_l)]
    if df.empty:
        return pd.DataFrame()
    # Redundancy filter
    j_thresh = float(getattr(config, 'SUBGROUP_JACCARD_MAX', 0.8))
    df = df.reset_index(drop=True)
    cover_sets = [set(data.index[_safe_cover_mask(r['subgroup'], data)]) for _, r in df.iterrows()]
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
        member_idx = data.index[_safe_cover_mask(r['subgroup'], data)]
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


def _analyze_differential_failures(cohorts_idx, X_features, config, depth: int):
    analyses = [
        ('SM_miss', 'FN_SM', 'TP_concordant', 'SM False Negatives vs Concordant True Positives'),
        ('SM_false_alarm', 'FP_SM', 'TN_concordant', 'SM False Positives vs Concordant True Negatives'),
        ('NM_miss', 'FN_NM', 'TP_concordant', 'NM False Negatives vs Concordant True Positives'),
        ('NM_false_alarm', 'FP_NM', 'TN_concordant', 'NM False Positives vs Concordant True Negatives'),
    ]
    results = {}
    for key, error_cohort, success_cohort, title in analyses:
        error_idx = cohorts_idx.get(error_cohort, pd.Index([]))
        success_idx = cohorts_idx.get(success_cohort, pd.Index([]))
        pop_index = error_idx.union(success_idx)
        X_pop = X_features.loc[X_features.index.isin(pop_index)]
        y = pd.Series(0, index=X_pop.index)
        y.loc[y.index.isin(error_idx)] = 1
        df = _run_subgroup_discovery(X_pop, y, depth, config)
        results[key] = {
            'title': title,
            'results': df,
            'error_count': len(error_idx),
            'success_count': len(success_idx),
        }
    return results


def _build_discordant_cohorts(artifact, X_index: pd.Index):
    cbp = artifact['cohorts_by_pos']
    idx = {k: X_index.take(np.asarray(v, dtype=int)) for k, v in cbp.items()}
    return {
        'SM_Win_on_Deaths': idx.get('FN_NM', pd.Index([])),
        'NM_Win_on_Deaths': idx.get('FN_SM', pd.Index([])),
        'SM_Win_on_Survivors': idx.get('FP_NM', pd.Index([])),
        'NM_Win_on_Survivors': idx.get('FP_SM', pd.Index([])),
    }


def _run_direct_discordance(X_features, cohorts: dict, config, depth: int):
    def _discover_for_pop(pos_idx: pd.Index, neg_idx: pd.Index):
        pop = pos_idx.union(neg_idx)
        X = X_features.loc[X_features.index.isin(pop)]
        y = pd.Series(0, index=X.index)
        y.loc[y.index.isin(pos_idx)] = 1
        return _run_subgroup_discovery(X, y, depth, config), _run_subgroup_discovery(X, 1 - y, depth, config)

    deaths_sm_df, deaths_nm_df = _discover_for_pop(cohorts['SM_Win_on_Deaths'], cohorts['NM_Win_on_Deaths'])
    surv_sm_df, surv_nm_df = _discover_for_pop(cohorts['SM_Win_on_Survivors'], cohorts['NM_Win_on_Survivors'])

    return {
        'IVB_deaths_SM': deaths_sm_df,
        'IVB_deaths_NM': deaths_nm_df,
        'IVB_survivors_SM': surv_sm_df,
        'IVB_survivors_NM': surv_nm_df,
    }


def _write_combined_report(
    out_path: str,
    depth_to_h2b: dict,
    depth_to_ivb: dict,
    cohorts_idx: dict,
    discordant_cohorts: dict,
    max_depths: list,
):
    lines = []
    lines.append('=' * 80)
    lines.append(f'H2b COMBINED ANALYSIS REPORT (depths={",".join(map(str, max_depths))})')
    lines.append('=' * 80)
    lines.append('')
    lines.append('METHOD: Subgroup Discovery (WRAcc) with restricted clinical search space')
    lines.append('Note: Stage A.1 thresholds selected via Youden on train+val (from H2a artifact).')
    lines.append('')

    # Cohort sizes (differential)
    lines.append('COHORT SIZES — Differential Failures')
    lines.append('-' * 40)
    for name, idx_vals in cohorts_idx.items():
        lines.append(f'  {name}: {len(idx_vals)} patients')
    lines.append('')

    # Cohort sizes (discordant)
    lines.append('COHORT SIZES — Direct Discordance')
    lines.append('-' * 40)
    for name in ['SM_Win_on_Deaths','NM_Win_on_Deaths','SM_Win_on_Survivors','NM_Win_on_Survivors']:
        lines.append(f'  {name}: {len(discordant_cohorts.get(name, []))} patients')
    lines.append('')

    for d in max_depths:
        lines.append('=' * 80)
        lines.append(f'ANALYSIS DEPTH: {d}')
        lines.append('=' * 80)
        lines.append('')

        # Differential failures section
        lines.append('— Differential Failures —')
        h2b = depth_to_h2b.get(d, {})
        order = [
            ('SM_miss', 'SM False Negatives vs Concordant True Positives'),
            ('SM_false_alarm', 'SM False Positives vs Concordant True Negatives'),
            ('NM_miss', 'NM False Negatives vs Concordant True Positives'),
            ('NM_false_alarm', 'NM False Positives vs Concordant True Negatives'),
        ]
        for key, title in order:
            lines.append('-' * 80)
            lines.append(title)
            lines.append('-' * 80)
            meta = h2b.get(key)
            if not meta or meta['results'].empty:
                lines.append('No significant patterns discovered')
                lines.append('')
                continue
            df = meta['results']
            for _, r in df.iterrows():
                lines.append(f"Rule #{int(r['rank'])}: {_normalize_rule_case(r['rule'])}")
                lines.append(f"  WRAcc={r['quality_WRAcc']:.4f}  Lift={r['lift']:.2f}x  Coverage={r['coverage']} ({r['coverage_pct']}%)  Target%={r['target_share']:.1f}%  Baseline%={r['baseline_rate']:.1f}%")
            lines.append('')

        # Direct discordance section
        lines.append('— Direct Discordance —')
        ivb = depth_to_ivb.get(d, {})
        order_ivb = [
            ('IVB_deaths_SM', 'Battleground of the Deceased — SM advantage (is_SM_win=1)'),
            ('IVB_deaths_NM', 'Battleground of the Deceased — NM advantage (is_SM_win=0)'),
            ('IVB_survivors_SM', 'Battleground of the Survivors — SM advantage (is_SM_win=1)'),
            ('IVB_survivors_NM', 'Battleground of the Survivors — NM advantage (is_SM_win=0)'),
        ]
        for key, title in order_ivb:
            lines.append('-' * 80)
            lines.append(title)
            lines.append('-' * 80)
            df = ivb.get(key, pd.DataFrame())
            if df is None or df.empty:
                lines.append('No significant patterns discovered')
                lines.append('')
                continue
            for _, r in df.iterrows():
                lines.append(f"Rule #{int(r['rank'])}: {_normalize_rule_case(r['rule'])}")
                lines.append(f"  WRAcc={r['quality_WRAcc']:.4f}  Lift={r['lift']:.2f}x  Coverage={r['coverage']} ({r['coverage_pct']}%)  Target%={r['target_share']:.1f}%  Baseline%={r['baseline_rate']:.1f}%")
            lines.append('')

    lines.append('=' * 80)
    lines.append('END OF REPORT')
    lines.append('=' * 80)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


_ERR_MAP = {'SM_miss': 'FN_SM', 'SM_false_alarm': 'FP_SM', 'NM_miss': 'FN_NM', 'NM_false_alarm': 'FP_NM'}


def _err_succ_keys(key: str):
    return _ERR_MAP.get(key, key), ('TP_concordant' if 'miss' in key else 'TN_concordant')


def _index_cohorts(cbp: dict, index: pd.Index, safe: bool = False) -> dict:
    n = len(index)
    out = {}
    for name, lst in cbp.items():
        arr = np.asarray(lst, dtype=int)
        if safe:
            arr = arr[(arr >= 0) & (arr < n)]
        out[name] = index.take(arr)
    return out


def _rule_record(family: str, key: str, r: pd.Series, depth_val: int) -> dict:
    return {
        'analysis_family': family,
        'analysis_key': key,
        'rank': int(r.get('rank', 0)),
        'rule_str': r.get('rule', ''),
        'coverage': int(r.get('coverage', 0)),
        'coverage_pct': float(r.get('coverage_pct', 0.0)),
        'lift': float(r.get('lift', 0.0)),
        'quality_WRAcc': float(r.get('quality_WRAcc', 0.0)),
        'target_share': float(r.get('target_share', 0.0)),
        'baseline_rate': float(r.get('baseline_rate', 0.0)),
        'members': r.get('members', ''),
        'source_depth': f'depth_{depth_val}',
        'p_value': float(r.get('p_value', np.nan)),
        'q_value': float(r.get('q_value', np.nan)),
    }


def main():
    parser = argparse.ArgumentParser(description='H2b Combined Analysis: Differential Failures + Direct Discordance')
    parser.add_argument('--config_file', type=str, default=None, help='Path to ConfigH2 .py file (e.g., config_h2_readmin30.py)')
    parser.add_argument('--depths', type=str, default=None, help='Comma-separated list of depths to sweep (e.g., 2,3,4)')
    args = parser.parse_args()

    cfg = _load_config(args.config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Resolve depths
    if args.depths:
        depths = [int(d.strip()) for d in args.depths.split(',') if d.strip()]
    else:
        depths = [int(getattr(cfg, 'SUBGROUP_MAX_DEPTH', 3))]

    # Load artifacts and features
    art_path = os.path.join(cfg.H2A_OUTPUT_DIR, 'h2a_to_h2b_artifact.pkl')
    if not os.path.exists(art_path):
        raise FileNotFoundError(f"Missing H2a artifact: {art_path}")
    with open(art_path, 'rb') as f:
        artifact = pickle.load(f)

    # Prefer engineered phenotypes if available; fallback to numeric (inverse-scaled)
    X_test_pheno = None
    if hasattr(cfg, 'X_TEST_PHENOS_PATH') and os.path.exists(cfg.X_TEST_PHENOS_PATH):
        try:
            X_test_pheno = pd.read_pickle(cfg.X_TEST_PHENOS_PATH)
        except Exception:
            X_test_pheno = None
    if isinstance(X_test_pheno, pd.DataFrame) and not X_test_pheno.empty:
        X_features = X_test_pheno
    else:
        with open(cfg.X_TEST_NUM_PATH, 'rb') as f:
            X_test_num = pickle.load(f)
        if not isinstance(X_test_num, pd.DataFrame):
            X_test_num = pd.DataFrame(X_test_num)
        try:
            with open(getattr(cfg, 'SCALER_PATH'), 'rb') as f:
                _sc = pickle.load(f)
            if hasattr(_sc, 'feature_names_in_'):
                X_test_num[_sc.feature_names_in_] = _sc.inverse_transform(X_test_num[_sc.feature_names_in_])
            else:
                X_test_num = pd.DataFrame(_sc.inverse_transform(X_test_num), columns=X_test_num.columns, index=X_test_num.index)
        except Exception:
            pass
        X_features = X_test_num

    # Optionally load train+val phenotypes for Stage B.1 (training discovery)
    X_trainval_pheno = None
    if hasattr(cfg, 'X_TRAINVAL_PHENOS_PATH') and os.path.exists(getattr(cfg, 'X_TRAINVAL_PHENOS_PATH')):
        try:
            X_trainval_pheno = pd.read_pickle(getattr(cfg, 'X_TRAINVAL_PHENOS_PATH'))
        except Exception:
            X_trainval_pheno = None

    full_index = pd.Index(X_features.index)
    cohorts_idx = _index_cohorts(artifact['cohorts_by_pos'], full_index)
    discordant_cohorts = _build_discordant_cohorts(artifact, full_index)
    # Optional training discordant cohorts (for Stage B on training)
    train_art = artifact.get('train_artifact') if isinstance(artifact, dict) else None
    full_index_train = None
    train_discordant = None
    if isinstance(train_art, dict) and 'cohorts_by_pos' in train_art and isinstance(X_trainval_pheno, pd.DataFrame) and not X_trainval_pheno.empty:
        full_index_train = pd.Index(X_trainval_pheno.index)
        train_discordant = _build_discordant_cohorts({'cohorts_by_pos': train_art['cohorts_by_pos']}, full_index_train)

    # Accumulate results across depths
    depth_to_h2b = {}
    depth_to_ivb = {}
    final_rows = []

    base_output = cfg.OUTPUT_DIR
    for d in depths:
        cfg.SUBGROUP_MAX_DEPTH = d

        # Differential failures (Stage B.1 on training cohorts if available)
        # Prefer training cohorts if exported; fallback to test cohorts
        train_art = artifact.get('train_artifact') if isinstance(artifact, dict) else None
        if isinstance(train_art, dict) and 'cohorts_by_pos' in train_art:
            X_for_train = X_trainval_pheno if isinstance(X_trainval_pheno, pd.DataFrame) and not X_trainval_pheno.empty else X_features
            tr_idx = _index_cohorts(train_art['cohorts_by_pos'], pd.Index(X_for_train.index), safe=True)
            h2b_results = _analyze_differential_failures(tr_idx, X_for_train, cfg, d)
        else:
            h2b_results = _analyze_differential_failures(cohorts_idx, X_features, cfg, d)
        depth_to_h2b[d] = h2b_results
        rows_this_depth = []
        for key, meta in h2b_results.items():
            df = meta['results']
            if df is None or df.empty:
                continue
            # Stage B.2: Statistical and generalizability pruning on top of discovery
            tr_err_key, tr_succ_key = _err_succ_keys(key)
            if isinstance(train_art, dict) and 'cohorts_by_pos' in train_art:
                tr_err, tr_succ = tr_idx.get(tr_err_key, pd.Index([])), tr_idx.get(tr_succ_key, pd.Index([]))
                X_train_for_prune = X_for_train
            else:
                tr_err, tr_succ = cohorts_idx.get(tr_err_key, pd.Index([])), cohorts_idx.get(tr_succ_key, pd.Index([]))
                X_train_for_prune = X_features
            te_err, te_succ = cohorts_idx.get(tr_err_key, pd.Index([])), cohorts_idx.get(tr_succ_key, pd.Index([]))
            df_pruned = _augment_and_prune_rules(
                analysis_key=key,
                df_rules=df,
                X_train=X_train_for_prune,
                train_error_idx=tr_err,
                train_success_idx=tr_succ,
                X_test=X_features,
                test_error_idx=te_err,
                test_success_idx=te_succ,
                alpha=0.05,
            )
            for _, r in df_pruned.iterrows():
                rec = _rule_record('H2b_differential', key, r, d)
                final_rows.append(rec); rows_this_depth.append(rec)

        # Direct discordance: prefer training cohorts if available (Stage B.1 on training), else test
        if train_discordant is not None and isinstance(X_trainval_pheno, pd.DataFrame) and not X_trainval_pheno.empty:
            ivb_results = _run_direct_discordance(X_trainval_pheno, train_discordant, cfg, d)
            _ivb_train_used = True
        else:
            ivb_results = _run_direct_discordance(X_features, discordant_cohorts, cfg, d)
            _ivb_train_used = False
        depth_to_ivb[d] = ivb_results
        for key, df in ivb_results.items():
            if df is None or df.empty:
                continue
            def _map_sets(k: str, use_train: bool):
                pos, neg = ('SM_Win_on_Deaths','NM_Win_on_Deaths') if 'deaths' in k.lower() else ('SM_Win_on_Survivors','NM_Win_on_Survivors')
                if k.endswith('_NM'):
                    pos, neg = neg, pos
                src = train_discordant if use_train and train_discordant is not None else discordant_cohorts
                return src[pos], src[neg]

            tr_err, tr_succ = _map_sets(key, _ivb_train_used)
            te_err, te_succ = _map_sets(key, False)
            X_tr = X_trainval_pheno if _ivb_train_used else X_features
            df_pruned = _augment_and_prune_rules(
                analysis_key=key,
                df_rules=df,
                X_train=X_tr,
                train_error_idx=tr_err,
                train_success_idx=tr_succ,
                X_test=X_features,
                test_error_idx=te_err,
                test_success_idx=te_succ,
                alpha=0.05,
            )
            for _, r in df_pruned.iterrows():
                rec = _rule_record('IVB_discordance', key, r, d)
                final_rows.append(rec); rows_this_depth.append(rec)

        # Minimal console output per depth
        kept_counts = sum(0 if meta['results'] is None or meta['results'].empty else meta['results'].shape[0] for meta in h2b_results.values())
        ivb_counts = sum(0 if df is None or df.empty else df.shape[0] for df in ivb_results.values())
        print(f"[depth={d}] H2b patterns kept: {kept_counts}; IVB patterns kept: {ivb_counts}")

        # Write per-depth final_subgroups
        if rows_this_depth:
            pd.DataFrame(rows_this_depth).to_csv(os.path.join(base_output, f'final_subgroups_depth_{d}.csv'), index=False)

    # Write single combined final_subgroups.csv
    if final_rows:
        pd.DataFrame(final_rows).to_csv(os.path.join(base_output, 'final_subgroups.csv'), index=False)

    # Write single combined detailed report
    _write_combined_report(
        out_path=os.path.join(base_output, 'h2b_combined_detailed_report.txt'),
        depth_to_h2b=depth_to_h2b,
        depth_to_ivb=depth_to_ivb,
        cohorts_idx=cohorts_idx,
        discordant_cohorts=discordant_cohorts,
        max_depths=depths,
    )


if __name__ == '__main__':
    main()