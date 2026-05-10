import os
import pickle
import argparse
import importlib.util
import numpy as np
import re
import pandas as pd
from scipy.stats import mannwhitneyu


def compute_meta_features(X: pd.DataFrame) -> pd.DataFrame:
    counts = [c for c in X.columns if c.endswith('_mean_count') or c.endswith('_mean_count_6h')]
    meta = pd.DataFrame(index=X.index)
    if counts:
        C = X[counts].clip(lower=0)
        fam_map = {}
        for c in counts:
            fam_map.setdefault(re.sub(r'_mean_count(_6h)?$', '', c), []).append(c)
        fam_max = pd.DataFrame({f: C[v].max(axis=1) for f, v in fam_map.items()})
        # Density: sum of per-family max across cadences (no double-counting)
        meta['total_measurement_events'] = fam_max.sum(axis=1)
        meta['unique_feature_families_measured'] = pd.DataFrame({f: (C[v] > 0).any(axis=1) for f, v in fam_map.items()}).sum(axis=1)
        z = (fam_max == 0).sum(axis=1)
        # Scale-free imputation burden
        meta['imputation_proportion'] = z / fam_max.shape[1]
    # Volatility: per-family averages to avoid overweighting families with more derived columns
    stddev_cols = [c for c in X.columns if 'stddev' in c]
    if stddev_cols:
        fam_map_std = {}
        for c in stddev_cols:
            fam = re.sub(r'_stddev.*$', '', c)
            fam_map_std.setdefault(fam, []).append(c)
        fam_std_means = pd.DataFrame({f: X[v].mean(axis=1) for f, v in fam_map_std.items()})
        meta['aggregate_stddev'] = fam_std_means.mean(axis=1)
    else:
        meta['aggregate_stddev'] = np.nan
    # Trend magnitude: prefer 24h slopes else 6h; use absolute value, then per-family mean
    slope_cols = [c for c in X.columns if 'slope' in c]
    if slope_cols:
        fam_map_slope_any = {}
        fam_map_slope_24 = {}
        fam_map_slope_6 = {}
        for c in slope_cols:
            fam = re.sub(r'_slope.*$', '', c)
            fam_map_slope_any.setdefault(fam, []).append(c)
            if re.search(r'_slope_24h', c):
                fam_map_slope_24.setdefault(fam, []).append(c)
            elif re.search(r'_slope_6h', c):
                fam_map_slope_6.setdefault(fam, []).append(c)
        fam_slope_vals = {}
        for f in fam_map_slope_any.keys():
            cols = fam_map_slope_24.get(f) or fam_map_slope_6.get(f) or fam_map_slope_any.get(f)
            fam_slope_vals[f] = X[cols].abs().mean(axis=1)
        fam_slope_means = pd.DataFrame(fam_slope_vals)
        meta['aggregate_slope'] = fam_slope_means.mean(axis=1)
    else:
        meta['aggregate_slope'] = np.nan
    meta['unique_feature_count'] = meta.get('unique_feature_families_measured', pd.Series(0, index=X.index))

    # Temporal Concentration: proportion of events in the final 6h vs total 24h
    counts_6h = [c for c in X.columns if c.endswith('_mean_count_6h')]
    counts_all = [c for c in X.columns if c.endswith('_mean_count')]
    if counts_6h and counts_all:
        sum_6h = X[counts_6h].clip(lower=0).sum(axis=1)
        sum_all = X[counts_all].clip(lower=0).sum(axis=1)
        # Avoid division by zero
        meta['temporal_concentration'] = np.where(sum_all > 0, sum_6h / sum_all, 0.0)
    else:
        meta['temporal_concentration'] = np.nan

    return meta


# Phase V does not discover new subgroups; it analyzes vetted Phase IV subgroups.


def mannwhitney(a: pd.Series, b: pd.Series):
    a = a.dropna()
    b = b.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan, np.nan
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    return float(stat), float(p), float(np.median(a)), float(np.median(b))


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


def _discover_final_archetypes(cfg, phase: str) -> str:
    # Prefer root OUTPUT_DIR/final_archetypes.csv (IV) or final_archetypes_ivb.csv (IVB)
    if phase.lower() == 'iv':
        root_file = 'final_archetypes.csv'
        filename = 'final_archetypes.csv'
        phase_dir = os.path.join(cfg.OUTPUT_DIR, 'phase_iv')
    elif phase.lower() == 'ivb':
        root_file = 'final_archetypes_ivb.csv'
        filename = 'final_archetypes_ivb.csv'
        phase_dir = os.path.join(cfg.OUTPUT_DIR, 'phase_ivb')
    else:
        raise ValueError('phase must be one of: IV, IVB')
    root_path = os.path.join(cfg.OUTPUT_DIR, root_file)
    if os.path.exists(root_path):
        return root_path
    # Legacy fallbacks by phase
    if os.path.isdir(phase_dir):
        for name in sorted(os.listdir(phase_dir)):
            p = os.path.join(phase_dir, name, filename)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f'{filename} not found in OUTPUT_DIR or {os.path.basename(phase_dir)} subfolders')


def _bh_fdr(pvals: pd.Series) -> pd.Series:
    # Benjamini-Hochberg (BH) adjusted p-values; ensures q >= p and monotone with rank
    s = pvals.copy()
    not_na = s.dropna()
    m = len(not_na)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if m == 0:
        return out
    order = not_na.sort_values(kind='mergesort').index
    pv = not_na.loc[order].to_numpy(dtype=float)
    ranks = np.arange(1, m + 1, dtype=float)
    adj = pv * (m / ranks)
    # enforce monotone non-decreasing with rank via reverse cumulative min
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out.loc[order] = adj
    return out


def main():
    parser = argparse.ArgumentParser(description='Phase V meta-analysis (reusable)')
    parser.add_argument('--config_file', type=str, default=None, help='Path to ConfigH2 .py file (e.g., config_h2_readmin30.py). Defaults to morthosp config if omitted.')
    parser.add_argument('--final_archetypes_path', type=str, default=None, help='Optional path to final_archetypes.csv or final_archetypes_ivb.csv; if omitted, auto-discover.')
    # Subgroup fallback removed for concise script
    parser.add_argument('--phenotypes_test_path', type=str, default=None, help='Path to X_test_phenotypes.pkl; if omitted, attempts default artifact location.')
    parser.add_argument('--phase', type=str, default='IV', help='Which phase subgroups to analyze: IV or IVB')
    parser.add_argument('--scaler_path', type=str, default=None, help='Path to the scaler.pkl used in Phase 1 to inverse-transform features to raw units.')
    parser.add_argument('--debug_fdr', action='store_true', help='Enable verbose debug logging for BH-FDR calculations')
    args = parser.parse_args()

    cfg = _load_config(args.config_file)
    np.random.seed(42)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    out_dir = os.path.join(cfg.OUTPUT_DIR, 'phase_v_meta')
    os.makedirs(out_dir, exist_ok=True)

    art_path = os.path.join(cfg.H2A_OUTPUT_DIR, 'h2a_to_h2b_artifact.pkl')
    try:
        with open(art_path, 'rb') as f:
            art = pickle.load(f)
    except Exception:
        art = pd.read_pickle(art_path)

    try:
        with open(cfg.X_TEST_NUM_PATH, 'rb') as f:
            X_test = pickle.load(f)
    except Exception:
        X_test = pd.read_pickle(cfg.X_TEST_NUM_PATH)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # --- Inverse Scaling to Raw Units ---
    if args.scaler_path:
        print(f"Inverse scaling X_test using {args.scaler_path}...")
        try:
            with open(args.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            # Only transform columns present in scaler
            feature_names = scaler.get_feature_names_out()
            numeric_cols = [c for c in feature_names if c in X_test.columns]
            if numeric_cols:
                X_test[numeric_cols] = scaler.inverse_transform(X_test[numeric_cols])
                # Special handling for counts: ensure non-negative and integer-interpretable
                count_cols = [c for c in numeric_cols if '_count' in c]
                for c in count_cols:
                    X_test[c] = X_test[c].clip(lower=0).round()
                print(f"Successfully inverse scaled {len(numeric_cols)} numeric columns.")
        except Exception as e:
            print(f"Warning: Inverse scaling failed: {e}. Proceeding with scaled units.")

    meta = compute_meta_features(X_test)
    # character counts logic removed
    # Align cohort positions to the actual index labels of X_test to avoid boolean length mismatches
    full_index = pd.Index(X_test.index)
    cohorts_idx = {k: full_index.take(np.asarray(v, dtype=int)) for k, v in art['cohorts_by_pos'].items()}

    # Load vetted Phase IV/IVB FINAL ARCHETYPES (preferred)
    final_path = args.final_archetypes_path or _discover_final_archetypes(cfg, args.phase)
    final_df = pd.read_csv(final_path)
    # If using combined outputs, filter by phase family
    if 'analysis_family' in final_df.columns:
        if args.phase.lower() == 'iv':
            final_df = final_df[final_df['analysis_family'] == 'H2b_differential'].copy()
        else:
            final_df = final_df[final_df['analysis_family'] == 'IVB_discordance'].copy()
    # idx_map will be built after phenotype load to robustly use ICU stay ids

    rule_col = 'rule_str' if 'rule_str' in final_df.columns else 'rule'

    # Helper to fetch archetype context for reporting
    def _ctx(analysis_key: str, rule_str: str) -> str:
        try:
            m = final_df[(final_df['analysis_key'] == analysis_key) & (final_df[rule_col] == rule_str)]
            if m.empty:
                return ''
            r = m.iloc[0]
            cov = r['coverage'] if 'coverage' in r else np.nan
            covp = r['coverage_pct'] if 'coverage_pct' in r else np.nan
            lift = r['lift'] if 'lift' in r else np.nan
            ts = r['target_share'] if 'target_share' in r else np.nan
            br = r['baseline_rate'] if 'baseline_rate' in r else np.nan
            rk = r['rank'] if 'rank' in r else np.nan
            return f"(rank={rk}, coverage={cov} ({covp}%), lift={lift}, target={ts} vs base={br})"
        except Exception:
            return ''

    # Load phenotypes (used to evaluate rule membership)
    phenos_path = args.phenotypes_test_path
    if phenos_path is None:
        # Default artifact location: parent directory of h2b → feature_engineering/artifacts/X_test_phenotypes.pkl
        base_dir = os.path.dirname(os.path.abspath(__file__))
        phenos_path = os.path.abspath(os.path.join(base_dir, '..', 'feature_engineering', 'artifacts', 'X_test_phenotypes.pkl'))
    try:
        with open(phenos_path, 'rb') as f:
            phenos = pickle.load(f)
    except Exception:
        phenos = pd.read_pickle(phenos_path)
    if not isinstance(phenos, pd.DataFrame):
        phenos = pd.DataFrame(phenos)
    # Align phenotype index to X_test and build an icustay id mapping
    if not phenos.index.equals(X_test.index):
        try:
            phenos = phenos.loc[X_test.index]
        except Exception:
            phenos.index = X_test.index
    # Prefer aligned ICU IDs from config artifact if available
    try:
        try:
            with open(cfg.ICUSTAY_IDS_TEST_PATH, 'rb') as f:
                ids_loaded = pickle.load(f)
        except Exception:
            ids_loaded = pd.read_pickle(cfg.ICUSTAY_IDS_TEST_PATH)
        if len(ids_loaded) == X_test.shape[0]:
            icu_ids = pd.Series(pd.Index(ids_loaded).astype(str).tolist(), index=X_test.index)
        else:
            raise ValueError('icu id length mismatch')
    except Exception:
        # Fallbacks
        if 'icustay_id' in phenos.columns:
            icu_ids = phenos['icustay_id'].astype(str)
        elif isinstance(X_test, pd.DataFrame) and 'icustay_id' in X_test.columns:
            icu_ids = X_test['icustay_id'].astype(str)
        else:
            icu_ids = pd.Series(X_test.index.astype(str), index=X_test.index)
    # Map from ICU id string -> row label in X_test for member resolution below
    idx_map = pd.Series(X_test.index, index=icu_ids.values)
    # character counts logic removed

    # (diagnostics removed for concision)

    # Rule parser: supports boolean equality, categorical equality, and interval syntax: feat: [low:high[
    interval_pat = re.compile(r"^([A-Za-z0-9_]+)\s*:\s*\[\s*([^:\]]+)\s*:\s*([^\]\[]+)\[$")
    eq_pat = re.compile(r"^([A-Za-z0-9_]+)\s*==\s*(True|False|'[^']+'|\"[^\"]+\")$")

    def _eval_rule_on_phenos(rule_str: str) -> pd.Index:
        parts = [p.strip() for p in re.split(r"\s+AND\s+", str(rule_str)) if p and p.strip()]
        if not parts:
            return phenos.index
        mask = pd.Series(True, index=phenos.index)
        for tok in parts:
            m = interval_pat.match(tok)
            if m:
                col, low_s, high_s = m.group(1), m.group(2), m.group(3)
                if col not in phenos.columns:
                    return phenos.index[[]]
                try:
                    low = float(str(low_s).strip())
                    high = float(str(high_s).strip())
                except Exception:
                    return phenos.index[[]]
                mask &= (phenos[col] >= low) & (phenos[col] < high)
                continue
            m = eq_pat.match(tok)
            if m:
                col, val_s = m.group(1), m.group(2)
                if col not in phenos.columns:
                    return phenos.index[[]]
                if val_s in ('True', 'False'):
                    target = (val_s == 'True')
                    # Allow numeric 0/1
                    col_series = phenos[col]
                    if col_series.dtype == bool:
                        mask &= (col_series == target)
                    else:
                        mask &= (col_series.astype(float) > 0.5) if target else (col_series.astype(float) <= 0.5)
                else:
                    # strip quotes
                    if val_s.startswith("'") and val_s.endswith("'"):
                        label = val_s[1:-1]
                    elif val_s.startswith('"') and val_s.endswith('"'):
                        label = val_s[1:-1]
                    else:
                        label = val_s
                    mask &= (phenos[col].astype(str) == label)
                continue
            # Fallback: unsupported token → empty
            return phenos.index[[]]
        return phenos.index[mask.values]

    rows = []
    for _, r in final_df.iterrows():
        key = r['analysis_key']
        rule = r[rule_col]
        if args.phase.lower() == 'iv':
            title_map = {
                'SM_miss': 'SM False Negatives vs Concordant True Positives',
                'SM_false_alarm': 'SM False Positives vs Concordant True Negatives',
                'NM_miss': 'NM False Negatives vs Concordant True Positives',
                'NM_false_alarm': 'NM False Positives vs Concordant True Negatives',
            }
            err_key_map = {
                'SM_miss': 'FN_SM', 'SM_false_alarm': 'FP_SM', 'NM_miss': 'FN_NM', 'NM_false_alarm': 'FP_NM'
            }
            suc_key_map = {
                'SM_miss': 'TP_concordant', 'SM_false_alarm': 'TN_concordant', 'NM_miss': 'TP_concordant', 'NM_false_alarm': 'TN_concordant'
            }
            if key not in err_key_map:
                continue
            err_key = err_key_map[key]
            suc_key = suc_key_map[key]
            title = title_map.get(key, key)
        else:
            # IVB: within-battleground comparison of advantaged vs opposite model
            # Build discordant battleground cohorts from artifact mapping
            sm_win_deaths = cohorts_idx.get('FN_NM', pd.Index([]))
            nm_win_deaths = cohorts_idx.get('FN_SM', pd.Index([]))
            sm_win_surv = cohorts_idx.get('FP_NM', pd.Index([]))
            nm_win_surv = cohorts_idx.get('FP_SM', pd.Index([]))
            m = re.match(r'^IVB_(deaths|survivors)_(SM|NM)$', str(key))
            if not m:
                continue
            domain, side = m.group(1), m.group(2)
            if domain == 'deaths':
                win_cohort = sm_win_deaths if side == 'SM' else nm_win_deaths
                opp_cohort = nm_win_deaths if side == 'SM' else sm_win_deaths
                title = f"Battleground of the Deceased — {'SM' if side=='SM' else 'NM'} advantage"
            else:
                win_cohort = sm_win_surv if side == 'SM' else nm_win_surv
                opp_cohort = nm_win_surv if side == 'SM' else sm_win_surv
                title = f"Battleground of the Survivors — {'SM' if side=='SM' else 'NM'} advantage"

        # Members of the vetted subgroup: evaluate rule on phenotypes
        mem_str = r.get('members', r.get('member'))
        if isinstance(mem_str, str) and mem_str:
            keys = [m for m in mem_str.split('|') if m in idx_map.index]
            members = pd.Index(idx_map.loc[keys].values)
        else:
            members = _eval_rule_on_phenos(rule)
            if len(members) == 0:
                print(f"Could not apply rule: {rule}. Error: parsed to empty set or unsupported syntax")
                continue
        if args.phase.lower() == 'iv':
            # Phase IV: compare within the vetted subgroup as well (aligns with IVB design)
            err_members = members.intersection(cohorts_idx[err_key])
            suc_members = members.intersection(cohorts_idx[suc_key])
        else:
            # Compare within the same battleground: advantaged vs opposite
            err_members = members.intersection(win_cohort)
            suc_members = members.intersection(opp_cohort)

        # (debug metrics removed)

        # Analyze ONLY the pre-specified meta-features per Phase V plan
        meta_feats = [c for c in [
            'total_measurement_events',        # density
            'unique_feature_count',            # density
            'temporal_concentration',          # temporal
            'aggregate_stddev',                # volatility
            'aggregate_slope',                 # volatility (magnitude, 24h preferred)
            'imputation_proportion'            # imputation (scale-free)
        ] if c in meta.columns]
        for feat in meta_feats:
            if feat not in meta.columns:
                continue
            a = meta.loc[err_members, feat]
            b = meta.loc[suc_members, feat]
            stat, p, med_a, med_b = mannwhitney(a, b)
            rows.append({
                'analysis': key,
                'title': title,
                'rule': rule,
                'meta_feature': feat,
                'u_stat': stat,
                'p_value': p,
                'median_subgroup_error': med_a,
                'median_concordant_success': med_b,
                'effect_size': (med_a - med_b) if pd.notna(med_a) and pd.notna(med_b) else np.nan,
                'n_error_in_subgroup': int(a.shape[0]),
                'n_success_concordant': int(b.shape[0]),
            })

    meta_df = pd.DataFrame(rows)
    # Deduplicate exact duplicates per (analysis, rule, meta_feature)
    if not meta_df.empty:
        meta_df = meta_df.sort_values(['analysis','rule','meta_feature','p_value']).drop_duplicates(subset=['analysis','rule','meta_feature'], keep='first')
    if not meta_df.empty:
        # Add direction and FDR q-values
        meta_df['direction'] = np.where(meta_df['effect_size'] > 0, 'higher_in_error', np.where(meta_df['effect_size'] < 0, 'higher_in_success', 'no_diff'))
        # Global BH-FDR across all tests (kept for backwards-compatibility)
        meta_df['q_value'] = _bh_fdr(meta_df['p_value'])
        meta_df['significant_p_lt_0_05'] = meta_df['p_value'] < 0.05
        meta_df['significant_q_lt_0_05'] = meta_df['q_value'] < 0.05
        # Per-family BH-FDR within each (analysis, rule, family)
        family_map = {
            'total_measurement_events': 'density',
            'unique_feature_count': 'density',
            'temporal_concentration': 'temporal',
            'aggregate_stddev': 'volatility',
            'aggregate_slope': 'volatility',
            'imputation_proportion': 'imputation',
        }
        meta_df['family'] = meta_df['meta_feature'].map(family_map).fillna('other')
        # Per-(analysis,rule,family) BH-FDR. This controls for multiple meta-features within each vetted subgroup.
        meta_df['q_value_family'] = (
            meta_df.groupby(['analysis', 'rule', 'family'], dropna=False)['p_value']
                   .transform(_bh_fdr)
        )
        # Safety: ensure adjusted q-values are never below raw p-values
        meta_df['q_value_family'] = np.maximum(meta_df['q_value_family'], meta_df['p_value'])
        if args.debug_fdr:
            # Debug summary of per-family groups and where q==p
            print("[DEBUG] ----- FDR family grouping diagnostics -----")
            grp_keys = ['analysis', 'rule', 'family']
            gobj = meta_df.groupby(grp_keys, dropna=False)
            counts = gobj['p_value'].size().rename('m')
            equal_mask = (meta_df['q_value_family'] == meta_df['p_value'])
            eq_counts = gobj.apply(lambda d: int((d['q_value_family'] == d['p_value']).sum())).rename('eq_count')
            min_p = gobj['p_value'].min().rename('min_p')
            max_p = gobj['p_value'].max().rename('max_p')
            dbg = pd.concat([counts, eq_counts, min_p, max_p], axis=1).reset_index()
            dbg['eq_rate'] = dbg.apply(lambda r: (r['eq_count'] / r['m']) if r['m'] else np.nan, axis=1)
            dbg['note'] = dbg.apply(lambda r: 'single-test family (m=1)' if r['m'] == 1 else ('all q==p at largest ranks' if r['eq_rate'] > 0 and r['m'] > 1 else ''), axis=1)
            # Print a compact table-like summary
            print(dbg.sort_values(['family','m','eq_rate'], ascending=[True, False, False]).head(50).to_string(index=False))
            # Show an example group with m>1 where any q>p for sanity
            try:
                example_key = next(k for k, d in gobj if gobj.size()[k] > 1)
                ex_df = gobj.get_group(example_key).copy()
                ex_df = ex_df.sort_values('p_value')
                print("[DEBUG] Example group (analysis, rule, family)=", example_key)
                print(ex_df[['meta_feature','p_value','q_value_family','u_stat','n_error_in_subgroup','n_success_concordant']].head(10).to_string(index=False))
            except StopIteration:
                print("[DEBUG] No multi-test families found; equality q==p is expected when m=1.")
        meta_df['significant_q_family_lt_0_05'] = meta_df['q_value_family'] < 0.05
        # Standardized rank effect sizes for comparability across sample sizes
        n1 = meta_df['n_error_in_subgroup'].replace(0, np.nan).astype(float)
        n2 = meta_df['n_success_concordant'].replace(0, np.nan).astype(float)
        denom = n1 * n2
        with np.errstate(invalid='ignore', divide='ignore'):
            a_stat = meta_df['u_stat'] / denom
        meta_df['stochastic_superiority_A'] = a_stat
        meta_df['rank_biserial'] = (2.0 * a_stat) - 1.0
        phase_tag = 'IVB' if args.phase.lower() == 'ivb' else 'IV'
        meta_df['phase'] = phase_tag
        meta_df.to_csv(os.path.join(out_dir, f'phase_v_meta_results_{phase_tag}.csv'), index=False)

    # Detailed text report
    lines = []
    lines.append('PHASE V META-ANALYSIS')
    if not meta_df.empty:
        sig_p = meta_df[meta_df['p_value'] < 0.05]
        sig_q = meta_df[meta_df['q_value'] < 0.05]
        sig_qf = meta_df[meta_df['q_value_family'] < 0.05]
        lines.append(f"Significant tests (p<0.05): {len(sig_p)}")
        lines.append(f"Significant tests (q<0.05 global BH-FDR): {len(sig_q)}")
        lines.append(f"Significant tests (q_family<0.05 per-family BH-FDR): {len(sig_qf)}")
        if not sig_qf.empty:
            fam_counts = sig_qf.groupby('family').size().to_dict()
            fam_s = ', '.join([f"{k}:{v}" for k, v in fam_counts.items()])
            lines.append(f"Significant by family (q_family<0.05): {fam_s}")
            feat_counts = sig_qf['meta_feature'].value_counts().to_dict()
            top_feats = ', '.join([f"{k}:{v}" for k, v in feat_counts.items()])
            lines.append(f"Top meta-features with q_family<0.05: {top_feats}")
            lines.append('')
            lines.append('Per-archetype significant differences (per-family FDR):')
            grp = sig_qf.sort_values(['analysis','rule','q_value_family']).groupby(['analysis','rule'])
            for (analysis_key, rule_str), g in grp:
                ctx = _ctx(analysis_key, rule_str)
                lines.append(f"- {analysis_key} | rule: {rule_str} {ctx}")
                # top-N per archetype for readability
                for _, row in g.head(5).iterrows():
                    lines.append(
                        f"  * {row['meta_feature']} [{row['family']}] ({row['direction']}): med_error={row['median_subgroup_error']:.3f}, med_success={row['median_concordant_success']:.3f}, "
                        f"effect={row['effect_size']:.3f}, p={row['p_value']:.3g}, q_family={row['q_value_family']:.3g}, n_err={row['n_error_in_subgroup']}, n_suc={row['n_success_concordant']}"
                    )
                lines.append('')
    phase_tag = 'IVB' if args.phase.lower() == 'ivb' else 'IV'
    with open(os.path.join(out_dir, f'phase_v_report_{phase_tag}.txt'), 'w', encoding='utf-8') as f:
        # Append concise note explaining when q == p can occur and the safety clamp
        lines.append('')
        lines.append('Notes on BH-FDR (per-family):')
        lines.append('- q_values are Benjamini-Hochberg adjusted within each (analysis, rule, family).')
        lines.append('- When family size m=1, q equals p. For small m, the worst-ranked test can also have q==p due to the monotone step.')
        lines.append('- The implementation enforces q_value_family >= p_value to avoid downward bias.')
        f.write('\n'.join(lines))

    # (diagnostics and debug file output removed)


if __name__ == '__main__':
    main()