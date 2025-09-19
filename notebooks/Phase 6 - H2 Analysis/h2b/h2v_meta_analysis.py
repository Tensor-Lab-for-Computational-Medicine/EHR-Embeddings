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
        meta['total_measurement_events'] = C.sum(axis=1)
        meta['unique_feature_families_measured'] = pd.DataFrame({f: (C[v] > 0).any(axis=1) for f, v in fam_map.items()}).sum(axis=1)
        fam_max = pd.DataFrame({f: C[v].max(axis=1) for f, v in fam_map.items()})
        z = (fam_max == 0).sum(axis=1)
        meta['total_imputation_count'] = z
        meta['imputation_proportion'] = z / fam_max.shape[1]
    stddev_cols = [c for c in X.columns if 'stddev' in c]
    slope_cols = [c for c in X.columns if 'slope' in c]
    meta['aggregate_stddev'] = X[stddev_cols].mean(axis=1) if stddev_cols else np.nan
    meta['aggregate_slope'] = X[slope_cols].mean(axis=1) if slope_cols else np.nan
    meta['unique_feature_count'] = meta.get('unique_feature_families_measured', pd.Series(0, index=X.index))
    # populated later from external text corpus by ICU stay id
    meta['input_character_count'] = np.nan
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


def _discover_final_subgroups(cfg, phase: str) -> str:
    raise FileNotFoundError('Subgroups fallback disabled; use final_archetypes* only')


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
    order = not_na.sort_values().index
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
    args = parser.parse_args()

    cfg = _load_config(args.config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    out_dir = os.path.join(cfg.OUTPUT_DIR, 'phase_v_meta')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(cfg.H2A_OUTPUT_DIR, 'h2a_to_h2b_artifact.pkl'), 'rb') as f:
        art = pickle.load(f)

    with open(cfg.X_TEST_NUM_PATH, 'rb') as f:
        X_test = pickle.load(f)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    # Note: Inverse scaling not required for Phase V meta-features; using current scale

    meta = compute_meta_features(X_test)
    # Compute input_character_count per ICU stay id using external text corpus
    def _build_id_to_path_map(base_dir: str) -> dict:
        mp = {}
        for root, _, files in os.walk(base_dir):
            for name in files:
                if not name.lower().endswith('.txt'):
                    continue
                sid = os.path.splitext(name)[0]
                if sid not in mp:
                    mp[sid] = os.path.join(root, name)
        return mp
    def _count_chars_from_path(path: str) -> float:
        # Fast path: byte size approximates character count for ASCII-heavy clinical text; fallback to decode length
        try:
            size = os.path.getsize(path)
            if size is not None and size >= 0:
                return float(size)
        except Exception:
            pass
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return float(len(f.read()))
        except Exception:
            print(f"[DEBUG] Failed to read text file for character count: {path}")
            return np.nan

    # character counts will be assigned after loading final_archetypes to ensure id alignment
    char_text_dir = r'D:\Projects\EHR Embeddings\notebooks\Phase 3\phase_3_serialized_data\F3_P5'
    print(f"[DEBUG] char_text_dir: {char_text_dir} | exists={os.path.isdir(char_text_dir)}")
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

    # Helper to fetch archetype context for reporting
    def _ctx(analysis_key: str, rule_str: str) -> str:
        try:
            m = final_df[(final_df['analysis_key'] == analysis_key) & (final_df['rule_str'] == rule_str)]
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
    with open(phenos_path, 'rb') as f:
        phenos = pickle.load(f)
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
        with open(cfg.ICUSTAY_IDS_TEST_PATH, 'rb') as f:
            ids_loaded = pickle.load(f)
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
    # Populate input_character_count for ALL X_test rows using ICU stay ids
    if os.path.isdir(char_text_dir):
        id_to_path = _build_id_to_path_map(char_text_dir)
        print(f"[DEBUG] Built ICU ID -> path map: {len(id_to_path)} entries")
        if len(id_to_path) > 0:
            sample_items = list(id_to_path.items())[:3]
            print(f"[DEBUG] Sample id_to_path entries: {sample_items}")
        cache = {}
        missing_sid_count = 0
        nan_read_count = 0
        mapped_values = []
        for sid in icu_ids:
            sid_str = str(sid)
            p = id_to_path.get(sid_str)
            if not p:
                missing_sid_count += 1
                mapped_values.append(np.nan)
                continue
            if p not in cache:
                cache[p] = _count_chars_from_path(p)
            val = cache[p]
            if pd.isna(val):
                nan_read_count += 1
            mapped_values.append(float(val) if pd.notna(val) else np.nan)
        total_ids = len(icu_ids)
        print(f"[DEBUG] ICU IDs total={total_ids} | ids_without_text={missing_sid_count} | read_failures={nan_read_count}")
        if total_ids > 0:
            sample_vals = mapped_values[:5]
            print(f"[DEBUG] Sample input_character_count values (first 5): {sample_vals}")
        meta['input_character_count'] = mapped_values
    else:
        print(f"[DEBUG] Character text directory does not exist. input_character_count remains NaN.")

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
    debug_lines = []
    for _, r in final_df.iterrows():
        key = r['analysis_key']
        rule = r['rule_str']
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

        # Debug: record cohort sizes to verify within-subgroup restriction
        try:
            debug_lines.append(f"{args.phase}|{key}|members={len(members)}|err={len(err_members)}|suc={len(suc_members)}")
        except Exception:
            pass

        # Analyze ONLY the pre-specified meta-features per Phase V plan
        meta_feats = [c for c in [
            'total_measurement_events',        # density
            'unique_feature_count',            # density
            'input_character_count',           # density (from ICU text; may be NaN)
            'aggregate_stddev',                # volatility
            'aggregate_slope',                 # volatility
            'total_imputation_count',          # imputation
            'imputation_proportion'            # imputation
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
        meta_df['q_value'] = _bh_fdr(meta_df['p_value'])
        meta_df['significant_p_lt_0_05'] = meta_df['p_value'] < 0.05
        meta_df['significant_q_lt_0_05'] = meta_df['q_value'] < 0.05
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
        lines.append(f"Significant tests (p<0.05): {len(sig_p)}")
        lines.append(f"Significant tests (q<0.05 BH-FDR): {len(sig_q)}")
        if not sig_q.empty:
            counts = sig_q['meta_feature'].value_counts().to_dict()
            top_feats = ', '.join([f"{k}:{v}" for k, v in counts.items()])
            lines.append(f"Top meta-features with q<0.05: {top_feats}")
            lines.append('')
            lines.append('Per-archetype significant differences:')
            grp = sig_q.sort_values(['analysis','rule','q_value']).groupby(['analysis','rule'])
            for (analysis_key, rule_str), g in grp:
                ctx = _ctx(analysis_key, rule_str)
                lines.append(f"- {analysis_key} | rule: {rule_str} {ctx}")
                # top-N per archetype for readability
                for _, row in g.head(5).iterrows():
                    lines.append(
                        f"  * {row['meta_feature']} ({row['direction']}): med_error={row['median_subgroup_error']:.3f}, med_success={row['median_concordant_success']:.3f}, "
                        f"effect={row['effect_size']:.3f}, p={row['p_value']:.3g}, q={row['q_value']:.3g}, n_err={row['n_error_in_subgroup']}, n_suc={row['n_success_concordant']}"
                    )
                lines.append('')
    phase_tag = 'IVB' if args.phase.lower() == 'ivb' else 'IV'
    with open(os.path.join(out_dir, f'phase_v_report_{phase_tag}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # Emit debug cohort sizes
    if debug_lines:
        with open(os.path.join(out_dir, f'phase_v_meta_debug.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(debug_lines))


if __name__ == '__main__':
    main()