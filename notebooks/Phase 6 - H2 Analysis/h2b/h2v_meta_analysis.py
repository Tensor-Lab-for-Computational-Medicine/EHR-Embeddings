import os
import pickle
import argparse
import importlib.util
import numpy as np
import re
import pandas as pd
from scipy.stats import mannwhitneyu


def compute_meta_features(X: pd.DataFrame) -> pd.DataFrame:
    base = X.copy()
    meta = pd.DataFrame(index=base.index)
    count_cols = [c for c in base.columns if c.endswith('_mean_count') or c.endswith('_mean_count_6h')]
    if count_cols:
        counts = base[count_cols].clip(lower=0).round()
        meta['num_features_measured'] = (counts > 0).sum(axis=1)
        meta['zero_count_features'] = (counts == 0).sum(axis=1)
        meta['total_measurement_events'] = counts.sum(axis=1)
        fam = [re.sub(r'_mean_count(_6h)?$','', c) for c in count_cols]
        fam_map = {}
        for col, f in zip(count_cols, fam):
            fam_map.setdefault(f, []).append(col)
        fam_any = pd.DataFrame({f: (counts[cols] > 0).any(axis=1) for f, cols in fam_map.items()}, index=counts.index)
        meta['unique_feature_families_measured'] = fam_any.sum(axis=1)
        meta['prop_zero_count_features'] = meta['zero_count_features'] / float(len(count_cols))
    else:
        meta['num_features_measured'] = 0
        meta['zero_count_features'] = 0
        meta['total_measurement_events'] = 0
        meta['unique_feature_families_measured'] = 0
        meta['prop_zero_count_features'] = 0.0
    stddev_cols = [c for c in base.columns if 'stddev' in c]
    slope_cols = [c for c in base.columns if 'slope' in c]
    meta['aggregate_stddev'] = base[stddev_cols].mean(axis=1) if stddev_cols else np.nan
    meta['aggregate_slope'] = base[slope_cols].mean(axis=1) if slope_cols else np.nan
    meta['imputation_proportion'] = 0.0
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
    # Prefer root OUTPUT_DIR/final_subgroups.csv (combined output supports both phases)
    root_path = os.path.join(cfg.OUTPUT_DIR, 'final_subgroups.csv')
    if os.path.exists(root_path):
        return root_path
    # Legacy fallbacks by phase
    if phase.lower() == 'iv':
        phase_dir = os.path.join(cfg.OUTPUT_DIR, 'phase_iv')
        filename = 'final_subgroups.csv'
    elif phase.lower() == 'ivb':
        phase_dir = os.path.join(cfg.OUTPUT_DIR, 'phase_ivb')
        filename = 'final_subgroups_ivb.csv'
    else:
        raise ValueError('phase must be one of: IV, IVB')
    if os.path.isdir(phase_dir):
        for name in sorted(os.listdir(phase_dir)):
            p = os.path.join(phase_dir, name, filename)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f'{filename} not found in OUTPUT_DIR or {os.path.basename(phase_dir)} subfolders')


def main():
    parser = argparse.ArgumentParser(description='Phase V meta-analysis (reusable)')
    parser.add_argument('--config_file', type=str, default=None, help='Path to ConfigH2 .py file (e.g., config_h2_readmin30.py). Defaults to morthosp config if omitted.')
    parser.add_argument('--final_subgroups_path', type=str, default=None, help='Optional path to final_subgroups.csv or final_subgroups_ivb.csv; if omitted, auto-discover.')
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
    # Revert standardized features to original clinical scale for interpretability
    with open(r'D:\Projects\EHR Embeddings\notebooks\Phase 1 and 2\phase_1_outputs\scaler.pkl', 'rb') as f:
        _sc = pickle.load(f)
    if hasattr(_sc, 'feature_names_in_'):
        X_test[_sc.feature_names_in_] = _sc.inverse_transform(X_test[_sc.feature_names_in_])
    else:
        X_test = pd.DataFrame(_sc.inverse_transform(X_test), columns=X_test.columns, index=X_test.index)

    meta = compute_meta_features(X_test)
    # Debug: log representative column names and values to guide meta-feature definitions
    debug_path = os.path.join(out_dir, 'phase_v_meta_debug.txt')
    count_cols = [c for c in X_test.columns if 'count' in c]
    stddev_cols = [c for c in X_test.columns if 'stddev' in c]
    slope_cols = [c for c in X_test.columns if 'slope' in c]
    sample_cols_count = count_cols[:5]
    sample_cols_std = stddev_cols[:5]
    sample_cols_slope = slope_cols[:5]
    sample_idx = list(X_test.index[:3])
    with open(debug_path, 'w', encoding='utf-8') as df:
        df.write(f"n_total_columns={X_test.shape[1]}\n")
        df.write(f"count_cols_first5={sample_cols_count}\n")
        df.write(f"stddev_cols_first5={sample_cols_std}\n")
        df.write(f"slope_cols_first5={sample_cols_slope}\n")
        for ix in sample_idx:
            row = X_test.loc[ix]
            df.write(f"\nindex={ix}\n")
            if sample_cols_count:
                df.write(f"counts={row[sample_cols_count].to_dict()}\n")
            if sample_cols_std:
                df.write(f"stddev={row[sample_cols_std].to_dict()}\n")
            if sample_cols_slope:
                df.write(f"slope={row[sample_cols_slope].to_dict()}\n")

    y_true = np.asarray(art['y_true']).astype(int)
    # Align cohort positions to the actual index labels of X_test to avoid boolean length mismatches
    full_index = pd.Index(X_test.index)
    cohorts_idx = {k: full_index.take(np.asarray(v, dtype=int)) for k, v in art['cohorts_by_pos'].items()}

    # Load vetted Phase IV/IVB subgroups
    final_path = args.final_subgroups_path or _discover_final_subgroups(cfg, args.phase)
    final_subgroups = pd.read_csv(final_path)
    # If using combined outputs, filter by phase
    if 'analysis_family' in final_subgroups.columns:
        if args.phase.lower() == 'iv':
            final_subgroups = final_subgroups[final_subgroups['analysis_family'] == 'H2b_differential'].copy()
        else:
            final_subgroups = final_subgroups[final_subgroups['analysis_family'] == 'IVB_discordance'].copy()
    idx_map = pd.Series(X_test.index, index=X_test.index.astype(str))

    rows = []
    for _, r in final_subgroups.iterrows():
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

        # Members of the vetted subgroup (prefer explicit membership from Phase IV)
        mem_str = r.get('members') if 'members' in r else None
        if isinstance(mem_str, str) and mem_str:
            keys = [m for m in mem_str.split('|') if m in idx_map.index]
            members = pd.Index(idx_map.loc[keys].values)
        else:
            try:
                members = X_test.query(rule).index
            except Exception as e:
                print(f"Could not apply rule: {rule}. Error: {e}")
                continue
        if args.phase.lower() == 'iv':
            err_members = members.intersection(cohorts_idx[err_key])
            suc_members = cohorts_idx[suc_key]
        else:
            # Compare within the same battleground: advantaged vs opposite
            err_members = members.intersection(win_cohort)
            suc_members = members.intersection(opp_cohort)

        meta_feats = [c for c in [
            'num_features_measured', 'total_measurement_events', 'unique_feature_families_measured',
            'zero_count_features', 'prop_zero_count_features', 'aggregate_stddev', 'aggregate_slope',
            'imputation_proportion'
        ] if c in meta.columns]
        for feat in meta_feats:
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
    if not meta_df.empty:
        meta_df.to_csv(os.path.join(out_dir, 'phase_v_meta_results.csv'), index=False)

    # Minimal report
    lines = []
    lines.append('PHASE V META-ANALYSIS')
    if not meta_df.empty:
        sig = meta_df[meta_df['p_value'] < 0.05]
        lines.append(f"Significant findings: {len(sig)} (p<0.05)")
    with open(os.path.join(out_dir, 'phase_v_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()