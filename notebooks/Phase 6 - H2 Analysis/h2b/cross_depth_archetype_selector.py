import os
import argparse
import importlib.util
import pandas as pd
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


def _count_conditions(rule_str: str) -> int:
    s = str(rule_str).lower()
    if ' and ' in s:
        return s.count(' and ') + 1
    if ' && ' in s:
        return s.count(' && ') + 1
    if ' & ' in s:
        return s.count(' & ') + 1
    return 1


def _load_all_candidates(base_dir: str, phase: str) -> pd.DataFrame:
    rows = []
    if phase.upper() == 'IV':
        # New combined outputs first
        combined_path = os.path.join(base_dir, 'final_subgroups.csv')
        if os.path.exists(combined_path):
            df = pd.read_csv(combined_path)
            if 'analysis_family' in df.columns:
                df_iv = df[df['analysis_family'] == 'H2b_differential']
            else:
                df_iv = df[df['analysis_key'].isin(['SM_miss','SM_false_alarm','NM_miss','NM_false_alarm'])]
            if not df_iv.empty:
                rows.append(df_iv)
        # Per-depth combined outputs
        for fn in sorted(os.listdir(base_dir)):
            if fn.startswith('final_subgroups_depth_') and fn.endswith('.csv'):
                df = pd.read_csv(os.path.join(base_dir, fn))
                if 'analysis_family' in df.columns:
                    df_iv = df[df['analysis_family'] == 'H2b_differential']
                else:
                    df_iv = df[df['analysis_key'].isin(['SM_miss','SM_false_alarm','NM_miss','NM_false_alarm'])]
                if not df_iv.empty:
                    rows.append(df_iv)
        # Legacy fallback
        if not rows:
            phase_dir = os.path.join(base_dir, 'phase_iv')
            if not os.path.isdir(phase_dir):
                return pd.DataFrame()
            for depth_folder in sorted(os.listdir(phase_dir)):
                d_path = os.path.join(phase_dir, depth_folder)
                if not os.path.isdir(d_path):
                    continue
                fsp = os.path.join(d_path, 'final_subgroups.csv')
                if os.path.exists(fsp):
                    df = pd.read_csv(fsp)
                    rows.append(df)
                else:
                    for fn in os.listdir(d_path):
                        if fn.startswith('h2b_patterns_') and fn.endswith('.csv'):
                            k = fn.replace('h2b_patterns_', '').replace('.csv', '')
                            df = pd.read_csv(os.path.join(d_path, fn))
                            df['analysis_key'] = k
                            df['source_depth'] = depth_folder
                            rows.append(df)
    elif phase.upper() == 'IVB':
        # New combined outputs first
        combined_path = os.path.join(base_dir, 'final_subgroups.csv')
        if os.path.exists(combined_path):
            df = pd.read_csv(combined_path)
            if 'analysis_family' in df.columns:
                df_ivb = df[df['analysis_family'] == 'IVB_discordance']
            else:
                df_ivb = df[df['analysis_key'].astype(str).str.startswith('IVB_')]
            if not df_ivb.empty:
                rows.append(df_ivb)
        # Per-depth combined outputs
        for fn in sorted(os.listdir(base_dir)):
            if fn.startswith('final_subgroups_depth_') and fn.endswith('.csv'):
                df = pd.read_csv(os.path.join(base_dir, fn))
                if 'analysis_family' in df.columns:
                    df_ivb = df[df['analysis_family'] == 'IVB_discordance']
                else:
                    df_ivb = df[df['analysis_key'].astype(str).str.startswith('IVB_')]
                if not df_ivb.empty:
                    rows.append(df_ivb)
        # Legacy fallback
        if not rows:
            phase_dir = os.path.join(base_dir, 'phase_ivb')
            if not os.path.isdir(phase_dir):
                return pd.DataFrame()
            for depth_folder in sorted(os.listdir(phase_dir)):
                d_path = os.path.join(phase_dir, depth_folder)
                if not os.path.isdir(d_path):
                    continue
                fsp = os.path.join(d_path, 'final_subgroups_ivb.csv')
                if os.path.exists(fsp):
                    df = pd.read_csv(fsp)
                    rows.append(df)
                else:
                    for fn in os.listdir(d_path):
                        if fn.startswith('ivb_patterns_') and fn.endswith('.csv'):
                            m = re.match(r'^ivb_patterns_(deaths|survivors)_(SM|NM)_depth_\d+\.csv$', fn)
                            if m:
                                domain, side = m.group(1), m.group(2)
                                analysis_key = f"IVB_{domain}_{side}"
                            else:
                                analysis_key = 'IVB_unknown'
                            df = pd.read_csv(os.path.join(d_path, fn))
                            df['analysis_key'] = analysis_key
                            df['source_depth'] = depth_folder
                            rows.append(df)
    else:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Ensure unified 'rule' column exists (final_subgroups use 'rule_str')
    if 'rule' not in out.columns:
        if 'rule_str' in out.columns:
            out['rule'] = out['rule_str']
        else:
            out['rule'] = ''
    out['rule'] = out['rule'].astype(str).apply(_normalize_rule_case)
    return out

def main():
    ap = argparse.ArgumentParser(description='Cross-depth archetype selection for Phase IV and IVB')
    ap.add_argument('--config_file', type=str, default=None)
    ap.add_argument('--phase', type=str, default='IVB', choices=['IV','IVB'])
    ap.add_argument('--min_coverage', type=int, default=25)
    ap.add_argument('--min_lift', type=float, default=1.25)
    ap.add_argument('--jaccard_thresh', type=float, default=0.80)
    ap.add_argument('--max_archetypes', type=int, default=3)
    args = ap.parse_args()

    cfg = _load_config(args.config_file)
    base_dir = cfg.OUTPUT_DIR
    os.makedirs(base_dir, exist_ok=True)

    def _write_empty(base_dir: str, phase: str):
        cols = ['analysis_key','rule','coverage','coverage_pct','lift','quality_WRAcc','target_share','baseline_rate','source_depth','advantaged_model','battleground','section']
        empty = pd.DataFrame(columns=cols)
        out_name = 'final_archetypes.csv' if phase.upper() == 'IV' else 'final_archetypes_ivb.csv'
        empty.to_csv(os.path.join(base_dir, out_name), index=False)
        print(f'No archetypes; wrote empty file at {os.path.join(base_dir, out_name)}')

    all_candidates = _load_all_candidates(base_dir, args.phase)
    if all_candidates.empty:
        print('No candidate patterns found to select archetypes from.')
        _write_empty(base_dir, args.phase)
        return

    # Phase 1: quantitative pruning before any clustering
    if not {'coverage', 'lift'}.issubset(set(all_candidates.columns)):
        missing = sorted(list({'coverage', 'lift'} - set(all_candidates.columns)))
        print(f"Error: Missing required columns for pruning: {', '.join(missing)}")
        return
    pruned = all_candidates[(all_candidates['coverage'] >= args.min_coverage) & (all_candidates['lift'] >= args.min_lift)].copy()
    if pruned.empty:
        print('No candidates remain after quantitative pruning. Adjust thresholds or review inputs.')
        _write_empty(base_dir, args.phase)
        return

    # Select per analysis_key independently
    winners_rows = []
    for ak, sub in pruned.groupby('analysis_key'):
        # Build member sets for Jaccard de-dup, then select optimal from each cluster
        if 'members' in sub.columns:
            def to_set(m):
                if isinstance(m, str):
                    return set(int(x) for x in m.split('|') if x)
                return set()
            sub = sub.copy()
            sub['member_set'] = sub['members'].apply(to_set)
        else:
            sub = sub.copy()
            sub['member_set'] = [set()]*len(sub)
        # Seed clusters prioritizing strongest signal (WRAcc/Lift) → coverage → parsimony
        sub = sub.copy()
        sub['n_conditions'] = sub['rule'].astype(str).apply(_count_conditions)
        if 'quality_WRAcc' in sub.columns:
            order = list(sub.sort_values(['quality_WRAcc','lift','coverage','n_conditions'], ascending=[False, False, False, True]).index)
        else:
            order = list(sub.sort_values(['lift','coverage','n_conditions'], ascending=[False, False, True]).index)
        used = set()
        clusters = []
        # If no member IDs are available, warn and skip clustering to avoid silent failure
        has_any_members = sub['member_set'].apply(lambda s: isinstance(s, set) and len(s) > 0).any()
        if not has_any_members:
            print(f"Warning: No 'members' available for analysis_key={ak}; skipping Jaccard clustering (each rule forms its own cluster).")
            clusters = [[i] for i in order]
        else:
            for i in order:
                if i in used:
                    continue
                seed_set = sub.loc[i,'member_set']
                group = [i]
                used.add(i)
                for j in order:
                    if j in used:
                        continue
                    sj = sub.loc[j,'member_set']
                    jac = _jaccard(seed_set, sj)
                    if jac >= args.jaccard_thresh:
                        group.append(j)
                        used.add(j)
                clusters.append(group)
        # From each cluster, rank by parsimony → lift → coverage → shallower depth
        def n_cond(rule: str) -> int:
            s = str(rule).lower()
            return s.count(' and ') + s.count(' && ') + s.count(' & ') + 1
        winners = []
        for grp in clusters:
            cand = sub.loc[grp].copy()
            cand['n_conditions'] = cand['rule_str' if 'rule_str' in cand.columns else 'rule'].apply(n_cond)
            # Prefer shallower depth: extract numeric
            def depth_num(x):
                try:
                    return int(str(x).split('_')[-1])
                except Exception:
                    return 999
            cand['_depth_num'] = cand['source_depth'].apply(depth_num) if 'source_depth' in cand.columns else 999
            cand = cand.sort_values(['n_conditions','lift','coverage','_depth_num'], ascending=[True, False, False, True])
            winners.append(cand.iloc[0])
        if winners:
            winners_rows.append(pd.DataFrame(winners))

    if not winners_rows:
        print('No archetypes selected under thresholds.')
        _write_empty(base_dir, args.phase)
        return

    final = pd.concat(winners_rows, ignore_index=True)
    # Add explicit advantage/battleground annotations for clarity
    if 'analysis_key' in final.columns:
        if args.phase.upper() == 'IVB':
            def _anno(ak: str):
                m = re.match(r'^IVB_(deaths|survivors)_(SM|NM)$', str(ak))
                if not m:
                    return pd.Series({'battleground': '', 'advantaged_model': ''})
                bg, side = m.group(1), m.group(2)
                return pd.Series({'battleground': bg, 'advantaged_model': 'Semantic Model' if side=='SM' else 'Numerical Model'})
            ann = final['analysis_key'].apply(_anno)
            final = pd.concat([final, ann], axis=1)
        else:
            # IV: carry human-readable section label; advantage concept not applicable here
            title_map = {
                'SM_miss': 'SM False Negatives vs Concordant True Positives',
                'SM_false_alarm': 'SM False Positives vs Concordant True Negatives',
                'NM_miss': 'NM False Negatives vs Concordant True Positives',
                'NM_false_alarm': 'NM False Positives vs Concordant True Negatives',
            }
            final['section'] = final['analysis_key'].map(title_map).fillna(final['analysis_key'])
            final['advantaged_model'] = ''
            final['battleground'] = ''

    # Enforce per-group cap (IV: per analysis_key; IVB: per battleground+advantaged_model)
    if isinstance(args.max_archetypes, int) and args.max_archetypes > 0:
        tmp = final.copy()
        rule_col = 'rule_str' if 'rule_str' in tmp.columns else 'rule'
        tmp['n_conditions'] = tmp[rule_col].astype(str).apply(_count_conditions)
        def _depth_num2(x):
            try:
                return int(str(x).split('_')[-1])
            except Exception:
                return 999
        tmp['_depth_num'] = tmp['source_depth'].apply(_depth_num2) if 'source_depth' in tmp.columns else 999
        if 'quality_WRAcc' in tmp.columns:
            sort_keys = ['quality_WRAcc','lift','coverage','n_conditions','_depth_num']
            ascending = [False, False, False, True, True]
        else:
            sort_keys = ['lift','coverage','n_conditions','_depth_num']
            ascending = [False, False, True, True]
        tmp = tmp.sort_values(sort_keys, ascending=ascending)
        if args.phase.upper() == 'IVB' and {'battleground','advantaged_model'}.issubset(tmp.columns):
            group_cols = ['battleground','advantaged_model']
        else:
            group_cols = ['analysis_key'] if 'analysis_key' in tmp.columns else None
        if group_cols:
            final = tmp.groupby(group_cols, group_keys=False).head(args.max_archetypes)
        else:
            final = tmp.head(args.max_archetypes)
    out_name = 'final_archetypes.csv' if args.phase.upper() == 'IV' else 'final_archetypes_ivb.csv'
    final.to_csv(os.path.join(base_dir, out_name), index=False)
    print(f'Wrote {final.shape[0]} archetypes to {os.path.join(base_dir, out_name)}')


if __name__ == '__main__':
    main()