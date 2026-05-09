import os
import argparse
import importlib.util
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve
import textwrap

# Reuse beauty functions and mapping from plot_meta_features.py
NAME_MAP = {
    "blood_pressure_state": "Blood pressure state",
    "heart_rate_state": "Heart rate state",
    "has_elevated_troponin": "Elevated troponin",
    "decreased_cardiac_output": "Decreased cardiac output",
    "shock_index": "Shock index",
    "has_creatinine_elevation": "Creatinine elevation",
    "has_severe_creatinine_elevation": "Severe creatinine elevation",
    "creatinine_elevation_stage": "Creatinine elevation stage",
    "hemoglobin_level_category": "Hemoglobin level category",
    "severe_lab_abnormality_count": "Severe lab abnormality count",
    "has_prolonged_coagulation": "Prolonged coagulation",
    "hepatic_lab_abnormality_type": "Hepatic lab abnormality state",
    "has_low_albumin": "Low albumin",
    "has_high_plateau_pressure": "High plateau pressure",
    "has_high_peep_set": "High PEEP set",
    "has_low_oxygen_saturation": "Low oxygen saturation",
    "has_elevated_co2": "Elevated CO2",
    "respiratory_rate_state": "Resp rate state",
    "pf_ratio": "P/F ratio",
    "pf_ratio_category": "P/F ratio category",
    "bun_creatinine_ratio": "BUN/Creatinine ratio",
    "creatinine_trend": "Creatinine 24h trend",
    "gcs_level": "GCS level",
    "severe_gcs_impairment_unconfounded_by_ventilation": "Severe GCS impairment (unconfounded)",
    "has_hyperlactatemia": "Hyperlactatemia",
    "lactate_and_ph_severity": "Metabolic stress severity (Lactate/pH)",
    "lactate_trend": "Lactate 24h trend",
    "acid_base_state": "Acid-base state",
    "has_low_bicarb_with_acidemia": "Low bicarb (with acidemia)",
    "anion_gap_corrected": "Anion gap",
    "glucose_state": "Glucose state",
    "sodium_state": "Sodium state",
    "potassium_state": "Potassium state",
    "has_high_neutrophil_count": "High neutrophil count",
    "has_low_neutrophil_count": "Low neutrophil count",
    "has_mechanical_ventilation_data": "Mechanical ventilation data",
    "has_cvp_or_pap_measurements": "CVP or PAP measurements",
    "has_hypocalcemia_ionized": "Ionized hypocalcemia",
    "has_hypomagnesemia": "Hypomagnesemia",
    "has_hypophosphatemia": "Hypophosphatemia",
    "platelet_count_category": "Platelet count category",
    "wbc_count_state": "WBC count state",
    "hr_volatility": "HR volatility",
    "sbp_volatility": "SBP volatility",
    "map_volatility": "MAP volatility",
    "rr_volatility": "RR volatility",
    "data_density_score": "Data density score",
    "sirs_rr_criterion": "SIRS RR criterion",
    "sirs_wbc_criterion": "SIRS WBC criterion",
    "sirs_temp_criterion": "SIRS temp criterion",
    "sirs_hr_criterion": "SIRS HR criterion",
    "meets_sirs_criteria": "Meets SIRS criteria",
    "has_effusion_fluid_labs": "Effusion fluid labs",
}

def beautify_var_name(name: str) -> str:
    if name in NAME_MAP:
        return NAME_MAP[name]
    words = name.replace("has_", "").replace("is_", "").replace("_", " ").strip()
    words = re.sub(r"\bcreatinine\b", "creatinine", words, flags=re.IGNORECASE)
    words = re.sub(r"\bwbc\b", "WBC", words, flags=re.IGNORECASE)
    words = re.sub(r"\brr\b", "respiratory rate", words, flags=re.IGNORECASE)
    words = re.sub(r"\baki\b", "creatinine elevation", words, flags=re.IGNORECASE)
    return words[:1].upper() + words[1:]

def pretty_condition(expr: str) -> str:
    expr = str(expr or "").strip()
    m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*([\[\(])\s*([^:]+)\s*:\s*([^\]\)]+)\s*([\]\)])$", expr)
    if m:
        var, lbr, a, b, rbr = m.groups()
        name = beautify_var_name(var)
        # Use a more compact scientific interval notation [a - b]
        return f"{name} [{a} - {b}]"

    m = re.match(r"^([A-Za-z0-9_]+)\s*==\s*'([^']+)'\s*$", expr)
    if m:
        var, val = m.groups()
        name = beautify_var_name(var)
        return f"{name} = {val.replace('_', ' ')}"

    m = re.match(r"^([A-Za-z0-9_]+)\s*==\s*(True|False)\s*$", expr)
    if m:
        var, b = m.groups()
        name = beautify_var_name(var)
        if b == "True":
            if name.lower().startswith("on ") or name.lower().startswith("invasive"):
                return name
            return f"{name} present"
        else:
            if name.lower().startswith("on "):
                return f"Not {name.lower()}"
            return f"No {name.lower()}"

    m = re.match(r"^([A-Za-z0-9_]+)\s*([<>]=?)\s*([0-9.]+)\s*$", expr)
    if m:
        var, op, val = m.groups()
        name = beautify_var_name(var)
        return f"{name} {op} {val}"

    expr = expr.replace("==", "=").replace("==True", "").strip()
    return beautify_var_name(expr)

def short_rule_tag(rule: str, max_words: int = 4) -> str:
    """Readable abbreviated archetype label for inset legends (no truncation)."""
    s = pretty_rule(rule, width=200).replace('\n', ' ').strip()
    repl = {
        'Blood pressure': 'BP',
        'Heart rate': 'HR',
        'respiratory rate': 'RR',
        'Creatinine elevation': 'Cr↑',
        'Severe creatinine elevation': 'Cr↑↑',
        'Prolonged coagulation': 'Coag↑',
        'Hepatic lab abnormality state': 'Liver',
        'Synthetic Dysfunction': 'SynDys',
        'Cholestatic': 'Chol',
        'Hepatocellular': 'HepCell',
        'Low albumin': 'Alb↓',
        'Hemoglobin': 'Hgb',
        'hemoglobin': 'Hgb',
        'level category': 'cat',
        'Hyperlactatemia': 'Lac↑',
        'Mechanical ventilation data': 'VentData',
        'P/F ratio': 'PF',
        'Platelet count category': 'PltCat',
        'WBC count state': 'WBCState',
        'Metabolic stress severity (Lactate/pH)': 'MetStress',
        'BUN/Creatinine ratio': 'BUN:Cr',
        'High neutrophil count': 'Neut↑',
        'Low neutrophil count': 'Neut↓',
        ' present': '',
        'No ': 'No ',
        'Not ': 'Not ',
        ' and ': ' AND ',
        ' = ': '=',
        ' [': '[',
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    # Keep all conditions; normalize separators to explicit logical conjunction.
    parts = [p.strip() for p in s.split(';') if p.strip()]
    s = ' AND '.join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    return s or str(rule)[:24]


def pretty_rule(rule: str, width: int = 35) -> str:
    if not isinstance(rule, str) or not rule:
        return ""
    parts = re.split(r"\s+AND\s+|\s+&\s+", rule)
    rule_txt = "; ".join([pretty_condition(p) for p in parts if str(p).strip()])
    # Apply multi-line wrapping for compact layout
    return textwrap.fill(rule_txt, width=width)

def _load_config(config_file: str):
    if not os.path.isabs(config_file):
        config_file = os.path.abspath(config_file)
    spec = importlib.util.spec_from_file_location('dynamic_config_h2', config_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConfigH2()

def evaluate_rule_mask(rule_str, df_full):
    """
    Evaluates a pysubgroup rule string on a DataFrame.
    """
    parts = [p.strip() for p in str(rule_str).split('AND')]
    mask = np.ones(len(df_full), dtype=bool)
    
    for p in parts:
        if not p: continue
        
        # 1. Equality: col == val (handles strings, booleans)
        m_eq = re.match(r"^([A-Za-z0-9_]+)\s*==\s*(.+)$", p)
        if m_eq:
            col, val = m_eq.groups()
            val = val.strip("'")
            if col in df_full.columns:
                s = df_full[col]
                if val in {'True', 'False'}:
                    target_bool = (val == 'True')
                    if pd.api.types.is_bool_dtype(s):
                        mask &= (s.fillna(False) == target_bool)
                    elif pd.api.types.is_numeric_dtype(s):
                        mask &= (pd.to_numeric(s, errors='coerce').fillna(-999) == (1 if target_bool else 0))
                    else:
                        norm = s.astype(str).str.strip().str.lower()
                        truthy = {'true', '1', 'yes', 'y', 't'}
                        falsy = {'false', '0', 'no', 'n', 'f'}
                        mask &= norm.isin(truthy if target_bool else falsy)
                else:
                    mask &= (s.astype(str).str.lower() == str(val).lower())
            continue
            
        # 2. Range: col:[low:high] or col:(low:high)
        m_rg = re.match(r"^([A-Za-z0-9_]+)\s*:\s*([\[\(])\s*([^:]+)\s*:\s*([^\]\)]+)\s*([\]\)])$", p)
        if m_rg:
            col, lbr, a, b, rbr = m_rg.groups()
            if col in df_full.columns:
                vals = pd.to_numeric(df_full[col], errors='coerce')
                try:
                    low = float(a)
                    high = float(b)
                    if lbr == '[': mask &= (vals >= low)
                    else: mask &= (vals > low)
                    
                    if rbr == ']': mask &= (vals <= high)
                    else: mask &= (vals < high)
                except ValueError:
                    pass
            continue

        m_ge = re.match(r"^([A-Za-z0-9_]+)\s*>=\s*([0-9.eE+-]+)\s*$", p)
        if m_ge:
            col, val = m_ge.groups()
            if col in df_full.columns:
                vals = pd.to_numeric(df_full[col], errors='coerce')
                mask &= (vals >= float(val))
            continue

        m_le = re.match(r"^([A-Za-z0-9_]+)\s*<=\s*([0-9.eE+-]+)\s*$", p)
        if m_le:
            col, val = m_le.groups()
            if col in df_full.columns:
                vals = pd.to_numeric(df_full[col], errors='coerce')
                mask &= (vals <= float(val))
            continue
            
    return mask

def format_pvalue(p):
    if p < 0.0001: return "< 0.0001"
    if p < 0.001: return "< 0.001"
    if p < 0.01: return "< 0.01"
    return f"= {p:.3f}"

def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty_like(q_ranked)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q

def clopper_pearson_ci(k, n, alpha=0.05):
    from scipy.stats import binomtest
    if n <= 0:
        return 0.0, 0.0
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=1 - alpha, method='exact')
    return float(ci.low), float(ci.high)


def youden_threshold(y_true, proba):
    """Optimal threshold maximizing Youden's J on the test set (same convention as H2a reports)."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    m = np.isfinite(p) & (y >= 0)
    if m.sum() < 5 or len(np.unique(y[m])) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y[m], p[m])
    j = tpr - fpr
    i = int(np.nanargmax(j))
    return float(thr[i])


def archetype_fpr_enrichment_curve(y_true, proba, arche_mask, thresholds, min_fp_count=20):
    """
    Enrichment = FPR(rule-matched negatives) / FPR(not-matched negatives), per threshold.
    FPR = P(pred >= t | y=0) in each subgroup.
    If min_fp_count > 0, returns nan where either subgroup has fewer than that many false positives
    (y=0 and pred >= t), to avoid ratio instability.
    Returns (enrichment array, valid_mask bool array).
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    m = np.asarray(arche_mask, dtype=bool).ravel()
    out, valid = [], []
    for t in thresholds:
        pred = p >= t
        neg = (y == 0)
        in_r = m & neg
        out_r = (~m) & neg
        n_fp_in = int(np.sum(in_r & pred))
        n_fp_out = int(np.sum(out_r & pred))
        ok_counts = (n_fp_in >= min_fp_count) and (n_fp_out >= min_fp_count)
        d_in = int(np.sum(in_r))
        d_out = int(np.sum(out_r))
        if not ok_counts or d_in == 0 or d_out == 0:
            out.append(np.nan)
            valid.append(False)
            continue
        f_in = n_fp_in / d_in
        f_out = n_fp_out / d_out
        if f_out == 0.0:
            val = np.nan if (f_in > 0) else 1.0
            out.append(val)
            valid.append(np.isfinite(val))
        else:
            out.append(f_in / f_out)
            valid.append(True)
    return np.asarray(out, dtype=float), np.asarray(valid, dtype=bool)


def calculate_net_benefit(y_true, proba, thresholds):
    """
    Decision Curve Analysis (DCA): Compute Net Benefit (NB).
    NB = (TP/N) - (FP/N) * (T / (1-T))
    """
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba).ravel()
    n = len(y)
    p_pos = (y == 1)
    p_neg = (y == 0)
    
    nb = []
    for t in thresholds:
        if t <= 0:
            # Treat all at t=0
            tp = np.sum(p_pos)
            fp = np.sum(p_neg)
        elif t >= 1:
            # Treat none at t=1
            tp = 0
            fp = 0
        else:
            pred = (p >= t).astype(int)
            tp = np.sum((pred == 1) & p_pos)
            fp = np.sum((pred == 1) & p_neg)
        
        weight = t / (1 - t) if t < 1 else 1e9
        val = (tp / n) - (fp / n) * weight
        nb.append(val)
    return np.array(nb)

def get_sweep_proportions(y_true, nm_proba, sm_proba, target_sens_range=None):
    if target_sens_range is None:
        target_sens_range = np.arange(0.05, 0.96, 0.01)
    
    y = np.asarray(y_true).astype(int).ravel()
    nm_p = np.asarray(nm_proba).ravel()
    sm_p = np.asarray(sm_proba).ravel()
    n_pos = int(y.sum())
    
    def _thresh_for_sens(proba, y_bin, target_sens):
        order = np.argsort(-proba)
        cum_pos = np.cumsum(y_bin[order])
        target_count = max(1, int(np.ceil(target_sens * n_pos)))
        idx = np.searchsorted(cum_pos, target_count, side='left')
        idx = min(idx, len(proba) - 1)
        return float(proba[order[idx]])

    shared = []
    nm_only = []
    sm_only = []
    
    for ts in target_sens_range:
        t_nm = _thresh_for_sens(nm_p, y, ts)
        t_sm = _thresh_for_sens(sm_p, y, ts)
        
        nm_err = (nm_p >= t_nm).astype(int) != y
        sm_err = (sm_p >= t_sm).astype(int) != y
        
        both_w = int((nm_err & sm_err).sum())
        n_w = int((nm_err & ~sm_err).sum())
        s_w = int((~nm_err & sm_err).sum())
        
        total = max(1, both_w + n_w + s_w)
        shared.append(100 * both_w / total)
        nm_only.append(100 * n_w / total)
        sm_only.append(100 * s_w / total)
        
    return target_sens_range * 100, np.array(shared), np.array(nm_only), np.array(sm_only)

_OKABE_ITO = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
    '#999999',  # gray
]


def _rule_color_map(rule_strings):
    uniq = sorted(set(rule_strings))
    return {r: _OKABE_ITO[i % len(_OKABE_ITO)] for i, r in enumerate(uniq)}


def process_config(config_path, title_prefix):
    """Load test data, archetype rules, and thresholds for threshold-sweep enrichment figures."""
    cfg = _load_config(config_path)
    art_path = os.path.join(cfg.H2A_OUTPUT_DIR, 'h2a_to_h2b_artifact.pkl')
    try:
        with open(art_path, 'rb') as f:
            art = pickle.load(f)
    except Exception:
        art = pd.read_pickle(art_path)

    y_true = np.array(art['y_true'])
    pop_prevalence = float(np.mean(y_true))

    try:
        X_num = pd.read_pickle(cfg.X_TEST_NUM_PATH)
        X_pheno = pd.read_pickle(cfg.X_TEST_PHENOS_PATH)
        df_full = pd.concat([X_num.reset_index(drop=True), X_pheno.reset_index(drop=True)], axis=1)
        df_full['TARGET_Y'] = y_true
    except Exception as e:
        print(f"Warning: Could not load features for rule evaluation: {e}")
        df_full = None

    # Phase IV only: per-model false positives (NM_false_alarm / SM_false_alarm).
    # Do not load IVB discordance / comparative archetypes (final_archetypes_ivb.csv).
    path_final = os.path.join(cfg.OUTPUT_DIR, 'final_archetypes.csv')
    if not os.path.exists(path_final):
        print(f"Warning: Missing {path_final}; skipping archetype enrichment for this task.")
        return None
    try:
        df = pd.read_csv(path_final)
    except Exception:
        return None
    if df_full is None:
        print("Warning: Full test feature matrix unavailable; skipping archetype figure generation.")
        return None

    thr = art.get('thresholds') or {}
    nm_t = float(thr.get('nm', youden_threshold(y_true, art.get('nm_proba'))))
    sm_t = float(thr.get('sm', youden_threshold(y_true, art.get('sm_proba'))))

    panel_keys = {'NM': 'NM_false_alarm', 'SM': 'SM_false_alarm'}

    archetypes_nm, archetypes_sm = [], []
    for mdl_key, analysis_key in panel_keys.items():
        sub_df = df[df['analysis_key'] == analysis_key].copy()
        for _, row in sub_df.iterrows():
            rule = row.get('rule_str', '')
            if not isinstance(rule, str) or not rule.strip():
                continue
            in_mask = evaluate_rule_mask(rule, df_full)
            if int(np.sum(in_mask)) < 5:
                continue
            neg_n = int(np.sum(in_mask & (y_true == 0)))
            if neg_n < 3:
                continue
            entry = {'rule_str': rule, 'label': pretty_rule(rule)}
            if mdl_key == 'NM':
                archetypes_nm.append(entry)
            else:
                archetypes_sm.append(entry)

    return {
        'title': title_prefix,
        'y_true': y_true,
        'nm_proba': np.asarray(art.get('nm_proba'), dtype=float),
        'sm_proba': np.asarray(art.get('sm_proba'), dtype=float),
        'pop_prev': pop_prevalence,
        'thresholds': {'nm': nm_t, 'sm': sm_t},
        'archetypes_nm': archetypes_nm,
        'archetypes_sm': archetypes_sm,
        'df_full': df_full,
    }


def _shade_x_suppression(ax, thresholds, panel_all_suppressed, zorder=0.5):
    """Light gray vertical bands where every curve is gated out (FP count < min)."""
    t = np.asarray(thresholds, dtype=float)
    m = np.asarray(panel_all_suppressed, dtype=bool)
    if not m.any():
        return
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        lo = max(0.05, t[i] - 0.005)
        hi = min(0.95, t[j - 1] + 0.005)
        ax.axvspan(lo, hi, facecolor='#c4c4c4', alpha=0.55, zorder=zorder)
        i = j


def generate_figure_sx_archetype_fpr_enrichment(task_bundles, output_path):
    """
    Figure S4: FPR enrichment vs classification threshold.
    Each panel shows archetype lines (rule mined from model-specific false positives) on the full test set,
    gated to thresholds where >= 20 false positives remain in both (rule & non-rule) groups.
    """
    if not task_bundles or len(task_bundles) < 2:
        return
    mort, readm = task_bundles[0], task_bundles[1]
    if mort is None or readm is None:
        return

    min_fp = 20
    y_cap_max = 15.0
    thresh_sweep = np.arange(0.05, 0.96, 0.01)

    all_rules = []
    for b in (mort, readm):
        for a in b['archetypes_nm'] + b['archetypes_sm']:
            all_rules.append(a['rule_str'])
    cmap = _rule_color_map(all_rules)

    panel_cfg = [
        ('A', mort, 'NM', 'nm_proba', 'nm'),
        ('B', mort, 'SM', 'sm_proba', 'sm'),
        ('C', readm, 'NM', 'nm_proba', 'nm'),
        ('D', readm, 'SM', 'sm_proba', 'sm'),
    ]
    pos = {'A': (0, 0), 'B': (0, 1), 'C': (1, 0), 'D': (1, 1)}
    task_lbl = {'Mortality': 'In-hospital mortality', 'Readmission': '30-day readmission'}

    def _task_row_cap(bundle, mdl_name):
        y_max = 1.0
        arch_list = bundle['archetypes_nm'] if mdl_name == 'NM' else bundle['archetypes_sm']
        proba = bundle['nm_proba'] if mdl_name == 'NM' else bundle['sm_proba']
        y_true = bundle['y_true']
        df_full = bundle['df_full']
        for ar in arch_list:
            m = evaluate_rule_mask(ar['rule_str'], df_full)
            curve, vm = archetype_fpr_enrichment_curve(
                y_true, proba, m, thresh_sweep, min_fp_count=min_fp
            )
            finite = curve[np.isfinite(curve)]
            if finite.size:
                y_max = max(y_max, float(np.nanmax(finite)))
        return min(y_cap_max, max(1.0, y_max * 1.08))

    y_cap_mort = max(_task_row_cap(mort, 'NM'), _task_row_cap(mort, 'SM'))
    y_cap_read = max(_task_row_cap(readm, 'NM'), _task_row_cap(readm, 'SM'))

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    axes_grid = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ]

    # Collect legend rules by model (Panels A/C = NM, Panels B/D = SM)
    # Inset legends will be placed per panel (NM in A/C, SM in B/D).

    for label, bundle, mdl_name, proba_key, thr_key in panel_cfg:
        r, c = pos[label]
        ax = axes_grid[r][c]
        y_true = bundle['y_true']
        proba = bundle[proba_key]
        df_full = bundle['df_full']
        arch_list = bundle['archetypes_nm'] if mdl_name == 'NM' else bundle['archetypes_sm']
        t_youden = float(bundle['thresholds'][thr_key])
        y_cap = y_cap_mort if r == 0 else y_cap_read

        # Null boundary: lightly shade below enrichment = 1.
        ax.axhspan(0.0, 1.0, facecolor='#d0d0d0', alpha=0.35, zorder=0)

        # Primary analysis threshold.
        ax.axvline(t_youden, color='#636363', linestyle='--', linewidth=1.7, zorder=3)

        if not arch_list:
            ax.text(0.5, 0.5, 'No qualifying archetypes', transform=ax.transAxes, ha='center', va='center', fontsize=11)
            ax.set_ylim(0.0, y_cap)
            ax.set_xlim(0.05, 0.95)
            continue

        # Panel D declutter: keep only top 4 archetypes by enrichment at Youden's J.
        if label == 'D':
            ranked = []
            for ar in arch_list:
                m = evaluate_rule_mask(ar['rule_str'], df_full)
                val, vm = archetype_fpr_enrichment_curve(y_true, proba, m, np.array([t_youden]), min_fp_count=min_fp)
                v = float(val[0]) if (vm.size and vm[0] and np.isfinite(val[0])) else -np.inf
                ranked.append((v, ar))
            ranked.sort(key=lambda x: x[0], reverse=True)
            ge15 = [ar for v, ar in ranked if np.isfinite(v) and v >= 1.5][:4]
            arch_list = ge15 if len(ge15) >= 3 else [ar for _, ar in ranked[:4]]

        valid_any = np.zeros(len(thresh_sweep), dtype=bool)
        curves_cache = []
        for ar in arch_list:
            rule = ar['rule_str']
            m = evaluate_rule_mask(rule, df_full)
            curve, vm = archetype_fpr_enrichment_curve(
                y_true, proba, m, thresh_sweep, min_fp_count=min_fp
            )
            curves_cache.append((rule, curve))
            valid_any |= vm

        if valid_any.any():
            last_idx = int(np.where(valid_any)[0].max())
        else:
            last_idx = 0
        x_end = float(thresh_sweep[last_idx])
        # Guard against degenerate ranges (can happen in tiny synthetic test fixtures).
        if x_end <= 0.05 + 1e-9:
            x_end = 0.051
        # Keep Youden's J visible; pad left when t ≈ 0.05 (stored NM threshold) or else line sits on the spine.
        t_y = float(t_youden)
        tmax = max(x_end, t_y)
        span = max(tmax - 0.05, 1e-6)
        x_pad = min(0.05, max(0.012, 0.07 * span))
        x_right = min(0.95, tmax + x_pad)
        x_left = 0.05
        if t_y < x_left + 0.01:
            x_left = max(0.01, t_y - 0.02)
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(0.0, y_cap)
        if label == 'D':
            ax.set_ylim(0.5, 3.0)

        # Adaptive ticks: fixed 0.1 steps yield zero ticks when x_right < 0.1 (common in Panel D).
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4, prune=None))
        ax.tick_params(axis='x', labelsize=10, labelbottom=True)
        ax.grid(True, alpha=0.25, linestyle=':')

        # Plot lines and right-endpoint markers only (no inline labels).
        for rule, curve in curves_cache:
            curve_disp = curve.copy()
            curve_disp = np.where(np.isfinite(curve_disp), np.clip(curve_disp, 0.0, y_cap), np.nan)

            # Restrict to the panel-visible range.
            curve_disp = curve_disp[: last_idx + 1]
            x_vals = thresh_sweep[: last_idx + 1]
            ax.plot(x_vals, curve_disp, color=cmap[rule], linewidth=2.0, alpha=0.95, zorder=2)

            idxs = np.where(np.isfinite(curve_disp))[0]
            if idxs.size == 0:
                continue
            k = int(idxs[-1])
            ax.scatter(
                float(x_vals[k]), float(curve_disp[k]), s=24, color=cmap[rule],
                edgecolors='white', linewidths=0.7, zorder=4
            )

        ax.set_title(
            f"{label}. {task_lbl.get(bundle['title'], bundle['title'])} — {mdl_name}",
            loc='left', fontsize=12, fontweight='bold'
        )
        ax.set_ylabel('FPR enrichment (rule / non-rule)', fontsize=11)
        if label == 'D':
            ax.text(
                0.99, 0.99, "y-axis: 0.5-3.0 (zoom)", transform=ax.transAxes,
                ha='right', va='top', fontsize=7.8, color='#4a4a4a'
            )

        # Panel-specific Youden label placement
        y_top = ax.get_ylim()[1]
        if label == 'C':
            ax.text(
                min(ax.get_xlim()[1], t_youden + 0.012), y_top * 0.92, f"t={t_youden:.2f}",
                fontsize=8.2, color='#3b3b3b', rotation=90, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75)
            )
        else:
            ax.text(
                t_youden, y_top * 0.96, f"t={t_youden:.2f}",
                fontsize=8.2, color='#3b3b3b', rotation=90, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.75)
            )

        # Inset legend per panel (only rules shown in this panel).
        panel_rules = sorted(set([r for r, _ in curves_cache]))
        panel_handles = [Line2D([0], [0], color=cmap[r], lw=2.0, label=short_rule_tag(r)) for r in panel_rules]
        if panel_handles:
            leg = ax.legend(
                handles=panel_handles, loc='upper left', ncol=1, frameon=True,
                fontsize=7.9, title=f'{mdl_name} archetypes', title_fontsize=8.6,
                handlelength=2.1, columnspacing=0.8, borderpad=0.3, labelspacing=0.28
            )
        else:
            leg = None
        if leg is not None:
            leg.get_frame().set_alpha(0.88)
            leg.get_frame().set_edgecolor('#d0d0d0')

    for bottom_ax in (axes_grid[1][0], axes_grid[1][1]):
        bottom_ax.set_xlabel('Predicted probability threshold', fontsize=12)
        bottom_ax.tick_params(axis='x', labelsize=10, labelbottom=True, which='major')

    fig.text(
        0.5, 0.035,
        f"Enrichment = (FP rate among y=0 patients matching rule) / (FP rate among y=0 patients not matching rule). "
        f"Curves terminate when either rule or non-rule group has < {min_fp} false positives at a threshold. "
        f"Panels A/B share one y-scale and Panels C/D share another; Panel D is zoomed to 0.5-3.0. "
        f"In Panel D, only top SM-readmission archetypes by enrichment at Youden's J are shown; omitted archetypes are <1.5.",
        ha='center', fontsize=8.6, color='#424242',
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.14)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.35, dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.35, dpi=300)
    plt.close()

def generate_figure_sx_plus_1(all_results, output_path):
    """
    Supplemental Figure S[X+1]: Error Decomposition Robustness.
    1x2 Horizontal Grid (NM/SM Shared/Unique Errors).
    """
    tasks = [r for r in all_results if r is not None]
    if not tasks: return

    fig, axes = plt.subplots(1, len(tasks), figsize=(16, 6))
    if len(tasks) == 1: axes = [axes]
    
    colors = {'NM': '#3498db', 'SM': '#e74c3c', 'Shared': '#7f8c8d'}
    
    for i, res in enumerate(tasks):
        title = res['title']
        ax = axes[i]
        
        # --- Error Decomposition Sweep ---
        sens, shared, nm_u, sm_u = get_sweep_proportions(res['y_true'], res['nm_proba'], res['sm_proba'])
        
        stack0 = np.zeros_like(shared)
        stack1 = shared
        stack2 = shared + nm_u
        stack3 = shared + nm_u + sm_u
        
        ax.fill_between(sens, stack0, stack1, color=colors['Shared'], alpha=0.4, label='Shared Errors')
        ax.fill_between(sens, stack1, stack2, color=colors['NM'], alpha=0.5, label='Unique NM Errors')
        ax.fill_between(sens, stack2, stack3, color=colors['SM'], alpha=0.5, label='Unique SM Errors')
        
        ax.set_title(f"{title}: Error Decomposition Robustness", fontsize=15, fontweight='bold')
        ax.set_ylabel("Proportion of Total Errors (%)", fontweight='bold')
        ax.set_xlabel("Recall / Sensitivity Range", fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2, linestyle='--')
        if i == 0: ax.legend(loc='lower center', ncol=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=[
        os.path.join(curr_dir, 'config_h2_morthosp.py'),
        os.path.join(curr_dir, 'config_h2_readmin30.py')
    ])
    args = parser.parse_args()
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_theme(style="white")
    
    results = []
    for cfg_path in args.configs:
        print(f"Processing data for {cfg_path}...")
        results.append(process_config(cfg_path, 'Mortality' if 'morthosp' in cfg_path else 'Readmission'))
    
    # Save manuscript-ready figures in repo-level manuscript_figures
    repo_root = os.path.abspath(os.path.join(curr_dir, '..', '..', '..'))
    out_dir = os.path.join(repo_root, 'manuscript_figures')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating Figure S[X]: Archetype FPR enrichment vs threshold...")
    generate_figure_sx_archetype_fpr_enrichment(results, os.path.join(out_dir, 'figure_sx_archetype_validation.pdf'))
    
    print("Generating Figure S[X+1]: Error Decomposition Robustness...")
    generate_figure_sx_plus_1(results, os.path.join(out_dir, 'figure_sx_plus_1_error_decomposition.pdf'))

    print(f"Workflow complete. Figures saved to {out_dir}")

if __name__ == '__main__':
    main()
