from typing import Optional

import glob
import os
import re
import textwrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.transforms as transforms

# Repo root: notebooks/Phase_6/h2b -> ../../../
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
MANUSCRIPT_FIGURES_DIR = os.path.join(_REPO_ROOT, "manuscript_outputs", "Figures")

# Reuse beauty functions
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
        lb = ">=" if lbr == "[" else ">"
        rb = "<=" if rbr == "]" else "<"
        return f"{name} in {lb} {a}, {rb} {b}"

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

    expr = expr.replace("==", "=")
    return beautify_var_name(expr)

def pretty_rule(rule: str) -> str:
    if not isinstance(rule, str) or not rule:
        return ""
    parts = re.split(r"\s+AND\s+", rule)
    return "; ".join([pretty_condition(p) for p in parts if str(p).strip()])

def pretty_analysis_label(analysis: str) -> str:
    s = str(analysis or "").strip().lower()
    if s == "nm_false_alarm":
        return "Subgroup Analysis: NM False Positives"
    if s == "sm_false_alarm":
        return "Subgroup Analysis: SM False Positives"
    if s == "nm_miss":
        return "Subgroup Analysis: NM Misses"
    if s == "sm_miss":
        return "Subgroup Analysis: SM Misses"
    return str(analysis).replace("_", " ").strip().title()

META_FEAT_MAP = {
    'total_measurement_events': 'Total events',
    'unique_feature_count': 'Measured feature\nfamilies',
    'temporal_concentration': 'Final 6h / 24h\nevent proportion',
    'aggregate_stddev': 'Within-stay\nvariability',
    'aggregate_slope': 'Absolute trend\nmagnitude',
    'imputation_proportion': 'Unmeasured family\nproportion',
}

FAMILIES = {
    'total_measurement_events': 'Density',
    'unique_feature_count': 'Density',
    'temporal_concentration': 'Temporal',
    'aggregate_stddev': 'Volatility',
    'aggregate_slope': 'Volatility',
    'imputation_proportion': 'Imputation',
}


def _find_phase_v_meta_iv_csv(base_dir: str, dataset: str) -> Optional[str]:
    """Resolve Phase V meta CSV; paths match h2b ConfigH2 OUTPUT_DIR / generate_manuscript_table."""
    if dataset == "mortality":
        dirs_try = [
            ("h2_results", "mort_hosp", "phase_v_meta"),
            ("h2_results", "phase_v_meta"),
        ]
        needle = "mort_hosp"
    else:
        dirs_try = [
            ("h2_results", "readmission_30", "phase_v_meta"),
            ("h2_results_readmission_30", "phase_v_meta"),
        ]
        needle = "readmission_30"
    for tag in ("IV", "IVB"):
        fn = f"phase_v_meta_results_{tag}.csv"
        for parts in dirs_try:
            p = os.path.join(base_dir, *parts, fn)
            if os.path.isfile(p):
                return p
    pattern = os.path.join(base_dir, "h2_results", "**", "phase_v_meta", "phase_v_meta_results_*.csv")
    matches = [
        m for m in glob.glob(pattern, recursive=True)
        if needle in m.replace("/", os.sep) and os.path.basename(m).startswith("phase_v_meta_results_")
    ]
    for tag in ("IV", "IVB"):
        fn = f"phase_v_meta_results_{tag}.csv"
        same = [m for m in matches if os.path.basename(m) == fn]
        if same:
            return same[0]
    return matches[0] if matches else None


def load_data():
    base_dir = _SCRIPT_DIR

    p_mort = _find_phase_v_meta_iv_csv(base_dir, "mortality")
    p_readm = _find_phase_v_meta_iv_csv(base_dir, "readmission")

    df_mort = pd.read_csv(p_mort) if p_mort else pd.DataFrame()
    if not df_mort.empty:
        df_mort = _align_to_manuscript_meta_selection(base_dir, df_mort, "mortality")
        df_mort["dataset"] = "Mortality"
        print(f"Loaded mortality Phase V meta: {p_mort}")

    df_readm = pd.read_csv(p_readm) if p_readm else pd.DataFrame()
    if not df_readm.empty:
        df_readm = _align_to_manuscript_meta_selection(base_dir, df_readm, "readmission")
        df_readm["dataset"] = "Readmission"
        print(f"Loaded readmission Phase V meta: {p_readm}")

    df = pd.concat([df_mort, df_readm], ignore_index=True)
    if df.empty:
        print(
            "No Phase V meta CSV loaded. Expected e.g. "
            "h2_results/mort_hosp/phase_v_meta/phase_v_meta_results_IV.csv "
            "(see config_h2_morthosp OUTPUT_DIR)."
        )
        return df
    
    df['rule_pretty'] = df['rule'].apply(pretty_rule)
    df['comparison_pretty'] = df['analysis'].apply(pretty_analysis_label)
    df['meta_pretty'] = df['meta_feature'].map(META_FEAT_MAP)
    df['family_pretty'] = df['meta_feature'].map(FAMILIES)
    
    # Extract Model
    df['Model'] = df['analysis'].apply(lambda x: 'SM' if 'SM' in x else ('NM' if 'NM' in x else 'Other'))
    
    # Rank biserial goes from -1 to 1. 
    # Let's verify: a positive rank biserial means "higher in error" according to h2v_meta_analysis.py.
    
    return df


def _align_to_manuscript_meta_selection(base_dir: str, meta_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Mirror S6/S10 selection logic so plots match manuscript tables."""
    ds_dir = "mort_hosp" if dataset == "mortality" else "readmission_30"
    arch_path = os.path.join(base_dir, "h2_results", ds_dir, "final_archetypes.csv")
    if not os.path.isfile(arch_path) or meta_df.empty:
        return meta_df

    adf = pd.read_csv(arch_path)
    if not {"analysis_key", "rule", "q_value"}.issubset(adf.columns):
        return meta_df
    if not {"analysis", "rule"}.issubset(meta_df.columns):
        return meta_df

    ordered_pairs = []
    for comp, group in adf.groupby("analysis_key"):
        g = group.sort_values("q_value").head(8)
        ordered_pairs.extend([(str(comp), str(r)) for r in g["rule"].astype(str)])
    if not ordered_pairs:
        return meta_df

    order_map = {p: i for i, p in enumerate(ordered_pairs)}
    df = meta_df.copy()
    df["_meta_input_order"] = np.arange(len(df))
    df = df[df.apply(lambda r: (str(r["analysis"]), str(r["rule"])) in order_map, axis=1)].copy()
    if df.empty:
        return df

    df["_archetype_order"] = df.apply(lambda r: order_map[(str(r["analysis"]), str(r["rule"]))], axis=1)
    report_df = df.copy()

    sort_cols = [c for c in ["_archetype_order", "analysis", "rule", "_meta_input_order"] if c in report_df.columns]
    return report_df.sort_values(sort_cols).copy() if sort_cols else report_df

def generate_plot(df):
    if df.empty:
        return

    datasets = ['Mortality', 'Readmission']
    meta_features = [
        'total_measurement_events', 'unique_feature_count', 'temporal_concentration',
        'aggregate_stddev', 'aggregate_slope',
        'imputation_proportion'
    ]

    # Filter to pre-specified meta features and display only statistically significant rows.
    df_f = df[df['meta_feature'].isin(meta_features)].copy()
    df_f['q_value_family'] = pd.to_numeric(df_f['q_value_family'], errors='coerce')
    df_f = df_f[df_f['q_value_family'] < 0.05]
    if df_f.empty:
        avail = sorted(df['meta_feature'].dropna().unique().tolist()) if 'meta_feature' in df.columns else []
        print(f"No significant rows after meta_feature filter; available meta_feature values: {avail[:20]}...")
        return
    df = df_f
    
    # Colors
    colors = {'SM': '#1f77b4', 'NM': '#ff7f0e'}  # Blue for SM, Orange for NM
    
    for ds in datasets:
        df_ds = df[df['dataset'] == ds].copy()
        if df_ds.empty:
            print(f"{ds} - No Data")
            continue
            
        ordered_pairs = df_ds[["analysis", "comparison_pretty", "rule", "rule_pretty"]].drop_duplicates()
        row_entries = []
        current_comparison = None
        for _, pair in ordered_pairs.iterrows():
            comparison = str(pair["comparison_pretty"])
            if comparison != current_comparison:
                row_entries.append({
                    "type": "header",
                    "label": comparison,
                    "key": None,
                })
                current_comparison = comparison
            row_entries.append({
                "type": "data",
                "label": "  " + textwrap.fill(str(pair["rule_pretty"]), width=62, subsequent_indent="  "),
                "key": (str(pair["analysis"]), str(pair["rule"])),
            })

        fig = plt.figure(figsize=(16, max(6.5, len(row_entries) * 0.48)))
        fig.suptitle(f"Cohort: {ds}", fontsize=18, fontweight='bold', y=1.03)
        
        axes = fig.subplots(1, len(meta_features), sharey=True)
        if len(meta_features) == 1: axes = [axes]
        
        y_positions = np.arange(len(row_entries))[::-1]
        row_to_y = {
            entry["key"]: y
            for entry, y in zip(row_entries, y_positions)
            if entry["type"] == "data"
        }
        labels = [entry["label"] for entry in row_entries]
        
        for ax, mf in zip(axes, meta_features):
            df_mf = df_ds[df_ds['meta_feature'] == mf]
            
            # Title for the column
            title = f"{META_FEAT_MAP.get(mf, mf)}"
            ax.set_title(title, fontsize=12)
            
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
            
            # Plot bars
            height = 0.35
            
            for _, row in df_mf.iterrows():
                key = (str(row['analysis']), str(row['rule']))
                if key not in row_to_y:
                    continue
                y = row_to_y[key]
                val = row['rank_biserial']
                mod = row['Model']
                
                c = colors.get(mod, 'gray')
                alpha = 1.0
                
                ax.barh(y, val, height=height, color=c, alpha=alpha, edgecolor='none')
                
            ax.set_xlabel('Effect Size\n(Rank-Biserial)', fontsize=10)
            
            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dddddd')
            
            ax.set_xlim(-0.6, 0.6) # Standardized limits for comparability
            
        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels(labels, fontsize=9)
        for tick, entry in zip(axes[0].get_yticklabels(), row_entries):
            if entry["type"] == "header":
                tick.set_fontweight("bold")
                tick.set_fontsize(10)
        axes[0].set_ylabel('Comparison / Archetype Rule', fontsize=12)
        
        # Add custom legend to the whole figure
        legend_elements = [
            Patch(facecolor=colors['SM'], label='SM Error vs Success'),
            Patch(facecolor=colors['NM'], label='NM Error vs Success'),
            Patch(facecolor='gray', alpha=1.0, label='Displayed rows: q < 0.05')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        os.makedirs(MANUSCRIPT_FIGURES_DIR, exist_ok=True)
        out_path = os.path.join(MANUSCRIPT_FIGURES_DIR, f"meta_features_forest_plot_{ds.lower()}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to {out_path}")

if __name__ == '__main__':
    df = load_data()
    generate_plot(df)
