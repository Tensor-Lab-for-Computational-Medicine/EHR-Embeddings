import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.transforms as transforms

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

META_FEAT_MAP = {
    'total_measurement_events': 'Total\nevents',
    'unique_feature_count': 'Unique\nfeatures',
    'temporal_concentration': 'Late-Window\nConcentration',
    'aggregate_stddev': 'Volatility\n(StdDev)',
    'aggregate_slope': 'Trend\nMagnitude',
    'imputation_proportion': 'Imputation\n%',
}

FAMILIES = {
    'total_measurement_events': 'Density',
    'unique_feature_count': 'Density',
    'temporal_concentration': 'Temporal',
    'aggregate_stddev': 'Volatility',
    'aggregate_slope': 'Volatility',
    'imputation_proportion': 'Imputation',
}

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Mortality
    p_mort = os.path.join(base_dir, 'h2_results', 'phase_v_meta', 'phase_v_meta_results_IV.csv')
    df_mort = pd.read_csv(p_mort) if os.path.exists(p_mort) else pd.DataFrame()
    if not df_mort.empty: df_mort['dataset'] = 'Mortality'
        
    # Readmission
    p_readm = os.path.join(base_dir, 'h2_results_readmission_30', 'phase_v_meta', 'phase_v_meta_results_IV.csv')
    df_readm = pd.read_csv(p_readm) if os.path.exists(p_readm) else pd.DataFrame()
    if not df_readm.empty: df_readm['dataset'] = 'Readmission'
        
    df = pd.concat([df_mort, df_readm], ignore_index=True)
    if df.empty: return df
    
    df['rule_pretty'] = df['rule'].apply(pretty_rule)
    df['meta_pretty'] = df['meta_feature'].map(META_FEAT_MAP)
    df['family_pretty'] = df['meta_feature'].map(FAMILIES)
    
    # Extract Model
    df['Model'] = df['analysis'].apply(lambda x: 'SM' if 'SM' in x else ('NM' if 'NM' in x else 'Other'))
    
    # Rank biserial goes from -1 to 1. 
    # Let's verify: a positive rank biserial means "higher in error" according to h2v_meta_analysis.py.
    
    return df

def generate_plot(df):
    if df.empty:
        print("No data found.")
        return
        
    datasets = ['Mortality', 'Readmission']
    meta_features = [
        'total_measurement_events', 'unique_feature_count', 'temporal_concentration',
        'aggregate_stddev', 'aggregate_slope',
        'imputation_proportion'
    ]
    
    # Filter to only these meta features
    df = df[df['meta_feature'].isin(meta_features)]
    
    # Colors
    colors = {'SM': '#1f77b4', 'NM': '#ff7f0e'}  # Blue for SM, Orange for NM
    
    for ds in datasets:
        df_ds = df[df['dataset'] == ds].copy()
        if df_ds.empty:
            print(f"{ds} - No Data")
            continue
            
        fig = plt.figure(figsize=(16, max(6, len(df_ds['rule_pretty'].unique()) * 0.6)))
        fig.suptitle(f"Cohort: {ds}", fontsize=18, fontweight='bold', y=1.05)
        
        # Get unique archetypes sorted by frequency or just alphabetical
        rules = df_ds['rule_pretty'].unique()
        rules = sorted(rules, key=lambda x: len(x)) # Simple heuristic sort
        
        axes = fig.subplots(1, len(meta_features), sharey=True)
        if len(meta_features) == 1: axes = [axes]
        
        # We need a unified y-axis
        y_positions = np.arange(len(rules))[::-1]  # Reverse to have first at top
        rule_to_y = {r: y for r, y in zip(rules, y_positions)}
        
        for ax, mf in zip(axes, meta_features):
            df_mf = df_ds[df_ds['meta_feature'] == mf]
            
            # Title for the column
            title = f"{META_FEAT_MAP.get(mf, mf)}"
            ax.set_title(title, fontsize=12)
            
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
            
            # Plot bars
            height = 0.35
            
            for _, row in df_mf.iterrows():
                y = rule_to_y[row['rule_pretty']]
                val = row['rank_biserial']
                sig = row['q_value_family'] < 0.05
                mod = row['Model']
                
                c = colors.get(mod, 'gray')
                alpha = 1.0 if sig else 0.3
                
                # Offset y based on model
                y_off = y + height/2 if mod == 'SM' else y - height/2
                
                ax.barh(y_off, val, height=height, color=c, alpha=alpha, edgecolor='none')
                
            ax.set_xlabel('Effect Size\n(Rank-Biserial)', fontsize=10)
            
            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dddddd')
            
            ax.set_xlim(-0.6, 0.6) # Standardized limits for comparability
            
        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels(rules, fontsize=10)
        axes[0].set_ylabel('Archetype Rule', fontsize=12)
        
        # Add custom legend to the whole figure
        legend_elements = [
            Patch(facecolor=colors['SM'], label='SM Error vs Success'),
            Patch(facecolor=colors['NM'], label='NM Error vs Success'),
            Patch(facecolor='gray', alpha=1.0, label='Significant (q < 0.05)'),
            Patch(facecolor='gray', alpha=0.3, label='Not Significant')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', f'meta_features_forest_plot_{ds.lower()}.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to {out_path}")

if __name__ == '__main__':
    df = load_data()
    generate_plot(df)
