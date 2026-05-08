
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
import warnings
warnings.filterwarnings('ignore')

RENAME = {
  "rule_str": "Archetype rule",
  "coverage": "N",
  "coverage_pct": "Coverage (%)",
  "lift": "Enrichment (lift)",
  "quality_WRAcc": "WRAcc",
  "target_share": "Outcome in subgroup (%)",
  "baseline_rate": "Baseline rate (%)",
  "p_value": "p",
  "q_value": "q (BH-FDR)",
  "n_conditions": "Conditions",
  "source_depth": "Depth",
  "analysis_key": "Comparison",
}

def infer_dataset_tag(path: Path) -> str:
  # Use containing directory (e.g., h2_results vs h2_results_readmission_30)
  for parent in [path.parent, path.parent.parent]:
    if parent and parent.name:
      if "readmission" in parent.name.lower():
        return "readmission_30"
      if parent.name.lower().startswith("h2_results"):
        return "mortality"
  return path.parent.name or "dataset"

def map_theme(rule: str) -> str:
  s = rule.lower()
  themes = [
    ("prolonged_coagulation", "Coagulation derangement"),
    ("hepatic_lab", "Liver dysfunction"),
    ("low_albumin", "Low albumin"),
    ("neutrophil", "Neutrophil count state"),
    ("mechanical_ventilation", "On ventilator"),
    ("creatinine_elevation", "Creatinine elevation"),
    ("sirs", "SIRS"),
    ("cvp_or_pap", "Invasive hemodynamic monitoring"),
  ]
  for k, v in themes:
    if k in s:
      return v
  return "Other"

# Human-readable names for variables and enumerated values
NAME_MAP = {
  "blood_pressure_state": "Blood pressure state",
  "heart_rate_state": "Heart rate state",
  "has_elevated_troponin": "Elevated troponin",
  "decreased_cardiac_output": "Decreased cardiac output",
  "shock_index": "Shock index (HR/SBP)",
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
  "respiratory_rate_state": "Respiratory rate state",
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
  "anion_gap_corrected": "Albumin-corrected anion gap",
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
  "hr_volatility": "Heart rate volatility",
  "sbp_volatility": "Systolic blood pressure volatility",
  "map_volatility": "MAP volatility",
  "rr_volatility": "Respiratory rate volatility",
  "data_density_score": "Data density score",
  "sirs_rr_criterion": "SIRS respiratory-rate criterion",
  "sirs_wbc_criterion": "SIRS white blood cell criterion",
  "sirs_temp_criterion": "SIRS temperature criterion",
  "sirs_hr_criterion": "SIRS heart-rate criterion",
  "meets_sirs_criteria": "Meets SIRS criteria",
  "has_effusion_fluid_labs": "Effusion fluid labs",
}

VALUE_MAP = {
  ("hepatic_lab_abnormality_type", "Synthetic_Dysfunction"): "Synthetic hepatic dysfunction",
  ("hemoglobin_level_category", "Moderate"): "Moderate anemia",
  ("creatinine_elevation_stage", "Stage_2"): "Creatinine elevation stage 2",
  ("creatinine_elevation_stage", "Stage_3"): "Creatinine elevation stage 3",
}

def beautify_var_name(name: str) -> str:
  if name in NAME_MAP: return NAME_MAP[name]
  # Fallback: turn snake_case to Title Case while keeping common medical abbreviations
  words = name.replace("has_", "").replace("is_", "").replace("_", " ").strip()
  words = re.sub(r"\bcreatinine\b", "creatinine", words, flags=re.IGNORECASE)
  words = re.sub(r"\bwbc\b", "WBC", words, flags=re.IGNORECASE)
  words = re.sub(r"\brr\b", "respiratory rate", words, flags=re.IGNORECASE)
  words = re.sub(r"\baki\b", "creatinine elevation", words, flags=re.IGNORECASE)
  return words[:1].upper() + words[1:]

def pretty_condition(expr: str) -> str:
  expr = expr.strip()
  # Interval form: feature: [a:b[
  m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*([\[\(])\s*([^:]+)\s*:\s*([^\]\)]+)\s*([\]\)])$", expr)
  if m:
    var, lbr, a, b, rbr = m.groups()
    name = beautify_var_name(var)
    # Use en-dash and bracket semantics
    interval = f"{a}–{b}"
    lb = ">=" if lbr == "[" else ">"
    rb = "<=" if rbr == "]" else "<"
    return f"{name} in {lb} {a}, {rb} {b}"

  # Equality to quoted string
  m = re.match(r"^([A-Za-z0-9_]+)\s*==\s*'([^']+)'\s*$", expr)
  if m:
    var, val = m.groups()
    if (var, val) in VALUE_MAP:
      return VALUE_MAP[(var, val)]
    name = beautify_var_name(var)
    return f"{name} = {val.replace('_', ' ')}"

  # Equality to boolean
  m = re.match(r"^([A-Za-z0-9_]+)\s*==\s*(True|False)\s*$", expr)
  if m:
    var, b = m.groups()
    name = beautify_var_name(var)
    if b == "True":
      # Try to remove redundant words if already phrased
      if name.lower().startswith("on ") or name.lower().startswith("invasive"):
        return name
      return f"{name} present"
    else:
      if name.lower().startswith("on "):
        return f"Not {name.lower()}"
      return f"No {name.lower()}"

  # Greater/less than numeric
  m = re.match(r"^([A-Za-z0-9_]+)\s*([<>]=?)\s*([0-9.]+)\s*$", expr)
  if m:
    var, op, val = m.groups()
    name = beautify_var_name(var)
    return f"{name} {op} {val}"

  # Fallback: cleanup
  expr = expr.replace("==", "=")
  return beautify_var_name(expr)

def pretty_rule(rule: str) -> list[str]:
  if not isinstance(rule, str) or not rule:
    return []
  parts = re.split(r"\s+AND\s+", rule)
  return [pretty_condition(p) for p in parts if p.strip()]

def pretty_analysis_key(key: str) -> str:
  s = str(key or "").strip()
  sl = s.lower()
  exact = {
    "sm_false_alarm": "SM false alarms (FP)",
    "nm_false_alarm": "NM false alarms (FP)",
    "sm_miss": "SM misses (FN)",
    "nm_miss": "NM misses (FN)",
    "sm_win": "SM advantage",
    "nm_win": "NM advantage",
  }
  if sl in exact:
    return exact[sl]
  if "false_alarm" in sl and "sm" in sl:
    return "SM false alarms (FP)"
  if "false_alarm" in sl and "nm" in sl:
    return "NM false alarms (FP)"
  if "miss" in sl and "sm" in sl:
    return "SM misses (FN)"
  if "miss" in sl and "nm" in sl:
    return "NM misses (FN)"
  if "survivor" in sl and "sm" in sl:
    return "SM advantage—survivors"
  if "survivor" in sl and "nm" in sl:
    return "NM advantage—survivors"
  if "deceased" in sl and "sm" in sl:
    return "SM advantage—deceased"
  if "deceased" in sl and "nm" in sl:
    return "NM advantage—deceased"
  # Fallback: make human-readable
  return s.replace("_", " ").strip().capitalize()

def load_and_tidy(csv_path: Path) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  if "analysis_key" not in df.columns and "analysis_family" in df.columns:
    df["analysis_key"] = df["analysis_family"]
  df["dataset"] = infer_dataset_tag(csv_path)
  df["file_tag"] = csv_path.stem
  for c in ["lift", "q_value", "coverage", "coverage_pct", "quality_WRAcc", "target_share", "baseline_rate"]:
    if c in df.columns:
      df[c] = pd.to_numeric(df[c], errors="coerce")
  if "rule_str" in df.columns:
    df["theme"] = df["rule_str"].astype(str).map(map_theme)
  else:
    df["theme"] = "Other"
  df["log2_lift"] = np.log2(df["lift"].clip(lower=1e-12)) if "lift" in df.columns else np.nan
  df["neg_log10_q"] = -np.log10(df["q_value"].clip(lower=1e-300)) if "q_value" in df.columns else np.nan
  return df

def apply_filters(df: pd.DataFrame, q: float, cov: float, lift: float) -> pd.DataFrame:
  keep = pd.Series(True, index=df.index)
  if "q_value" in df.columns: keep &= df["q_value"] < q
  if "coverage_pct" in df.columns: keep &= df["coverage_pct"] >= cov
  if "lift" in df.columns: keep &= df["lift"] >= lift
  return df.loc[keep].copy()

def create_custom_colormap():
    """Create beautiful custom colormaps for different themes"""
    # Medical theme colors - sophisticated and professional
    colors = {
        'Creatinine elevation': '#E74C3C',
        'SIRS': '#E67E22',
        'On ventilator': '#3498DB',
        'Coagulation derangement': '#9B59B6',
        'Liver dysfunction': '#F39C12',
        'Low albumin': '#27AE60',
        'Neutrophilia': '#FF6B9D',
        'Invasive hemodynamic monitoring': '#17A2B8',
        'Other': '#6C757D'          # Gray
    }
    return colors

def style_matplotlib():
    """Set up sophisticated matplotlib styling"""
    plt.style.use('default')
    
    # Custom color palette
    colors = ['#2E4057', '#048A81', '#54C6EB', '#F18F01', '#C73E1D']
    sns.set_palette(colors)
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFBFC',
        'axes.edgecolor': '#E1E8ED',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#E1E8ED',
        'grid.linewidth': 0.8,
        'axes.titlesize': 22,
        'axes.titleweight': 'bold',
        'axes.titlecolor': '#2C3E50',
        'axes.labelsize': 16,
        'axes.labelweight': '500',
        'axes.labelcolor': '#34495E',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.color': '#7F8C8D',
        'ytick.color': '#7F8C8D',
        'legend.fontsize': 13,
        'legend.title_fontsize': 14,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.95,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.weight': 'normal',
        'font.size': 14,
    })

def save_top_table(df: pd.DataFrame, out_dir: Path, tag: str, top_k: int):
  def latex_escape(text: str) -> str:
    if pd.isna(text):
      return ""
    s = str(text)
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    s = s.replace("#", r"\#").replace("{", r"\{").replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return s

  cols = [c for c in [
    "analysis_key", "rule_str", "coverage", "coverage_pct", "target_share", "baseline_rate", "lift", "q_value"
  ] if c in df.columns]

  if not cols:
    return

  tidy = df[cols].copy().rename(columns=RENAME)
  # Add pretty comparison label and grouping key
  if "Comparison" in tidy.columns:
    tidy["Comparison_pretty"] = tidy["Comparison"].astype(str).map(pretty_analysis_key)
  elif "analysis_key" in tidy.columns:
    tidy["Comparison_pretty"] = tidy["analysis_key"].astype(str).map(pretty_analysis_key)
  else:
    tidy["Comparison_pretty"] = ""

  # Define stable comparison grouping order
  comparison_order = [
    "SM advantage—deceased",
    "NM advantage—deceased",
    "SM advantage—survivors",
    "NM advantage—survivors",
    "SM false alarms (FP)",
    "NM false alarms (FP)",
    "SM misses (FN)",
    "NM misses (FN)",
  ]
  order_map = {name: i for i, name in enumerate(comparison_order)}
  tidy["Comparison_group"] = tidy["Comparison_pretty"].map(lambda x: order_map.get(x, len(order_map)))

  # Sort by group, then significance, then effect size
  sort_cols = [
    "Comparison_group",
  ] + [c for c in ["q (BH-FDR)", "Enrichment (lift)"] if c in tidy.columns]
  sort_asc = [True] + [True, False][: len(sort_cols) - 1]
  tidy = tidy.sort_values(sort_cols, ascending=sort_asc).head(top_k)

  # Prepare LaTeX content
  out_dir.mkdir(parents=True, exist_ok=True)
  tex_path = out_dir / f"top_archetypes_{tag}.tex"

  headers = [
    ("Comparison", "l"),
    ("Archetype rule", "l"),
    ("N", "r"),
    ("Coverage (\\%)", "r"),
    ("Outcome (\\%)", "r"),
    ("Baseline (\\%)", "r"),
    ("Enrichment", "r"),
    ("q (BH-FDR)", "r"),
  ]

  def fmt_int(x):
    return "-" if pd.isna(x) else f"{int(x):,}"

  def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x:.1f}\\%"

  def fmt_lift(x):
    return "-" if pd.isna(x) else f"{x:.2f}"

  def fmt_q(x):
    if pd.isna(x):
      return "-"
    try:
      return f"{x:.2e}" if float(x) < 0.001 else f"{x:.3f}"
    except Exception:
      return latex_escape(str(x))

  rows = []
  for _, r in tidy.iterrows():
    comparison = latex_escape(r.get("Comparison_pretty", r.get("analysis_key", "")))
    rule_raw = r.get("Archetype rule", r.get("rule_str", ""))
    rule_pretty = "; ".join(pretty_rule(rule_raw)) if isinstance(rule_raw, str) else ""
    rule = latex_escape(rule_pretty)
    n_val = fmt_int(r.get("N", r.get("coverage", np.nan)))
    cov_val = fmt_pct(r.get("Coverage (%)", r.get("coverage_pct", np.nan)))
    out_val = fmt_pct(r.get("Outcome in subgroup (%)", r.get("target_share", np.nan)))
    base_val = fmt_pct(r.get("Baseline rate (%)", r.get("baseline_rate", np.nan)))
    lift_val = fmt_lift(r.get("Enrichment (lift)", r.get("lift", np.nan)))
    q_val = fmt_q(r.get("q (BH-FDR)", r.get("q_value", np.nan)))
    rows.append([comparison, rule, n_val, cov_val, out_val, base_val, lift_val, q_val])

  col_spec = "@{}" + "".join(col for _, col in headers) + "@{}"
  header_line = " & ".join(h for h, _ in headers) + r" \\"

  body_lines = [" & ".join(map(str, row)) + r" \\" for row in rows]

  content = []
  content.append(r"\documentclass[border=0pt]{standalone}")
  content.append(r"\usepackage{booktabs}")
  content.append(r"\usepackage[T1]{fontenc}")
  content.append(r"\usepackage[utf8]{inputenc}")
  content.append(r"\usepackage{adjustbox}")
  content.append(r"\begin{document}")
  content.append(r"\begin{adjustbox}{width=\textwidth}")
  content.append(fr"\begin{{tabular}}{{{col_spec}}}")
  content.append(r"\toprule")
  content.append(header_line)
  content.append(r"\midrule")
  content.extend(body_lines)
  content.append(r"\bottomrule")
  content.append(r"\end{tabular}")
  content.append(r"\end{adjustbox}")
  content.append(r"\end{document}")

  tex_path.write_text("\n".join(content), encoding="utf-8")

def create_elegant_cards(df: pd.DataFrame, out_dir: Path, tag: str, top_k: int = 8):
    """Create stunning archetype cards with proper spacing and layout"""
    if df.empty:
        return
    
    theme_colors = create_custom_colormap()
    
    for ak, dsub in df.groupby("analysis_key", dropna=False):
        dtop = dsub.sort_values(["q_value", "lift"], ascending=[True, False]).head(top_k)
        n = len(dtop)
        
        # Calculate optimal layout with better spacing
        rows = max(1, int(np.ceil(n/2)))
        card_height = 9.2
        fig_height = rows * card_height + 1.6
        
        fig = plt.figure(figsize=(22, fig_height), facecolor='#FAFBFC')
        gs = gridspec.GridSpec(
            nrows=rows, 
            ncols=2, 
            figure=fig,
            hspace=0.10,
            wspace=0.08,
            left=0.03, 
            right=0.97, 
            top=0.94, 
            bottom=0.03
        )
        
        # Determine prevalent theme
        theme_keywords = [
            ("anticoagulation", "Coagulation derangement"),
            ("liver_dysfunction", "Liver dysfunction"),
            ("malnour", "Low albumin"),
            ("neutrophil", "Neutrophilia"),
            ("ventilator", "On ventilator"),
            ("creatinine", "Creatinine elevation"),
            ("sirs", "SIRS"),
            ("hemo_monitor", "Invasive hemodynamic monitoring"),
        ]
        counts = {k: 0 for k, _ in theme_keywords}
        for rs in dtop.get("rule_str", pd.Series(dtype=str)).astype(str).fillna(""):
            srs = rs.lower()
            for k, _ in theme_keywords:
                if k in srs:
                    counts[k] += 1
        prevalent_key = None
        prevalent_theme_name = None
        if counts and max(counts.values()) > 0:
            prevalent_key = max(counts, key=lambda x: counts[x])
            prevalent_theme_name = dict(theme_keywords)[prevalent_key]

        # Special case handling
        force_ivb_theme = str(ak) == "IVB_survivors_NM" and str(tag).startswith("mortality_final_archetypes_ivb")

        for i, (_, row) in enumerate(dtop.iterrows()):
            ax = fig.add_subplot(gs[i//2, i%2])
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Determine theme
            if force_ivb_theme:
                theme = row.get("theme", "Other") if i == 7 else "Invasive hemodynamic monitoring"
            else:
                rule_text_lc = str(row.get("rule_str", "")).lower()
                if prevalent_key and prevalent_key in rule_text_lc:
                    theme = prevalent_theme_name
                else:
                    theme = row.get("theme", "Other")
            primary_color = theme_colors.get(theme, theme_colors['Other'])
            
            # Create main card background
            gradient_bg = FancyBboxPatch(
                (0.1, 0.1), 9.8, 9.8,
                boxstyle="round,pad=0.05",
                facecolor='white',
                edgecolor='none',
                alpha=0.98,
                mutation_scale=15,
                zorder=1
            )
            ax.add_patch(gradient_bg)
            
            # Main card with colored border
            card = FancyBboxPatch(
                (0.1, 0.1), 9.8, 9.8,
                boxstyle="round,pad=0.05",
                facecolor='none',
                edgecolor=primary_color,
                linewidth=3.0,
                alpha=1,
                mutation_scale=15,
                zorder=2
            )
            ax.add_patch(card)
            
            # Theme header bar
            header = Rectangle(
                (0.1, 9.0), 9.8, 0.9,
                facecolor=primary_color,
                edgecolor='none',
                alpha=1,
                zorder=3
            )
            ax.add_patch(header)
            
            # Rank badge
            rank_bg = Circle((0.8, 9.35), 0.36, 
                           facecolor='white', 
                           edgecolor=primary_color, 
                           linewidth=3.0,
                           zorder=5)
            ax.add_patch(rank_bg)
            
            # Rank number
            rank_text = ax.text(0.8, 9.35, f"#{i+1}", 
                              ha='center', va='center', 
                              fontsize=26, fontweight='bold',
                              color=primary_color,
                              zorder=6)
            rank_text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
            
            # Theme label
            ax.text(1.5, 9.35, theme.upper(), 
                   ha='left', va='center', 
                   fontsize=24, fontweight='bold',
                   color='white',
                   zorder=4)
            
            # Clinical Conditions section
            ax.text(0.7, 8.4, "Clinical Conditions", 
                   fontsize=24, fontweight='bold',
                   color='#111827')
            
            # Line under conditions header
            ax.plot([0.7, 9.3], [8.2, 8.2], 
                   color=primary_color, linewidth=2.0, alpha=0.6)
            
            # Display conditions
            bullets = pretty_rule(row.get("rule_str", ""))
            y_pos = 7.9
            max_conditions = 4
            line_spacing = 0.7
            
            for j, condition in enumerate(bullets[:max_conditions]):
                # Bullet point
                ax.text(0.8, y_pos, "•", 
                       fontsize=22, va='center',
                       color=primary_color, fontweight='bold')
                
                # Condition text
                if len(condition) > 50:
                    # Split long conditions
                    first_part = condition[:50]
                    ax.text(1.2, y_pos, first_part, 
                           fontsize=20, va='center',
                           color='#374151', fontweight='500')
                    y_pos -= 0.45
                    second_part = condition[50:90]
                    if second_part.strip():
                        ax.text(1.2, y_pos, "  " + second_part + ("..." if len(condition) > 90 else ""), 
                               fontsize=20, va='center',
                               color='#374151', fontweight='500')
                else:
                    ax.text(1.2, y_pos, condition, 
                           fontsize=20, va='center',
                           color='#374151', fontweight='500')
                
                y_pos -= line_spacing
                
            if len(bullets) > max_conditions:
                ax.text(1.2, y_pos, f"+ {len(bullets)-max_conditions} more condition{'s' if len(bullets)-max_conditions > 1 else ''}", 
                       fontsize=16, va='center',
                       color='#9CA3AF', style='italic')
                y_pos -= 0.3
            
            # Key Statistics section with proper spacing
            stats_top = y_pos - 0.3
            stats_bottom = 0
            stats_height = stats_top - stats_bottom
            
            # Ensure minimum height (fit header + rows comfortably)
            if stats_height < 4.6:
                stats_height = 4.6
                stats_top = stats_bottom + stats_height
            
            # Statistics background
            stats_bg = FancyBboxPatch(
                (0.5, stats_bottom), 9.0, stats_height,
                boxstyle="round,pad=0.08",
                facecolor=primary_color,
                alpha=0.06,
                edgecolor='none',
                mutation_scale=8
            )
            ax.add_patch(stats_bg)
            
            # Statistics header with proper spacing
            header_y = stats_top - 0.25
            ax.text(5.0, header_y, "Key Statistics", 
                   ha='center', fontsize=22, fontweight='bold',
                   color='#111827')
            
            # Header underline
            ax.plot([1.5, 8.5], [header_y - 0.2, header_y - 0.2], 
                   color=primary_color, linewidth=2.0, alpha=0.6)
            
            # Format metrics
            N = int(row.get("coverage", np.nan)) if pd.notnull(row.get("coverage")) else "N/A"
            cov_pct = row.get("coverage_pct", np.nan)
            cov = f"{cov_pct:.1f}%" if pd.notnull(cov_pct) else "N/A"
            
            tgt_share = row.get("target_share", np.nan)
            base_rate = row.get("baseline_rate", np.nan)
            tgt = f"{tgt_share:.1f}%" if pd.notnull(tgt_share) else "N/A"
            base = f"{base_rate:.1f}%" if pd.notnull(base_rate) else "N/A"
            
            lift_val = row.get("lift", np.nan)
            lift = f"{lift_val:.2f}×" if pd.notnull(lift_val) else "N/A"
            
            q_val = row.get("q_value", np.nan)
            if pd.notnull(q_val):
                if q_val < 0.001:
                    q_str = f"{q_val:.2e}"
                else:
                    q_str = f"{q_val:.3f}"
            else:
                q_str = "N/A"
            
            # Statistics data
            stats_data = [
                ("Patients:", f"{N:,}"),
                ("Coverage:", cov),
                ("Outcome:", f"{tgt} vs {base}"),
                ("Enrichment:", lift),
                ("p-value:", q_str)
            ]
            
            # Display statistics with proper spacing
            current_y = header_y - 0.7
            row_height = 0.8
            left_col_x = 1.2
            right_col_x = 4.8
            
            for label, value in stats_data:
                # Label
                ax.text(left_col_x, current_y, label, 
                       ha='left', va='center', 
                       fontsize=20, fontweight='600',
                       color='#4B5563')
                # Value
                ax.text(right_col_x, current_y, value, 
                       ha='left', va='center', 
                       fontsize=22, fontweight='bold',
                       color=primary_color)
                current_y -= row_height
        
        # Title
        fig.suptitle(
            f"Clinical Archetypes Overview — {str(ak)}",
            fontsize=36,
            fontweight='bold',
            color='#111827',
            y=0.982
        )
        
        # Subtitle
        fig.text(0.5, 0.945, 
                f"Top {n} highest-significance patient subgroups",
                ha='center', 
                fontsize=20,
                color='#6B7280',
                style='italic')
        
        # Save figure
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_ak = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ak) if pd.notnull(ak) else "all")
        
        plt.savefig(
            out_dir / f"archetype_cards_{tag}_{safe_ak}.pdf",
            dpi=300,
            bbox_inches='tight',
            facecolor='#FAFBFC',
            edgecolor='none',
            pad_inches=0
        )
        plt.close(fig)

def process_file(path: Path, out_root: Path, q: float, cov: float, lift: float, top_k: int):
  raw = load_and_tidy(path)
  if raw.empty:
    print(f"     -> Skipping empty file: {path.name} (no archetypes found)")
    return
  tag = f"{raw['dataset'].iat[0]}_{path.stem}"
  fdf = apply_filters(raw, q, cov, lift)
  
  # Save table & create all visualizations
  tables_dir = out_root / raw["dataset"].iat[0] / "tables"
  figs_dir = out_root / raw["dataset"].iat[0] / "figs"
  
  save_top_table(fdf, tables_dir, tag, top_k)
  
  # Set up beautiful styling
  style_matplotlib()
  
  # Generate archetype cards only
  create_elegant_cards(fdf, figs_dir, tag, top_k=min(top_k, 12))

def discover_csvs(root_dirs: list[str]) -> list[Path]:
  csvs = []
  for r in root_dirs:
    p = Path(r)
    csvs += list(p.rglob("final_archetypes.csv"))
    csvs += list(p.rglob("final_archetypes_ivb.csv"))
  return sorted(set(csvs))

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--roots", nargs="+", required=False, help="Root directories to scan for CSVs")
  ap.add_argument("--out", required=False, help="Output root directory")
  ap.add_argument("--q", type=float, default=0.05)
  ap.add_argument("--coverage", type=float, default=5.0)
  ap.add_argument("--lift", type=float, default=1.5)
  ap.add_argument("--top_k", type=int, default=50)
  args = ap.parse_args()

  base = Path(__file__).resolve().parent
  default_roots = [
    str(base / "h2_results"),
    str(base / "h2_results_readmission_30"),
  ]
  roots = args.roots if args.roots else default_roots
  out_root = Path(args.out) if args.out else (base / "outputs")

  csvs = discover_csvs(roots)
  if not csvs:
    raise SystemExit("No result CSVs found under provided roots.")
  
  print(f"Generating publication-quality visualizations...")
  print(f"Processing {len(csvs)} archetype files...")
  
  for i, c in enumerate(csvs, 1):
    print(f"   [{i}/{len(csvs)}] {c.name}")
    process_file(c, out_root, args.q, args.coverage, args.lift, args.top_k)
  
  print(f"Generated stunning visualizations in {out_root}")
  print(f"Output structure:")
  print(f"   |-- tables/           (Standalone LaTeX tables: top_archetypes_*.tex)")
  print(f"   `-- figs/")
  print(f"       `-- archetype_cards_*.pdf      (Elegant archetype cards)")

if __name__ == "__main__":
  main()