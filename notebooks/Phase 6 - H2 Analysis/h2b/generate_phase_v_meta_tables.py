from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np


RENAME = {
  "analysis": "Comparison",
  "rule": "Archetype rule",
  "meta_feature": "Meta-feature",
  "family": "Family",
  "direction": "Direction",
  "p_value": "p",
  "q_value": "q (BH-FDR)",
  "q_value_family": "q_family (per-family BH-FDR)",
  "median_subgroup_error": "Median (error)",
  "median_concordant_success": "Median (success)",
  "effect_size": "Effect",
  "n_error_in_subgroup": "n_err",
  "n_success_concordant": "n_suc",
}


def infer_dataset_tag(path: Path) -> str:
  for parent in [path.parent, path.parent.parent]:
    if parent and parent.name:
      if "readmission" in parent.name.lower():
        return "readmission_30"
      if parent.name.lower().startswith("h2_results"):
        return "mortality"
  return path.parent.name or "dataset"


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
  return s.replace("_", " ").strip().capitalize()


# Human-readable names for variables and enumerated values (shared with archetype reports)
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
  "temporal_concentration": "Late-window temporal concentration",
}

VALUE_MAP = {
  ("hepatic_lab_abnormality_type", "Synthetic_Dysfunction"): "Synthetic hepatic dysfunction",
  ("hemoglobin_level_category", "Moderate"): "Moderate anemia",
  ("creatinine_elevation_stage", "Stage_2"): "Creatinine elevation stage 2",
  ("creatinine_elevation_stage", "Stage_3"): "Creatinine elevation stage 3",
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
    if (var, val) in VALUE_MAP:
      return VALUE_MAP[(var, val)]
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

def pretty_rule(rule: str) -> list[str]:
  if not isinstance(rule, str) or not rule:
    return []
  parts = re.split(r"\s+AND\s+", rule)
  return [pretty_condition(p) for p in parts if str(p).strip()]

def latex_escape(text: str) -> str:
  if pd.isna(text):
    return ""
  s = str(text)
  s = s.replace("\\", r"\textbackslash{}")
  s = s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
  s = s.replace("#", r"\#").replace("{", r"\{").replace("}", r"\}")
  s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
  return s


def load_and_tidy_meta(csv_path: Path) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  # Normalize key columns
  if "analysis_key" in df.columns and "analysis" not in df.columns:
    df["analysis"] = df["analysis_key"]
  for c in [
    "p_value", "q_value", "q_value_family", "effect_size",
    "median_subgroup_error", "median_concordant_success",
  ]:
    if c in df.columns:
      df[c] = pd.to_numeric(df[c], errors="coerce")
  df["dataset"] = infer_dataset_tag(csv_path)
  df["file_tag"] = csv_path.stem
  # Helpful sort keys
  if "q_value_family" in df.columns:
    df["qf_sort"] = df["q_value_family"].fillna(1.0)
  else:
    df["qf_sort"] = 1.0
  df["p_sort"] = df.get("p_value", pd.Series(1.0, index=df.index)).fillna(1.0)
  df["abs_effect"] = df.get("effect_size", pd.Series(np.nan, index=df.index)).abs()
  return df


def apply_filters(df: pd.DataFrame, p: float, qf: float) -> pd.DataFrame:
  keep = pd.Series(True, index=df.index)
  if "p_value" in df.columns and p is not None:
    keep &= df["p_value"] < p
  if "q_value_family" in df.columns and qf is not None:
    keep &= df["q_value_family"] < qf
  return df.loc[keep].copy()


def save_meta_table(df: pd.DataFrame, out_dir: Path, tag: str, top_k: int):
  # Columns to include if present
  cols = [c for c in [
    "analysis", "rule", "meta_feature", "family", "direction",
    "median_subgroup_error", "median_concordant_success", "effect_size",
    "p_value", "q_value", "q_value_family", "n_error_in_subgroup", "n_success_concordant",
  ] if c in df.columns]
  if not cols:
    return

  tidy = df[cols].copy().rename(columns=RENAME)
  # Pretty comparison label
  if "Comparison" in tidy.columns:
    tidy["Comparison_pretty"] = tidy["Comparison"].astype(str).map(pretty_analysis_key)
  else:
    tidy["Comparison_pretty"] = ""

  # Sort by family q-value, then p-value, then |effect|
  sort_cols = [c for c in ["q_family (per-family BH-FDR)", "p", "Effect"] if c in tidy.columns]
  sort_asc = [True, True, False][: len(sort_cols)]
  tidy = tidy.sort_values(sort_cols, ascending=sort_asc).head(top_k)

  # Output path
  out_dir.mkdir(parents=True, exist_ok=True)
  tex_path = out_dir / f"top_meta_{tag}.tex"

  # Prepare prettified rule text for grouping
  rule_col = "Archetype rule" if "Archetype rule" in tidy.columns else ("Rule" if "Rule" in tidy.columns else None)
  tidy["Rule_pretty"] = ""
  if rule_col:
    tidy["Rule_pretty"] = tidy[rule_col].astype(str).map(lambda s: "; ".join(pretty_rule(s)) if isinstance(s, str) and s else "")

  # Prettify simple categorical/value columns for readability
  def pretty_direction(val: str) -> str:
    s = str(val or "").strip().lower()
    if "higher_in_error" in s: return "Higher in error"
    if "higher_in_success" in s: return "Higher in success"
    return s.replace("_", " ").capitalize() if s else ""

  tidy["Direction_pretty"] = tidy.get("Direction", pd.Series("", index=tidy.index)).astype(str).map(pretty_direction)
  tidy["Family_pretty"] = tidy.get("Family", pd.Series("", index=tidy.index)).astype(str).map(lambda x: beautify_var_name(x))
  tidy["Meta_pretty"] = tidy.get("Meta-feature", pd.Series("", index=tidy.index)).astype(str).map(lambda x: beautify_var_name(x))

  # LaTeX headers and formatting (omit Rule column; we will group by it)
  headers = [
    ("Comparison", "l"),
    ("Meta-feature", "l"),
    ("Feature family", "l"),
    ("Direction of difference", "l"),
    ("Median (error)", "r"),
    ("Median (success)", "r"),
    ("Effect size", "r"),
    ("p-value", "r"),
    ("q (per-family BH-FDR)", "r"),
    ("n (error)", "r"),
    ("n (success)", "r"),
  ]

  def fmt_num(x, nd=3):
    return "-" if pd.isna(x) else f"{float(x):.{nd}f}"

  def fmt_q(x):
    if pd.isna(x):
      return "-"
    try:
      xv = float(x)
      return f"{xv:.2e}" if xv < 1e-3 else f"{xv:.3f}"
    except Exception:
      return latex_escape(str(x))

  rows = []
  # Preserve group order based on first appearance in the sorted, truncated data
  rule_order = list(dict.fromkeys(tidy.get("Rule_pretty", pd.Series([""] * len(tidy))).tolist()))
  for grp in rule_order:
    grp_label = latex_escape(grp) if grp else "Unspecified"
    # Aggregate counts per rule for context
    gdf = tidy.loc[tidy["Rule_pretty"] == grp]
    try:
      grp_err = int(pd.to_numeric(gdf.get("n_err", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
      grp_suc = int(pd.to_numeric(gdf.get("n_suc", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
      counts_str = f" \\textit{{(n err: {grp_err:,}, n suc: {grp_suc:,})}}"
    except Exception:
      counts_str = ""
    rows.append([fr"\multicolumn{{{len(headers)}}}{{l}}{{\textbf{{Rule: {grp_label}}}{counts_str}}}"])
    for _, r in tidy.loc[tidy["Rule_pretty"] == grp].iterrows():
      comparison = latex_escape(r.get("Comparison_pretty", r.get("Comparison", "")))
      feat = latex_escape(r.get("Meta_pretty", r.get("Meta-feature", r.get("meta_feature", ""))))
      fam = latex_escape(r.get("Family_pretty", r.get("Family", r.get("family", ""))))
      direction = latex_escape(r.get("Direction_pretty", r.get("Direction", r.get("direction", ""))))
      med_err = fmt_num(r.get("Median (error)", r.get("median_subgroup_error", np.nan)))
      med_suc = fmt_num(r.get("Median (success)", r.get("median_concordant_success", np.nan)))
      effect = fmt_num(r.get("Effect", r.get("effect_size", np.nan)))
      pval = fmt_q(r.get("p", r.get("p_value", np.nan)))
      qf = fmt_q(r.get("q_family (per-family BH-FDR)", r.get("q_value_family", np.nan)))
      nerr = "-" if pd.isna(r.get("n_err")) else f"{int(r.get('n_err', 0)):,}"
      nsuc = "-" if pd.isna(r.get("n_suc")) else f"{int(r.get('n_suc', 0)):,}"
      rows.append([comparison, feat, fam, direction, med_err, med_suc, effect, pval, qf, nerr, nsuc])

  col_spec = "@{}" + "".join(col for _, col in headers) + "@{}"
  header_line = " & ".join(latex_escape(h) for h, _ in headers) + r" \\" 

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


def discover_meta_csvs(root_dirs: list[str]) -> list[Path]:
  csvs = []
  for r in root_dirs:
    p = Path(r)
    csvs += list(p.rglob("phase_v_meta_results_*.csv"))
  return sorted(set(csvs))


def process_file(path: Path, out_root: Path, p: float, qf: float, top_k: int):
  raw = load_and_tidy_meta(path)
  tag = f"{raw['dataset'].iat[0]}_{path.stem}"
  fdf = apply_filters(raw, p, qf)

  tables_dir = out_root / raw["dataset"].iat[0] / "tables"
  save_meta_table(fdf, tables_dir, tag, top_k)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--roots", nargs="+", required=False, help="Root directories to scan for phase_v_meta CSVs")
  ap.add_argument("--out", required=False, help="Output root directory")
  ap.add_argument("--p", type=float, default=None, help="Optional p-value filter (e.g., 0.05)")
  ap.add_argument("--qf", type=float, default=0.05, help="Per-family q-value filter (default 0.05)")
  ap.add_argument("--top_k", type=int, default=50)
  args = ap.parse_args()

  base = Path(__file__).resolve().parent
  default_roots = [
    str(base / "h2_results"),
    str(base / "h2_results_readmission_30"),
  ]
  roots = args.roots if args.roots else default_roots
  out_root = Path(args.out) if args.out else (base / "outputs")

  csvs = discover_meta_csvs(roots)
  if not csvs:
    raise SystemExit("No phase_v_meta result CSVs found under provided roots.")

  print("Generating LaTeX tables for phase_v_meta results...")
  print(f"Processing {len(csvs)} files...")

  for i, c in enumerate(csvs, 1):
    print(f"   [{i}/{len(csvs)}] {c.name}")
    process_file(c, out_root, args.p, args.qf, args.top_k)

  print(f"LaTeX tables written under {out_root}/*/tables/")


if __name__ == "__main__":
  main()


