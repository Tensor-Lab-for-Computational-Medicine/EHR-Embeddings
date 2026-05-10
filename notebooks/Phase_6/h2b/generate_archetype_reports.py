from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for utility imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from manuscript_table_utils import save_manuscript_html, TABLES_OUTPUT_DIR, format_3_sig_figs

RENAME = {
  "rule": "Archetype rule",
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

import re

VALUE_MAP = {
    ("marital_status", "Single"): "Single",
    ("marital_status", "Married"): "Married",
    ("marital_status", "Unknown"): "Marital status unknown",
    ("hepatic_lab_abnormality_type", "Synthetic_Dysfunction"): "Synthetic hepatic dysfunction",
    ("hemoglobin_level_category", "Moderate"): "Moderate anemia",
    ("creatinine_elevation_stage", "Stage_2"): "Creatinine elevation stage 2",
    ("creatinine_elevation_stage", "Stage_3"): "Creatinine elevation stage 3",
}

NAME_MAP = {
    "rr_volatility": "Respiratory rate volatility",
    "data_density_score": "Data density score",
    "sirs_rr_criterion": "SIRS respiratory-rate criterion",
    "sirs_wbc_criterion": "SIRS white blood cell criterion",
    "sirs_temp_criterion": "SIRS temperature criterion",
    "sirs_hr_criterion": "SIRS heart-rate criterion",
    "meets_sirs_criteria": "Meets SIRS criteria",
    "has_effusion_fluid_labs": "Effusion fluid labs",
}

def beautify_var_name(name: str) -> str:
    """Standardize variable names for display"""
    if pd.isna(name): return ""
    s = str(name).replace('_', ' ').replace('IVB', '').replace('IV', '').strip()
    words = s.split()
    words = " ".join([w.capitalize() if w.lower() not in ["and", "or", "on", "in", "to", "of"] else w.lower() for w in words])
    # Clinical overrides
    words = re.sub(r"\bbmi\b", "BMI", words, flags=re.IGNORECASE)
    words = re.sub(r"\bicu\b", "ICU", words, flags=re.IGNORECASE)
    words = re.sub(r"\bcreatinine\b", "creatinine", words, flags=re.IGNORECASE)
    words = re.sub(r"\bwbc\b", "WBC", words, flags=re.IGNORECASE)
    words = re.sub(r"\brr\b", "respiratory rate", words, flags=re.IGNORECASE)
    words = re.sub(r"\baki\b", "creatinine elevation", words, flags=re.IGNORECASE)
    return words[:1].upper() + words[1:]

def pretty_condition(expr: str) -> str:
    expr = expr.strip()
    # Regex for intervals like feat: [a:b[ or feat: [a:b]
    m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*([\[\(])\s*([^:]+)\s*:\s*([^\]\)|\[]+)\s*([\]\)|\[])$", expr)
    if m:
        var, lbr, a, b, rbr = m.groups()
        name = beautify_var_name(var)
        # Always use closed brackets for manuscript [a:b]
        return f"{name}: [{a}:{b}]"
    m = re.match(r"^([A-Za-z0-9_]+)\s*==\s*'([^']+)'\s*$", expr)
    if m:
        var, val = m.groups()
        if (var, val) in VALUE_MAP: return VALUE_MAP[(var, val)]
        name = beautify_var_name(var)
        return f"{name} = {val.replace('_', ' ')}"
    m = re.match(r"^([A-Za-z0-9_]+)\s*==?\s*(True|False)\s*$", expr)
    if m:
        var, b = m.groups()
        name = beautify_var_name(var)
        if str(b) == "True":
            if name.lower().startswith("on ") or name.lower().startswith("invasive"): return name
            return f"{name} present"
        else:
            if name.lower().startswith("on "): return f"Not {name.lower()}"
            return f"No {name.lower()}"
    m = re.match(r"^([A-Za-z0-9_]+)\s*([<>]=?)\s*([0-9.]+)\s*$", expr)
    if m:
        var, op, val = m.groups()
        name = beautify_var_name(var)
        return f"{name} {op} {val}"
    
    clean_expr = expr.replace("==", "=")
    if "=" in clean_expr:
        var_part = clean_expr.split("=")[0].strip()
        val_part = clean_expr.split("=")[1].strip()
        return f"{beautify_var_name(var_part)} = {val_part.replace('_', ' ')}"
        
    return beautify_var_name(clean_expr)

def pretty_analysis_key(key: str, dataset: str = "") -> str:
    """Human-readable comparison labels. IVB keys name mortality strata (deaths/survivors) but the
    same cohorts mean positive-outcome vs negative-outcome battlegrounds on readmission tasks."""
    s = str(key or "").strip().lower()
    ds = str(dataset or "").lower()
    is_readm = "readmission" in ds

    if is_readm:
        ivb_exact = {
            "ivb_survivors_sm": "Non-readmitted discordance — SM correct (NM false positive)",
            "ivb_survivors_nm": "Non-readmitted discordance — NM correct (SM false positive)",
            "ivb_deaths_sm": "Readmitted discordance — SM correct (NM miss)",
            "ivb_deaths_nm": "Readmitted discordance — NM correct (SM miss)",
        }
    else:
        ivb_exact = {
            "ivb_survivors_sm": "Survivors discordance — SM correct (NM false positive)",
            "ivb_survivors_nm": "Survivors discordance — NM correct (SM false positive)",
            "ivb_deaths_sm": "Deceased discordance — SM correct (NM miss)",
            "ivb_deaths_nm": "Deceased discordance — NM correct (SM miss)",
        }
    if s in ivb_exact:
        return ivb_exact[s]

    exact = {
        "sm_false_alarm": "SM False Positives (FP)",
        "nm_false_alarm": "NM False Positives (FP)",
        "sm_miss": "SM Misses (FN)",
        "nm_miss": "NM Misses (FN)",
        "sm_win": "SM Advantage",
        "nm_win": "NM Advantage",
        "ivb_deceased_sm": ivb_exact.get("ivb_deaths_sm", "Deceased discordance — SM correct (NM miss)"),
        "ivb_deceased_nm": ivb_exact.get("ivb_deaths_nm", "Deceased discordance — NM correct (SM miss)"),
    }
    if s in exact:
        return exact[s]
    
    for k, v in exact.items():
        if k in s: return v
        
    if "sm_win_mort" in s: return "SM Advantage—Deceased"
    if "nm_win_mort" in s: return "NM Advantage—Deceased"
    if "sm_win_surv" in s: return "SM Advantage—Survivors"
    if "nm_win_surv" in s: return "NM Advantage—Survivors"
    if "sm_fp" in s: return "SM False Positives (FP)"
    if "nm_fp" in s: return "NM False Positives (FP)"
    if "sm_fn" in s: return "SM Misses (FN)"
    if "nm_fn" in s: return "NM Misses (FN)"
    
    return s.replace("_", " ").strip().title()

def pretty_rule(rule_str: str) -> list[str]:
    if not isinstance(rule_str, str) or not rule_str: return []
    # Handle & or AND
    rule_str = re.sub(r"\s+AND\s+", " AND ", rule_str, flags=re.IGNORECASE)
    rule_str = rule_str.replace(" & ", " AND ")
    parts = rule_str.split(" AND ")
    return [pretty_condition(p) for p in parts if p.strip()]

def load_and_tidy(path: Path) -> pd.DataFrame:
  df = pd.read_csv(path)
  if df.empty: return df
  if "mortality" in str(path).lower() or "mort_hosp" in str(path).lower():
    df["dataset"] = "mortality"
  elif "readmission" in str(path).lower():
    df["dataset"] = "readmission"
  else:
    df["dataset"] = "other"
  
  if "analysis_key" in df.columns:
    df["theme"] = df["analysis_key"].apply(lambda x: "Comparative" if "ivb" in str(x).lower() else "Independent")
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


# Stable row order for IVB discordance panels (matches h2b_combined report ordering).
_IVB_ANALYSIS_KEY_ORDER = [
    "IVB_deaths_SM", "IVB_deaths_NM", "IVB_survivors_SM", "IVB_survivors_NM",
]


def save_top_table(df: pd.DataFrame, out_dir: Path, tag: str, top_k: int):
  cols = [c for c in [
    "analysis_key", "rule", "coverage", "coverage_pct", "target_share", "baseline_rate", "lift", "q_value"
  ] if c in df.columns]

  if not cols:
    return

  tidy = df[cols].copy()
  ak_raw = tidy["analysis_key"].astype(str).copy() if "analysis_key" in tidy.columns else None
  tidy = tidy.rename(columns=RENAME)

  dataset_hint = (
      str(df["dataset"].iloc[0])
      if "dataset" in df.columns and len(df)
      else tag.split("_")[0]
  )

  # Determine if comparative or independent based on filename or data
  is_comparative = "ivb" in tag.lower()
  mode_prefix = "Comparative" if is_comparative else "Independent"

  if ak_raw is not None:
    tidy["Comparison_pretty"] = ak_raw.map(lambda k: pretty_analysis_key(k, dataset_hint))
  else:
    tidy["Comparison_pretty"] = ""

  comparison_order = [
    "SM Advantage—Deceased",
    "NM Advantage—Deceased",
    "SM Advantage—Survivors",
    "NM Advantage—Survivors",
    "SM False Positives (FP)",
    "NM False Positives (FP)",
    "SM Misses (FN)",
    "NM Misses (FN)",
    "SM Advantage",
    "NM Advantage",
  ]
  order_map = {name: i for i, name in enumerate(comparison_order)}

  def _comparison_group_row(i: int) -> int:
    if ak_raw is not None:
      k = ak_raw.iloc[i]
      if k in _IVB_ANALYSIS_KEY_ORDER:
        return _IVB_ANALYSIS_KEY_ORDER.index(k)
    pretty = tidy["Comparison_pretty"].iloc[i]
    return order_map.get(pretty, len(order_map))

  tidy["Comparison_group"] = [ _comparison_group_row(i) for i in range(len(tidy)) ]

  sort_cols = ["Comparison_group"] + [c for c in ["q (BH-FDR)", "Enrichment (lift)"] if c in tidy.columns]
  sort_asc = [True] + [True, False][: len(sort_cols) - 1]
  tidy = tidy.sort_values(sort_cols, ascending=sort_asc).head(top_k)

  out_pct_hdr = "Win-side (%)" if is_comparative else "Outcome (%)"
  base_pct_hdr = "Test-wide baseline (%)" if is_comparative else "Baseline (%)"

  report_rows = []
  for _, r in tidy.iterrows():
    rule_raw = r.get("Archetype rule", r.get("rule_str", ""))
    rule_pretty = "; ".join(pretty_rule(rule_raw)) if isinstance(rule_raw, str) else ""

    report_rows.append({
        "Comparison": r.get("Comparison_pretty", r.get("Comparison", "")),
        "Archetype Rule": rule_pretty,
        "N": r.get("N", r.get("coverage", np.nan)),
        "Coverage (%)": r.get("Coverage (%)", r.get("coverage_pct", np.nan)),
        out_pct_hdr: r.get("Outcome in subgroup (%)", r.get("target_share", np.nan)),
        base_pct_hdr: r.get("Baseline rate (%)", r.get("baseline_rate", np.nan)),
        "Enrichment": r.get("Enrichment (lift)", r.get("lift", np.nan)),
        "q-value": r.get("q (BH-FDR)", r.get("q_value", np.nan)),
    })

  if not report_rows:
    return

  res_df = pd.DataFrame(report_rows)

  dataset_name = tag.split("_")[0].capitalize()
  title = (
      f"{mode_prefix} discordance archetypes: {dataset_name}"
      if is_comparative
      else f"{mode_prefix} failure modes: {dataset_name} analysis"
  )
  filename = f"Table_N_{dataset_name}_{mode_prefix}_Archetypes_{tag}"

  ivb_footnotes = [
      "IVB rows: subgroup discovery contrasts the two discordant slices (which model was correct when the models disagreed).",
      "Win-side (%) is the fraction of patients matching the rule who fall on the advantaged model’s side of that contrast — not the raw clinical outcome rate within the rule.",
      "Enrichment compares the win-side rate inside the rule to the win-side rate on the full test set; q-values are BH-FDR within the exported rule set.",
  ] if is_comparative else None

  save_manuscript_html(
      res_df,
      title,
      filename,
      TABLES_OUTPUT_DIR,
      table_number="N",
      compact=True,
      footnotes=ivb_footnotes,
  )

def create_elegant_cards(df: pd.DataFrame, out_dir: Path, tag: str, top_k: int = 8):
    """Create stunning archetype cards with proper spacing and layout"""
    if df.empty: return
    
    # Sort for impact: highest lift, lowest q
    df = df.sort_values(['lift', 'q_value'], ascending=[False, True]).head(top_k)
    
    # Group by analysis key for multi-page output
    for ak, group in df.groupby('analysis_key'):
        safe_ak = str(ak).replace('/', '_').replace('\\', '_')
        pretty_ak = pretty_analysis_key(ak, tag.split("_")[0])
        
        n_cards = len(group)
        cols = 2
        rows = (n_cards + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        if rows == 1: axes = [axes]
        axes = np.array(axes).flatten()
        
        for i, (_, row) in enumerate(group.iterrows()):
            ax = axes[i]
            # Card background
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='white', ec='#E1E4E8', lw=1, transform=ax.transAxes))
            
            # Text layout
            rule = pretty_rule(row['rule_str'])
            y_start = 0.85
            
            # Title: Rule conditions
            ax.text(0.05, y_start, "Archetype Profile", fontsize=12, fontweight='bold', color='#24292E', transform=ax.transAxes)
            
            # Conditions
            for j, cond in enumerate(rule[:4]): # Max 4 conditions for space
                ax.text(0.08, y_start - 0.12 - (j * 0.08), f"• {cond}", fontsize=10, color='#586069', transform=ax.transAxes)
                
            # Stats strip at bottom
            ax.add_patch(plt.Rectangle((0, 0), 1, 0.25, color='#F6F8FA', transform=ax.transAxes))
            
            stats = [
                (f"{format_3_sig_figs(row['lift'])}x", "Lift"),
                (f"{format_3_sig_figs(row['target_share'])}%", "Outcome"),
                (f"N={int(row['coverage']):,}", "Size")
            ]
            
            for k, (val, label) in enumerate(stats):
                ax.text(0.15 + k*0.3, 0.15, val, fontsize=14, fontweight='bold', color='#0366D6', transform=ax.transAxes)
                ax.text(0.15 + k*0.3, 0.05, label, fontsize=9, color='#586069', transform=ax.transAxes)
                
            ax.set_axis_off()
            
        # Hide empty axes
        for j in range(i + 1, len(axes)):
            axes[j].set_axis_off()
            
        plt.suptitle(f"Discovery: {pretty_ak}", fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
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
  # Save standalone HTML tables
  tables_dir = out_root / raw["dataset"].iat[0] / "tables"
  save_top_table(fdf, tables_dir, tag, top_k)

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
  
  print(f"Generating publication-quality visualizations and tables...")
  print(f"Processing {len(csvs)} archetype files...")
  
  for i, c in enumerate(csvs, 1):
    print(f"   [{i}/{len(csvs)}] {c.name}")
    process_file(c, out_root, args.q, args.coverage, args.lift, args.top_k)
  
  print(f"Generated Professional HTML tables in {TABLES_OUTPUT_DIR}")

if __name__ == "__main__":
  main()