import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys
import pickle

# Add parent directory to path for utility imports
# Since this is in notebooks/Phase_6/h2b/, we need to go up 3 levels to find notebooks/
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from manuscript_table_utils import save_manuscript_html, TABLES_OUTPUT_DIR, format_3_sig_figs

# Same pretty functions as in generate_archetype_reports.py for consistency
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
    "total_measurement_events": "Total measurement events",
    "unique_feature_count": "Number of measured feature families",
    "temporal_concentration": "Late-window measurement concentration",
    "aggregate_stddev": "Average within-stay variability",
    "aggregate_slope": "Average absolute trend magnitude",
    "imputation_proportion": "Unmeasured feature-family proportion",
}

VALUE_MAP = {
    ("hepatic_lab_abnormality_type", "Synthetic_Dysfunction"): "Synthetic hepatic dysfunction",
    ("hemoglobin_level_category", "Moderate"): "Moderate anemia",
    ("creatinine_elevation_stage", "Stage_2"): "Creatinine elevation stage 2",
    ("creatinine_elevation_stage", "Stage_3"): "Creatinine elevation stage 3",
}

def beautify_var_name(name: str) -> str:
    if name in NAME_MAP: return NAME_MAP[name]
    words = name.replace("has_", "").replace("is_", "").replace("_", " ").strip()
    words = re.sub(r"\bcreatinine\b", "creatinine", words, flags=re.IGNORECASE)
    words = re.sub(r"\bwbc\b", "WBC", words, flags=re.IGNORECASE)
    words = re.sub(r"\brr\b", "respiratory rate", words, flags=re.IGNORECASE)
    words = re.sub(r"\baki\b", "creatinine elevation", words, flags=re.IGNORECASE)
    return words[:1].upper() + words[1:]

def pretty_condition(expr: str) -> str:
    expr = expr.strip()
    # Handle interval formats (e.g., "age: [20:30[" or "age: [20:30]")
    m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*([\[\(])\s*([^:]+)\s*:\s*([^\]\)]+|[^\[]+)\s*([\]\)|\[])$", expr)
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
    m = re.match(r"^([A-Za-z0-9_]+)\s*==\s*(True|False)\s*$", expr)
    if not m:
        # Catch case where it might be ==True (double equals) or just =True
        m = re.match(r"^([A-Za-z0-9_]+)\s*==?\s*(True|False)\s*$", expr)
        
    if m:
        var, b = m.groups()
        name = beautify_var_name(var)
        if str(b) == "True":
            if name.lower().startswith("on ") or name.lower().startswith("invasive"): return name
            # Strip redundant 'present' if the name already implies a finding/state
            if any(x in name.lower() for x in ["elevation", "impairment", "abnormality", "low ", "high ", "meets ", "prolonged", "severe", "decreased", "increased"]):
                return name
            return f"{name} present"
        else:
            if name.lower().startswith("on "): return f"Not {name.lower()}"
            if any(x in name.lower() for x in ["elevation", "impairment", "abnormality", "low ", "high ", "meets ", "prolonged", "severe", "decreased", "increased"]):
                return f"No {name.lower()}"
            return f"No {name.lower()} present"
    m = re.match(r"^([A-Za-z0-9_]+)\s*([<>]=?)\s*([0-9.]+)\s*$", expr)
    if m:
        var, op, val = m.groups()
        name = beautify_var_name(var)
        return f"{name} {op} {val}"
    
    # Final fallback: replace any == with = and beautify
    clean_expr = expr.replace("==", "=")
    if "=" in clean_expr:
        var_part = clean_expr.split("=")[0].strip()
        val_part = clean_expr.split("=")[1].strip()
        return f"{beautify_var_name(var_part)} = {val_part.replace('_', ' ')}"
        
    return beautify_var_name(clean_expr)

def pretty_rule(rule: str) -> str:
    if not isinstance(rule, str) or not rule: return ""
    # Standardize spacing around AND
    rule = re.sub(r"\s+AND\s+", " AND ", rule, flags=re.IGNORECASE)
    parts = rule.split(" AND ")
    return "; ".join([pretty_condition(p) for p in parts if p.strip()])

def pretty_analysis_key(key: str, task_hint: str = "") -> str:
    s = str(key or "").strip().lower()
    
    # Determine terminology based on task
    is_readm = "readmission" in task_hint.lower()
    pos_term = "Readmitted" if is_readm else "Deceased"
    neg_term = "Non-Readmitted" if is_readm else "Survivors"
    
    # Explicit labels for specific analysis keys
    exact = {
        "sm_false_alarm": "Subgroup Analysis: SM False Positives",
        "nm_false_alarm": "Subgroup Analysis: NM False Positives",
        "sm_miss": "Subgroup Analysis: SM Misses",
        "nm_miss": "Subgroup Analysis: NM Misses",
        "sm_win": "Subgroup Analysis: SM Advantage",
        "nm_win": "Subgroup Analysis: NM Advantage",
        "ivb_survivors_sm": "Model Disagreement: SM False Positives (NM Correct)",
        "ivb_survivors_nm": "Model Disagreement: NM False Positives (SM Correct)",
        "ivb_deaths_sm": "Model Disagreement: SM Misses (NM Correct)",
        "ivb_deaths_nm": "Model Disagreement: NM Misses (SM Correct)",
        "ivb_deceased_sm": "Model Disagreement: SM Misses (NM Correct)",
        "ivb_deceased_nm": "Model Disagreement: NM Misses (SM Correct)",
    }
    if s in exact: return exact[s]
    
    # Handle variations in casing from the CSV
    for k, v in exact.items():
        if s == k.lower(): return v
    
    # Fallback for primary analysis keys if they appear
    is_comparative = "ivb" in task_hint.lower() or "discordance" in task_hint.lower() or "comparative" in task_hint.lower()
    prefix = "Model Disagreement: " if is_comparative else "Subgroup Analysis: "

    if "primary" in s:
        label = "Primary Analysis"
        if "survivor" in s: label = f"Primary Analysis ({neg_term})"
        elif "death" in s or "deceased" in s: label = f"Primary Analysis ({pos_term})"
        return prefix + label
        
    return prefix + s.replace("_", " ").strip().title()

def calculate_performance_gap(rules_df, task_type, base_dir):
    """
    Calculates the Targeted Performance Gap Ratio (ErrorRate_A / ErrorRate_B) 
    strictly within the relevant outcome class for discordance archetypes.
    """
    # 1. Determine paths
    if task_type == 'mortality':
        art_path = base_dir / "notebooks" / "Phase_6" / "h2a" / "h2_results" / "mort_hosp" / "h2a_to_h2b_artifact.pkl"
        pheno_path = base_dir / "notebooks" / "Phase_6" / "feature_engineering" / "artifacts" / "X_test_phenotypes.pkl"
    else: # readmission
        art_path = base_dir / "notebooks" / "Phase_6" / "h2a" / "h2_results" / "readmission_30" / "h2a_to_h2b_artifact.pkl"
        pheno_path = base_dir / "notebooks" / "Phase_6" / "feature_engineering" / "artifacts" / "X_test_phenotypes.pkl"

    if not art_path.exists() or not pheno_path.exists():
        return rules_df

    # 2. Load data
    try:
        with open(art_path, 'rb') as f:
            art = pickle.load(f)
    except Exception as e:
        print(f"   Warning: could not load artifact for performance-gap recalculation ({e}); using existing lift values.")
        return rules_df
    phenos = pd.read_pickle(pheno_path)
    
    y_true = art['y_true']
    nm_wrong = (art['nm_pred'] != y_true)
    sm_wrong = (art['sm_pred'] != y_true)
    
    # 3. Calculate Absolute Error Ratio (ErrorRate_A / ErrorRate_B)
    valid_indices = []
    new_lifts = []
    
    # Check if we have pre-calculated test_validated_ratio from the selector
    has_precalc = 'test_validated_ratio' in rules_df.columns
    
    for idx, row in rules_df.iterrows():
        if has_precalc:
            ratio = row['test_validated_ratio']
            if ratio > 1.001:
                valid_indices.append(idx)
                new_lifts.append(ratio)
            continue

        rule_str = row['rule_str']
        analysis_key = row['analysis_key'].lower()
        
        # Determine who is 'A' (the failing model)
        # Analysis keys look like 'Comparative: NM Discordance (Deceased)'
        # In Phase IVB (Comparative), the name indicates the WINNER.
        # "NM Discordance" -> NM Won, SM Failed.
        # "SM Discordance" -> SM Won, NM Failed.
        # In Phase IV (Independent), the name indicates the FAILING cohort.
        if "discordance" in analysis_key:
            is_nm_failing = "sm discordance" in analysis_key
        else:
            is_nm_failing = "nm " in analysis_key
        err_a = nm_wrong if is_nm_failing else sm_wrong
        err_b = sm_wrong if is_nm_failing else nm_wrong
        
        try:
            # Robust rule evaluation on test set
            parts = rule_str.split(" AND ")
            mask = pd.Series(True, index=phenos.index)
            for part in parts:
                part = part.strip()
                # Support spaces in variable names and both [a:b[ and [a:b] formats
                m_int = re.match(r"([^:]+):\s*([\[\(])\s*([^:]+)\s*:\s*([^\]\)|\[]+)\s*([\]\)|\[])$", part)
                if m_int:
                    col, lbr, start, end, rbr = m_int.groups()
                    col = col.strip()
                    mask &= (phenos[col] >= float(start)) & (phenos[col] < float(end))
                else:
                    if '==' in part:
                        col, val = [s.strip() for s in part.split('==')]
                        val = val.strip("'").strip('"')
                        if val.lower() == 'true': mask &= (phenos[col] == True)
                        elif val.lower() == 'false': mask &= (phenos[col] == False)
                        else:
                            val_clean = val.replace('_', ' ')
                            mask &= (phenos[col].astype(str).str.replace('_', ' ') == val_clean)
                    else:
                        mask &= (phenos[part] == True)
            
            # Filter by the target outcome class
            is_pos_class = any(x in analysis_key for x in ['deceased', 'readmitted', 'deaths'])
            target_val = 1 if is_pos_class else 0
            combined_mask = mask & (y_true == target_val)
            
            if combined_mask.sum() == 0:
                continue

            # Absolute Error Rates
            rate_a = err_a[combined_mask].mean()
            rate_b = err_b[combined_mask].mean()
            
            # Error Ratio
            if rate_b == 0:
                ratio = 2.0 if rate_a > 0 else 1.0 # Modest bound for absolute errors
            else:
                ratio = rate_a / rate_b
            
            # ONLY keep if ratio > 1.0 (validated on test set)
            if ratio > 1.001:
                valid_indices.append(idx)
                new_lifts.append(ratio)
        except Exception:
            continue
    
    filtered_df = rules_df.loc[valid_indices].copy()
    filtered_df['lift'] = new_lifts
    
    # SORT by the test-set Error Ratio (Impact)
    filtered_df = filtered_df.sort_values('lift', ascending=False)
    
    return filtered_df
            
    rules_df['lift'] = new_lifts
    return rules_df


def main():
    # Determine base directory relative to this script
    script_dir = Path(__file__).resolve().parent
    # If the script is already inside 'h2b', the base_dir should be script_dir
    # If the script is one level up, base_dir should be script_dir / 'h2b'
    base_dir = script_dir if script_dir.name == "h2b" else script_dir / "h2b"

    # Common footnotes for archetype tables
    ARCHETYPE_FOOTNOTES = [
        "Archetypes identified via subgroup discovery on model error patterns.",
        "q-values represent the significance of the subgroup error rate relative to the global population (Fisher's exact test), with Benjamini-Hochberg FDR correction.",
        "Coverage represents the proportion of patients in the total test population (N=5,648) who meet the archetype criteria (i.e., absolute rule prevalence).",
        "Error Ratio represents the ratio of the subgroup error rate to the overall test-population error rate."
    ]
    
    META_FOOTNOTES = [
        "Meta-features represent clinical characteristics (e.g., labs, vitals) that statistically distinguish the archetype subgroup.",
        "q-values represent the significance of the difference between the target population and its comparison cohort (Mann-Whitney U test), with per-family Benjamini-Hochberg FDR correction.",
        "Medians represent the raw clinical units (e.g., absolute counts, proportions, or magnitudes) for each meta-feature.",
        "Effect (Diff) represents the absolute difference in medians between the target population and the comparison cohort."
    ]
    
    file_groups = {
        "mortality_IV": {
            "path": base_dir / "h2_results" / "mort_hosp" / "final_archetypes.csv",
            "title": "Independent Mortality Failure Modes",
            "filename": "Table_S4_Mortality_Independent_Archetypes",
            "number": "S4"
        },
        "mortality_IVB": {
            "path": base_dir / "h2_results" / "mort_hosp" / "final_archetypes_ivb.csv",
            "title": "Comparative Mortality Discordance",
            "filename": "Table_S5_Mortality_Comparative_Discordance",
            "number": "S5"
        },
        "readmission_IV": {
            "path": base_dir / "h2_results" / "readmission_30" / "final_archetypes.csv",
            "title": "Independent Readmission Failure Modes",
            "filename": "Table_S8_Readmission_Independent_Archetypes",
            "number": "S8"
        },
        "readmission_IVB": {
            "path": base_dir / "h2_results" / "readmission_30" / "final_archetypes_ivb.csv",
            "title": "Comparative Readmission Discordance",
            "filename": "Table_S9_Readmission_Comparative_Discordance",
            "number": "S9"
        },
    }
    
    meta_groups = {
        "mortality_meta_IV": {
            "path": base_dir / "h2_results" / "mort_hosp" / "phase_v_meta" / "phase_v_meta_results_IV.csv",
            "title": "Independent Mortality Meta-Feature Analysis",
            "filename": "Table_S6_Mortality_Meta_Independent",
            "number": "S6"
        },
        "mortality_meta_IVB": {
            "path": base_dir / "h2_results" / "mort_hosp" / "phase_v_meta" / "phase_v_meta_results_IVB.csv",
            "title": "Comparative Mortality Meta-Analysis",
            "filename": "Table_S7_Mortality_Meta_Comparative",
            "number": "S7"
        },
        "readmission_meta_IV": {
            "path": base_dir / "h2_results" / "readmission_30" / "phase_v_meta" / "phase_v_meta_results_IV.csv",
            "title": "Independent Readmission Meta-Feature Analysis",
            "filename": "Table_S10_Readmission_Meta_Independent",
            "number": "S10"
        },
        "readmission_meta_IVB": {
            "path": base_dir / "h2_results" / "readmission_30" / "phase_v_meta" / "phase_v_meta_results_IVB.csv",
            "title": "Comparative Readmission Meta-Analysis",
            "filename": "Table_S11_Readmission_Meta_Comparative",
            "number": "S11"
        },
    }

    output_dir = base_dir / "manuscript_tables"
    output_dir.mkdir(exist_ok=True)


    print(f"\nGenerating {len(file_groups) + len(meta_groups)} separate manuscript tables...")

    paired_archetypes = {}
    paired_archetype_order = {}

    # Process Archetype Tables
    for key, config in file_groups.items():
        path = config["path"]
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["_meta_input_order"] = np.arange(len(df))
        
        # Calculate Targeted Performance Gap Ratio for Comparative tables (S5, S9)
        if config["number"] in ["S5", "S9"]:
            print(f"   Calculating Targeted Performance Gap Ratios for {config['number']}...")
            task_type = 'mortality' if 'Mortality' in config['title'] else 'readmission'
            df = calculate_performance_gap(df, task_type, base_dir.parent.parent.parent)
            
        grouped_rows = []
        for comparison, group in df.groupby("analysis_key"):
            group = group.sort_values("q_value").head(8) # Top 8 per category
            for _, row in group.iterrows():
                grouped_rows.append((str(comparison), str(row.get("rule", ""))))
        paired_archetypes[key] = set(grouped_rows)
        paired_archetype_order[key] = {pair: i for i, pair in enumerate(grouped_rows)}
        formatted_rows = []
        # Group by comparison to keep it neat
        for comparison, group in df.groupby("analysis_key"):
            group = group.sort_values("q_value").head(8) # Top 8 per category
            for _, row in group.iterrows():
                formatted_rows.append({
                    "Comparison": pretty_analysis_key(comparison, task_hint=config["filename"]),
                    "Archetype Rule": pretty_rule(row.get("rule", "")),
                    "N": int(row["coverage"]),
                    "Coverage (%)": row['coverage_pct'],
                    "Error Ratio": row['lift'],
                    "q-value": row['q_value']
                })
        
        # Use standardized footnotes
        current_footnotes = ARCHETYPE_FOOTNOTES.copy()
        if config["number"] in ["S5", "S9"]:
            current_footnotes[1] = "q-values represent the significance of the difference in failure rates between the two models within the archetype (Exact Binomial test), with Benjamini-Hochberg FDR correction."
            current_footnotes[-1] = "Error Ratio represents the relative failure risk (Error Rate Model A / Error Rate Model B) calculated strictly on the held-out test set."

        res_df = pd.DataFrame(formatted_rows)
        save_manuscript_html(
            res_df, 
            config["title"], 
            config["filename"], 
            TABLES_OUTPUT_DIR,
            footnotes=current_footnotes,
            table_number=config["number"],
            compact=True
        )

    meta_to_archetype = {
        "mortality_meta_IV": "mortality_IV",
        "mortality_meta_IVB": "mortality_IVB",
        "readmission_meta_IV": "readmission_IV",
        "readmission_meta_IVB": "readmission_IVB",
    }

    # Process Meta-Analysis Tables
    for key, config in meta_groups.items():
        path = config["path"]
        if not path.exists():
            print(f"   Missing meta file: {path}")
            continue
        df = pd.read_csv(path)
        
        # Enforce strict archetype alignment with paired archetype table
        ref_key = meta_to_archetype.get(key)
        ref_pairs = paired_archetypes.get(ref_key, set())
        ref_order = paired_archetype_order.get(ref_key, {})
        if ref_pairs and {"analysis", "rule"}.issubset(df.columns):
            df = df[df.apply(lambda r: (str(r["analysis"]), str(r["rule"])) in ref_pairs, axis=1)].copy()
            df["_archetype_order"] = df.apply(
                lambda r: ref_order.get((str(r["analysis"]), str(r["rule"])), 10**9),
                axis=1
            )
        else:
            df["_archetype_order"] = 10**9

        formatted_rows = []
        report_df = df.copy()

        if "q_value_family" in report_df.columns:
            report_df["q_value_family"] = pd.to_numeric(report_df["q_value_family"], errors="coerce")
            report_df = report_df[report_df["q_value_family"] < 0.05].copy()

        # Force row order to follow the paired archetype table order.
        sort_cols = [c for c in ["_archetype_order", "analysis", "rule", "_meta_input_order"] if c in report_df.columns]
        if sort_cols:
            report_df = report_df.sort_values(sort_cols)

        def fmt_meta_number(value):
            return format_3_sig_figs(value)

        for (analysis, rule), group in report_df.groupby(["analysis", "rule"], sort=False):
            meta_lines = []
            subgroup_lines = []
            concordant_lines = []
            effect_lines = []
            q_lines = []
            for _, row in group.sort_values(["q_value_family", "meta_feature"]).iterrows():
                meta_lines.append(beautify_var_name(row["meta_feature"]))
                subgroup_lines.append(fmt_meta_number(row["median_subgroup_error"]))
                concordant_lines.append(fmt_meta_number(row["median_concordant_success"]))
                effect_lines.append(fmt_meta_number(row["effect_size"]))
                q_lines.append(f"<strong>{fmt_meta_number(row['q_value_family'])}</strong>")
            formatted_rows.append({
                "Comparison": pretty_analysis_key(analysis, task_hint=config["filename"]),
                "Archetype Rule": pretty_rule(rule),
                "Significant Meta-Features": "<br>".join(meta_lines),
                "Subgroup Median": "<br>".join(subgroup_lines),
                "Concordant Median": "<br>".join(concordant_lines),
                "Effect (Diff)": "<br>".join(effect_lines),
                "q-value": "<br>".join(q_lines),
            })
        
        if not formatted_rows:
            print(f"   No rows formatted for {key}")
            continue

        res_df = pd.DataFrame(formatted_rows)
        # Determine specific meta-analysis footnotes
        current_meta_footnotes = [
            "Only statistically significant meta-feature differences are displayed (per-family Benjamini-Hochberg FDR q < 0.05).",
            "Meta-features are patient-level summaries derived from the engineered numerical input matrix and used for post hoc archetype characterization.",
            "Late-window measurement concentration is the proportion of counted 24h measurement events that occurred in the final 6 hours before prediction; higher values indicate measurement activity clustered near prediction time.",
            "Total measurement events and number of measured feature families summarize data density; average within-stay variability and average absolute trend magnitude summarize physiological volatility; unmeasured feature-family proportion summarizes missingness/imputation burden.",
            "Medians are reported in each meta-feature's native unit: counts, proportions, or feature-scale magnitudes.",
            "Effect (Diff) represents the absolute difference in medians between the target population and the comparison cohort."
        ]
        
        is_comparative = "Comparative" in config["title"]
        if is_comparative:
            current_meta_footnotes.insert(1, "q-values represent the significance of the difference in meta-features between cases where Model A failed (Model B correct) and cases where Model B failed (Model A correct) within the same clinical archetype (Mann-Whitney U test), with per-family Benjamini-Hochberg FDR correction.")
        else:
            current_meta_footnotes.insert(1, "q-values represent the significance of the difference in meta-features between archetype members and the general study population (Mann-Whitney U test), with per-family Benjamini-Hochberg FDR correction.")

        save_manuscript_html(
            res_df, 
            config["title"], 
            config["filename"], 
            TABLES_OUTPUT_DIR,
            footnotes=current_meta_footnotes,
            table_number=config["number"],
            compact=True
        )

    # Process specifically identified manuscript table
    standalone_csv = base_dir / "manuscript_archetype_table.csv"
    if standalone_csv.exists():
        print(f"   Processing standalone manuscript CSV: {standalone_csv.name}")
        df = pd.read_csv(standalone_csv)
        # Ensure it has Comparison column or similar for grouping
        if "Comparison" not in df.columns and "analysis_key" in df.columns:
            df["Comparison"] = df["analysis_key"].apply(pretty_analysis_key)
        
        save_manuscript_html(
            df,
            "Clinical Archetypes Summary Table",
            "Table_Archetypes_Consolidated",
            TABLES_OUTPUT_DIR,
            footnotes=[
                "Coverage represents the proportion of patients in the total test population (N=5,648) who meet the archetype criteria.",
                "Enrichment represents the ratio of the subgroup error rate to the overall test-population error rate."
            ]
        )

    print(f"\nAll tables saved to: {TABLES_OUTPUT_DIR.absolute()}")
    print("\nACTION: Open the .html files in your browser, select the table (Ctrl+A), and copy-paste into Google Docs.")
    print("TIP: For the best look in Google Docs, set the 'Normal text' font to Times New Roman after pasting.")

if __name__ == "__main__":
    main()
