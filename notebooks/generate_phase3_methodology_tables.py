import sys
import os
import pandas as pd
from pathlib import Path

# Add the root directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Phase_3'))

from Phase_3.config import PROMPTS, FEATURE_LABEL_MAP
from manuscript_table_utils import save_manuscript_html, TABLES_OUTPUT_DIR

def generate_prompting_strategies_table():
    data = []
    for key, prompt in PROMPTS.items():
        if key == 'P0':
            desc = "Control - Null Prompt (No prefix)"
        else:
            desc = prompt
        data.append({"Strategy ID": key, "Prompt Description": desc})
        
    df = pd.DataFrame(data)
    
    # Save as LaTeX
    latex_path = TABLES_OUTPUT_DIR / "Table_Prompting_Strategies.tex"
    df.to_latex(latex_path, index=False, escape=False, column_format='l p{12cm}')
    
    # Save as HTML
    save_manuscript_html(
        df=df,
        title="Prompting Strategies",
        filename="Table_S1_Prompting_Strategies",
        output_dir=TABLES_OUTPUT_DIR,
        table_number="S1"
    )
    print(f"Generated Prompting Strategies Table -> {latex_path} and HTML")

def generate_formatting_strategies_table():
    data = [
        {
            "Formatting Strategy": "F1 (Basic Structured)",
            "Description": "Structured Key-Value representation displaying human-readable feature labels alongside raw physiological measurements, grouped by clinical variable."
        },
        {
            "Formatting Strategy": "F2 (Contextualized Structured)",
            "Description": "Advanced Structured representation appending qualitative evaluation flags (e.g., '(Normal)', '(High > x)', '(Low < x)') to raw values using laboratory reference bounds."
        },
        {
            "Formatting Strategy": "F3 (Abnormal-Only Structured)",
            "Description": "Abstractive clinical representation summarizing key patient demographics and strictly isolating abnormal findings to maximize data density and interpretability."
        }
    ]
    
    df = pd.DataFrame(data)
    
    # Save as LaTeX
    latex_path = TABLES_OUTPUT_DIR / "Table_Formatting_Strategies.tex"
    df.to_latex(latex_path, index=False, escape=False, column_format='l p{12cm}')
    
    # Save as HTML
    save_manuscript_html(
        df=df,
        title="Formatting Strategies",
        filename="Table_S2_Formatting_Strategies",
        output_dir=TABLES_OUTPUT_DIR,
        table_number="S2"
    )
    print(f"Generated Formatting Strategies Table -> {latex_path} and HTML")

def generate_feature_engineering_table():
    """
    Generates a categorized feature engineering table showing:
    - The 6 engineering categories from the preprocessing pipeline
    - The variables in each category and the statistical features computed
    - Meta-features used in Phase V analysis, grouped by family
    """

    # Part 1: Engineering category table
    # Based on the 6 categories defined in data_preprocessing_LOS.py and the
    # original feature_classification.csv (104 raw MIMIC-III vitals_labs_mean columns).
    #
    # Each category uses a different engineering function producing different feature sets.
    category_rows = [
        {
            "Engineering Category": "High-Frequency Physiological",
            "Representative Variables": (
                "Heart rate, Systolic BP, Diastolic BP, Mean arterial pressure, "
                "Respiratory rate, SpO₂ (pulse ox), Temperature"
            ),
            "Engineered Statistics": (
                "Last value, Mean (24h), Min (24h), Max (24h), Std Dev (24h), "
                "Measurement count (24h), Measurement count (6h), "
                "Slope (24h), Slope (6h)"
            ),
            "N Features per Variable": "9"
        },
        {
            "Engineering Category": "Labile Lab",
            "Representative Variables": (
                "Sodium, Potassium, Chloride, Bicarbonate, BUN, Creatinine, "
                "Glucose, Hemoglobin, Hematocrit, WBC, Platelets, "
                "Bilirubin (total), ALT, AST, Alkaline phosphatase, "
                "Lactate, Albumin, Phosphate, Magnesium, Calcium (total), "
                "Calcium (ionized), pH, PaO₂, PaCO₂, Base excess, "
                "Anion gap, INR, PT, PTT"
            ),
            "Engineered Statistics": (
                "Last value, Mean (24h), Std Dev (24h), "
                "Measurement count (24h), Slope (24h)"
            ),
            "N Features per Variable": "5"
        },
        {
            "Engineering Category": "Stable Index",
            "Representative Variables": (
                "GCS (total, verbal, motor, eye), Height, Weight, BMI, "
                "FiO₂, PEEP, Tidal volume, Urine output"
            ),
            "Engineered Statistics": "Last value, Measurement count (24h)",
            "N Features per Variable": "2"
        },
        {
            "Engineering Category": "Sparse Dynamic",
            "Representative Variables": (
                "Troponin I/T, BNP/proBNP, CK, CK-MB, Lipase, Amylase, "
                "Cortisol, TSH, Free T4, LDH, Fibrinogen, D-dimer, "
                "Procalcitonin, ESR, CRP"
            ),
            "Engineered Statistics": "Last value, Mean (24h), Measurement count (24h), Slope (24h)",
            "N Features per Variable": "4"
        },
        {
            "Engineering Category": "Static",
            "Representative Variables": "Height, Weight",
            "Engineered Statistics": "First recorded value, Measurement count",
            "N Features per Variable": "2"
        },
        {
            "Engineering Category": "Event-Driven",
            "Representative Variables": (
                "Vasopressor administration (binary), "
                "Mechanical ventilation status (binary)"
            ),
            "Engineered Statistics": "Last value, Measurement count (24h)",
            "N Features per Variable": "2"
        },
    ]

    # Footnote explaining the total feature count
    footnotes = [
        "Total features after engineering: 458. This includes 4 demographic features (age, gender, ethnicity, insurance) "
        "and 454 engineered clinical features derived from 104 primary variables. "
        "Preprocessing log confirmed (14825 × 458) training feature matrix."
    ]

    # Part 2: Phase V meta-features (from metafeatures.txt), grouped by family
    meta_rows = [
        {
            "Family": "Density",
            "Meta-feature": "Total measurement events",
            "Derivation": "Total observed measurement burden per patient. For each clinical feature family, the maximum available count across 24h and 6h count features is used, then summed across families to avoid double-counting the same measurement stream."
        },
        {
            "Family": "Density",
            "Meta-feature": "Number of measured feature families",
            "Derivation": "Number of distinct clinical feature families with at least one recorded measurement after combining 24h and 6h count features."
        },
        {
            "Family": "Temporality",
            "Meta-feature": "Late-window measurement concentration",
            "Derivation": "Proportion of measurement events occurring in the final 6 hours of the 24h observation window: sum of 6h measurement counts divided by sum of 24h measurement counts. Higher values indicate measurements were concentrated closer to prediction time."
        },
        {
            "Family": "Volatility",
            "Meta-feature": "Average within-stay variability",
            "Derivation": "Mean of per-feature-family standard deviation summaries across available clinical variables. Higher values indicate more variable physiology during the observation window."
        },
        {
            "Family": "Volatility",
            "Meta-feature": "Average absolute trend magnitude",
            "Derivation": "Mean absolute value of per-feature-family temporal slopes, using 24h slopes when available and 6h slopes as fallback. Absolute values measure trend strength regardless of direction."
        },
        {
            "Family": "Imputation",
            "Meta-feature": "Unmeasured feature-family proportion",
            "Derivation": "Proportion of canonical clinical feature families with no recorded measurement. Higher values indicate greater missingness and greater reliance on imputed values."
        },
    ]

    # --- Build and save Part 1: Variable Category Table ---
    df1 = pd.DataFrame(category_rows)
    latex_path1 = TABLES_OUTPUT_DIR / "Table_Feature_Engineering_Variables.tex"
    df1.to_latex(latex_path1, index=False, escape=False,
                 column_format='l p{4.5cm} p{5.5cm} c')
    save_manuscript_html(
        df=df1,
        title="Feature Engineering (Clinical Variables)",
        filename="Table_S3a_Feature_Engineering_Variables",
        output_dir=TABLES_OUTPUT_DIR,
        table_number="S3a",
        footnotes=footnotes
    )

    # --- Build and save Part 2: Meta-feature Table ---
    df2 = pd.DataFrame(meta_rows)
    latex_path2 = TABLES_OUTPUT_DIR / "Table_Feature_Engineering_MetaFeatures.tex"
    df2.to_latex(latex_path2, index=False, escape=False,
                 column_format='l l p{9cm}')
    save_manuscript_html(
        df=df2,
        title="Feature Engineering (Patient-Level Meta-Features)",
        filename="Table_S3b_Feature_Engineering_MetaFeatures",
        output_dir=TABLES_OUTPUT_DIR,
        table_number="S3b",
        footnotes=[
            "Meta-features are calculated per patient from the engineered numerical input matrix and are used only for post hoc archetype characterization.",
            "The late-window measurement concentration is a proportion: 0 means no recorded measurements in the final 6 hours, while 1 means all counted 24h measurement events occurred in the final 6 hours."
        ]
    )

    print(f"Generated Feature Engineering Tables -> {latex_path1} and {latex_path2} (HTML)")


if __name__ == '__main__':
    generate_prompting_strategies_table()
    generate_formatting_strategies_table()
    generate_feature_engineering_table()
    print("Successfully generated all methodology tables!")
