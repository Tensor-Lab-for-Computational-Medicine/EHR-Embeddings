"""
Supplementary Figures Script - Combined Edition
Generates two publication-quality figures, each with panels A–C (mortality vs readmission).
"""

import os
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import re
import glob
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

# ==============================================================================
# 0. Paths & constants
# ==============================================================================
NUMERICAL_DATA_PATH = 'notebooks/Phase 1 and 2/phase_1_outputs/X_test.pkl'
Y_TEST_PATH         = 'notebooks/Phase 1 and 2/phase_1_outputs/y_test.pkl'
OUTPUT_DIR          = 'notebooks/Phase 4/figures/supplementary/combined'

TASKS = {
    'mort_hosp': {
        'nm_model_path': 'notebooks/Phase 1 and 2/phase_1_outputs/mort_hosp/model_1_xgboost_baseline_calibrated.pkl',
        'sm_model_path': 'notebooks/Phase 5/embedding_model_results/text-embedding-004/mort_hosp/model_F3_P5.pkl',
        'embedding_dir': 'notebooks/Phase 4/embeddings_models_text-embedding-004/F3_P5/test',
        'title': 'In-Hospital Mortality (text-embedding-004)'
    },
    'readmission_30': {
        'nm_model_path': 'notebooks/Phase 1 and 2/phase_1_outputs/readmission_30/model_1_xgboost_baseline_calibrated.pkl',
        'sm_model_path': 'notebooks/Phase 5/embedding_model_results/text-embedding-005/readmission_30/model_F1_P0.pkl',
        'embedding_dir': 'notebooks/Phase 4/embeddings_text-embedding-005/F1_P0/test',
        'title': '30-Day Readmission (text-embedding-005)'
    }
}

# Colour palette
C_SM      = '#d62728'
C_NM      = '#1f77b4'

FONT = 'DejaVu Sans'
plt.rcParams.update({'font.family': FONT, 'axes.spines.top': False,
                     'axes.spines.right': False})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Helper functions
# ==============================================================================
def clean_label(name):
    suffixes = {
        '_mean_count_6h': ' (6h Cnt)',
        '_mean_count': ' (Cnt)',
        '_mean_last': ' (Last)',
        '_mean_mean': ' (Mean)',
        '_mean_min_24h': ' (Min)',
        '_mean_max_24h': ' (Max)',
        '_mean_slope_24h': ' (Slope)',
        '_mean_slope_6h': ' (6h Slope)',
        '_mean_stddev_24h': ' (Var)',
        '_mean_value': ' (Val)',
        '_encoded': ''
    }
    s = name
    mod = ''
    for suff, m in suffixes.items():
        if s.endswith(suff):
            s = s[:-len(suff)]
            mod = m
            break
            
    s = s.replace('_', ' ').strip().title()
    
    abbrevs = {
        'Central Venous Pressure': 'CVP',
        'Pulmonary Artery Pressure Systolic': 'PAP Systolic',
        'Pulmonary Artery Pressure Mean': 'PAP Mean',
        'Pulmonary Artery Pressure': 'PAP',
        'Pulmonary Capillary Wedge Pressure': 'PCWP',
        'Partial Pressure Of Carbon Dioxide': 'pCO2',
        'Partial Pressure Of Oxygen': 'pO2',
        'Fraction Inspired Oxygen': 'FiO2',
        'Positive End-Expiratory Pressure Set': 'PEEP Set',
        'Positive End-Expiratory Pressure': 'PEEP',
        'Peak Inspiratory Pressure': 'PIP',
        'Glascow Coma Scale Total': 'GCS',
        'Alanine Aminotransferase': 'ALT',
        'Asparate Aminotransferase': 'AST',
        'Blood Urea Nitrogen': 'BUN',
        'White Blood Cell Count': 'WBC',
        'Prothrombin Time Inr': 'PT INR',
        'Partial Thromboplastin Time': 'PTT',
        'Mean Corpuscular Hemoglobin Concentration': 'MCHC',
        'Mean Corpuscular Hemoglobin': 'MCH',
        'Mean Corpuscular Volume': 'MCV',
        'Red Blood Cell Count': 'RBC',
        'Respiratory Rate Set': 'RR Set',
        'Respiratory Rate': 'RR',
        'Diastolic Blood Pressure': 'DBP',
        'Systolic Blood Pressure': 'SBP',
        'Mean Arterial Pressure': 'MAP',
        'Systemic Vascular Resistance': 'SVR',
        'Pulmonary Artery Pressure': 'PAP',
        'Partial Pressure Of Oxygen': 'pO2',
        'Tidal Volume Observed': 'TV Observed',
        'Tidal Volume Set': 'TV Set',
        'Tidal Volume Spontaneous': 'TV Spontaneous',
        'Tidal Volume': 'TV',
        'Lactic Acid': 'Lactate',
        'Alkaline Phosphate': 'ALP',
        'Lactate Dehydrogenase': 'LDH',
        'Potassium Serum': 'Potassium',
        'Has Any Aki': 'Any AKI',
        'Has Severe Aki': 'Severe AKI',
        'Has Hypocalcemia Ionized': 'Hypocalcemia',
        'Has Hyperlactatemia': 'Hyperlactatemia',
        'Is On Ventilator': 'Ventilated',
        'Sirs Wbc Criterion': 'SIRS WBC',
        'Has Neutrophilia': 'Neutrophilia',
        'Liver Dysfunction Type': 'Liver Dysfunction',
        'Shock Index': 'Shock Index',
        'Has Anticoagulation Derangement': 'Anticoagulation Derange',
        'Has Myocardial Injury': 'Myocardial Injury',
        'Anemia Severity': 'Anemia Severity',
        'Is Malnourished Proxy': 'Malnourished',
        'Data Density Score': 'Data Density',
        'Has Invasive Hemo Monitoring': 'Invasive Hemo',
        'Gcs Level': 'GCS Level',
        'Cardiac Output Thermodilution': 'CO (Thermo)',
        'Temperature': 'Temp',
        'Central Venous Pressure': 'CVP',
        'Systemic Vascular Resistance': 'SVR'
    }
    for full, abbr in abbrevs.items():
        if full in s:
            s = s.replace(full, abbr)
            
    return s + mod

CLINICAL_GROUPS = {
    "Acid-Base":       (["ph", "co2", "carbon dioxide", "bicarbonate", "etco2", "pco2", "lactate", "lactic", "base excess"], "#4e79a7"),
    "Resp Mech":  (["peep", "positive end-expiratory pressure", "peak inspiratory pressure", "tidal volume", "fio2", "fraction inspired oxygen", "ventilator", "ventilation", "spontaneous", "respiratory rate", "rr", "oxygen", "spo2", "po2"], "#59a14f"),
    "Heme Coag": (["prothrombin time", "partial thromboplastin", "platelets", "hemoglobin", "hematocrit", "pt inr", "ptt", "rbc", "red blood cell", "mcv", "mch", "anticoagulation", "anemia", "coag"], "#8c564b"),
    "Metab Renal": (["glascow", "gcs", "glucose", "creatinine", "blood urea nitrogen", "bun", "potassium", "sodium", "calcium", "magnesium", "anion gap", "weight", "aki", "malnourished", "renal", "kidney", "phosphate", "phosphorus", "albumin", "protein", "chloride"], "#b07aa1"),
    "Hemo Vasc": (["heart rate", "hr", "blood pressure", "bp", "sbp", "dbp", "map", "central venous", "cvp", "cardiac", "vascular", "troponin", "pap", "pcwp", "shock", "myocardial", "hemo", "cv", "svr", "pulmonary"], "#e377c2"),
    "Hepatic":  (["bilirubin", "asparate aminotransferase", "alanine aminotransferase", "alkaline phosphate", "lactate dehydrogenase", "fibrinogen", "alt", "ast", "alp", "ldh", "liver"], "#e15759"),
    "Immune":  (["neutrophil", "neutrophils", "lymphocyte", "lymphocytes", "monocyte", "monocytes", "basophil", "basophils", "wbc", "white blood cell", "sirs", "temperature", "temp"], "#f28e2b"),
    "Demo Misc": (["age", "gender", "race", "ethnicity", "density"], "#7f7f7f"),
}

def assign_group(col_name):
    name_lower = col_name.lower()
    for group, (kws, _) in CLINICAL_GROUPS.items():
        for kw in kws:
            # Word boundary regex: allow spaces, underscores, parentheses, or start/end of string
            pattern = rf"(^|[\s_\(\)]){re.escape(kw.lower())}([\s_\(\)]|$)"
            if re.search(pattern, name_lower):
                return group
    return "Other"

def get_baseline_icu_vars():
    """Return a foundational set of ICU physiological markers."""
    return {
        'heart rate_mean_last', 'heart rate_mean_count',
        'blood pressure mean_mean_last', 'blood pressure mean_mean_count',
        'systolic blood pressure_mean_last', 'systolic blood pressure_mean_count',
        'oxygen saturation_mean_last', 'oxygen saturation_mean_count',
        'temperature_mean_last', 'temperature_mean_count',
        'respiratory rate_mean_last', 'respiratory rate_mean_count',
        'glascow coma scale total_mean_last', 'glascow coma scale total_mean_count',
        'glucose_mean_last', 'glucose_mean_count',
        'platelets_mean_last', 'platelets_mean_count',
        'white blood cell count_mean_last', 'white blood cell count_mean_count',
        'blood urea nitrogen_mean_last', 'blood urea nitrogen_mean_count',
        'creatinine_mean_last', 'creatinine_mean_count',
        'sodium_mean_last', 'sodium_mean_count',
        'potassium_mean_last', 'potassium_mean_count',
        'bilirubin_mean_last', 'bilirubin_mean_count',
        'albumin_mean_last', 'albumin_mean_count',
        'lactate_mean_last', 'lactate_mean_count',
        'calcium ionized_mean_last', 'calcium ionized_mean_count',
        'age', 'gender_encoded'
    }

def extract_archetype_vars(task):
    """Dynamically scan archetype files for clinical variables used in error rules."""
    archetype_dir = 'notebooks/Phase 6 - H2 Analysis/h2b/h2_results'
    if task == 'readmission_30':
        archetype_dir += '_readmission_30'
    
    files = glob.glob(os.path.join(archetype_dir, 'final_archetypes*.csv'))
    found_vars = set()
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'rule_str' in df.columns:
                rules = " ".join(df['rule_str'].astype(str).tolist())
                # Extract words that are followed by operators or brackets (variable names)
                vars_in_rules = re.findall(r'([\w\s]+?)(?:[=<>:!\[])', rules)
                for v in vars_in_rules:
                    v_clean = v.strip()
                    if v_clean: found_vars.add(v_clean)
        except Exception:
            continue
    return found_vars

def get_group_color(col_name):
    group = assign_group(col_name)
    if group in CLINICAL_GROUPS:
        return CLINICAL_GROUPS[group][1]
    return "#aaaaaa"

# ==============================================================================
# 2. Linear Probes (Ridge Regression)
# ==============================================================================
print("Loading numerical data...")
X_test_full = pd.read_pickle(NUMERICAL_DATA_PATH)
X_test_full = X_test_full.loc[:, X_test_full.std() > 0]
y_test_full = pd.read_pickle(Y_TEST_PATH)

ORIGINAL_COLS = X_test_full.columns.tolist()

# Compute Curated Variables
if 'calcium ionized_mean_count' in X_test_full.columns and 'calcium ionized_mean_last' in X_test_full.columns:
    X_test_full['has_hypocalcemia_ionized'] = ((X_test_full['calcium ionized_mean_count'] > 0) & 
                                               (X_test_full['calcium ionized_mean_last'] < 4.6))
if 'lactate_mean_count' in X_test_full.columns:
    X_test_full['has_hyperlactatemia'] = ((X_test_full['lactate_mean_count'] > 0) & 
                                          (X_test_full['lactate_mean_last'] > 2.0))

def is_count_or_value(col_name):
    excluded_suffixes = (
        '_mean_mean',
        '_mean_min_24h',
        '_mean_max_24h',
        '_mean_slope_24h',
        '_mean_slope_6h',
        '_mean_stddev_24h'
    )
    return not col_name.endswith(excluded_suffixes)

def count_last_pair_partner(col_name, available):
    """Return paired _mean_count <-> _mean_last if present in available (index/set of column names)."""
    a = available if isinstance(available, set) else set(available)
    if col_name.endswith('_mean_last'):
        p = col_name[: -len('_mean_last')] + '_mean_count'
    elif col_name.endswith('_mean_count') and not col_name.endswith('_mean_count_6h'):
        p = col_name[: -len('_mean_count')] + '_mean_last'
    else:
        return None
    return p if p in a else None

def ensure_count_last_pairs(vars_list, available):
    """Extend vars so every _mean_last/_mean_count in the list has its sibling when it exists."""
    avail, out = (available if isinstance(available, set) else set(available)), set(vars_list)
    for v in list(out):
        p = count_last_pair_partner(v, avail)
        if p:
            out.add(p)
    return list(out)

def process_task_data(task_name, config):
    print(f"\nProcessing data for: {task_name}")
    embedding_vectors, valid_indices = [], []
    for icustay_id in X_test_full.index:
        fp = os.path.join(config['embedding_dir'], f"{icustay_id}.npy")
        if os.path.exists(fp):
            embedding_vectors.append(np.load(fp))
            valid_indices.append(icustay_id)
            
    if len(valid_indices) == 0:
        return None
        
    E_test = np.vstack(embedding_vectors)
    X_test = X_test_full.loc[valid_indices].copy()
    
    with open(config['nm_model_path'], 'rb') as f:
        nm_model = pickle.load(f)
    with open(config['sm_model_path'], 'rb') as f:
        sm_model = pickle.load(f)
        
    # Linear Probes
    X_vals = X_test.values.astype(float)
    X_norm = (X_vals - X_vals.mean(0)) / (X_vals.std(0) + 1e-9)
    E_norm = (E_test - E_test.mean(0)) / (E_test.std(0) + 1e-9)
    corr_matrix = X_norm.T @ E_norm / X_norm.shape[0]
    corr_df = pd.DataFrame(corr_matrix, index=X_test.columns, columns=[f"Dim_{i}" for i in range(E_test.shape[1])])
    
    print("Fitting linear probes (5-fold CV)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(X_vals)
    for train_idx, test_idx in kf.split(E_test):
        model = Ridge(alpha=1.0)
        model.fit(E_test[train_idx], X_vals[train_idx])
        preds[test_idx] = model.predict(E_test[test_idx])
        
    r2_scores = pd.Series([max(0, r2_score(X_vals[:, i], preds[:, i])) for i in range(X_vals.shape[1])], index=X_test.columns)
    
    # SHAP Importance
    N_SHAP_SAMPLES = min(500, len(X_test))
    shap_idx = np.random.RandomState(42).choice(len(X_test), N_SHAP_SAMPLES, replace=False)
    
    print("Calculating SHAP...")
    if 'xgboost' in config['nm_model_path']:
        if hasattr(nm_model, 'calibrated_classifiers_'):
            calibrated_clf = nm_model.calibrated_classifiers_[0]
            if hasattr(calibrated_clf, 'estimator'):
                base_nm_model = calibrated_clf.estimator
            else:
                base_nm_model = calibrated_clf.base_estimator
        else:
            base_nm_model = nm_model
        
        # Use raw model output (Log-Odds) for better scale comparability on imbalanced tasks
        nm_explainer = shap.TreeExplainer(base_nm_model, model_output='raw')
        
        # Ensure we only pass features the model was trained on
        X_nm_shap = X_test.iloc[shap_idx]
        if hasattr(base_nm_model, 'feature_names'):
            X_nm_shap = X_nm_shap[base_nm_model.feature_names]
        else:
            # Fallback to ORIGINAL_COLS if specific feature names not available
            X_nm_shap = X_nm_shap[[c for c in ORIGINAL_COLS if c in X_nm_shap.columns]]
            
        nm_shap_vals = nm_explainer.shap_values(X_nm_shap)
    else:
        X_bg = X_test.iloc[shap_idx]
        # Filter for original features for consistency
        X_bg = X_bg[[c for c in ORIGINAL_COLS if c in X_bg.columns]]
        nm_explainer = shap.LinearExplainer(nm_model, X_test[[c for c in ORIGINAL_COLS if c in X_test.columns]])
        nm_shap_vals = nm_explainer.shap_values(X_bg)
    if isinstance(nm_shap_vals, list): nm_shap_vals = nm_shap_vals[1]

    # Align semantic model to the same raw logit scale
    sm_explainer = shap.TreeExplainer(sm_model, model_output='raw')
    sm_shap_vals = sm_explainer.shap_values(E_test[shap_idx])
    if isinstance(sm_shap_vals, list): sm_shap_vals = sm_shap_vals[1]

    nm_mean_abs = np.abs(nm_shap_vals).mean(axis=0)
    unique_nm_labels = set()
    dedup_nm_idx = []
    
    # Map back to original indices using the columns passed to SHAP
    nm_cols = X_nm_shap.columns if 'xgboost' in config['nm_model_path'] else [c for c in ORIGINAL_COLS if c in X_test.columns]
    
    for idx in np.argsort(nm_mean_abs)[::-1]:
        col_name = nm_cols[idx]
        if not is_count_or_value(col_name):
            continue
            
        lbl = clean_label(col_name)
        if lbl not in unique_nm_labels:
            unique_nm_labels.add(lbl)
            # Find the index in X_test
            dedup_nm_idx.append(X_test.columns.get_loc(col_name))
        if len(dedup_nm_idx) == 15: break
            
    nm_top15_vals  = nm_mean_abs[dedup_nm_idx]
    nm_top15_labels = [clean_label(X_test.columns[i]) for i in dedup_nm_idx]

    sm_mean_abs = np.abs(sm_shap_vals).mean(axis=0)
    sm_top15_dim_idx = np.argsort(sm_mean_abs)[::-1][:15]
    sm_top15_vals    = sm_mean_abs[sm_top15_dim_idx]

    # DYNAMIC VARIABLE DISCOVERY FOR FIGURE 1A
    # ---------------------------------------
    dynamic_selection = set()
    dynamic_selection.update(get_baseline_icu_vars())
    dynamic_selection.update([X_test.columns[i] for i in dedup_nm_idx])
    
    sm_top15_labels = []
    for d in sm_top15_dim_idx:
        dim_corrs = corr_df.iloc[:, d].abs().sort_values(ascending=False)
        
        # 3. Discovery from Semantic Model Dimensions
        count = 0
        top_k = {}
        seen_base = set()
        for var, val in dim_corrs.items():
            if not is_count_or_value(var): continue
            
            # Add to Panel A selection
            if count < 3:
                dynamic_selection.add(var)
                count += 1
            
            # Format label for Panel C/F
            lbl = clean_label(var)
            base_var = lbl.strip()
            if base_var not in seen_base and len(top_k) < 3:
                seen_base.add(base_var)
                top_k[base_var] = val**2
            
            if count >= 3 and len(top_k) >= 3: break
            
        label_parts = [f"{(s[:19]+'...') if len(s)>22 else s} R²={r2:.2f}" for s, r2 in top_k.items()]
        sm_top15_labels.append(f"Dim {d}\n({', '.join(label_parts)})")
            
    # 4. Archetype-specific variables (Dynamically from rules)
    dynamic_selection.update(extract_archetype_vars(task_name))
    
    # Filter for availability and variance
    final_vars = [v for v in dynamic_selection if v in r2_scores.index]
    final_vars = ensure_count_last_pairs(final_vars, r2_scores.index)
    
    # DEDUPLICATE LABELS: Consolidate synonyms (e.g., Lactate and Lactic Acid)
    # Group by clean label and keep the one with the highest R2 score
    label_to_best_var = {}
    for v in final_vars:
        lbl = clean_label(v)
        score = r2_scores[v]
        if lbl not in label_to_best_var or score > label_to_best_var[lbl][1]:
            label_to_best_var[lbl] = (v, score)
    
    deduped_vars = [v for v, score in sorted(label_to_best_var.values(), key=lambda x: x[1])]
    
    panel_a_df = pd.DataFrame({
        'variable':  deduped_vars,
        'r2_score': r2_scores[deduped_vars].values,
        'group':     [assign_group(v) for v in deduped_vars],
        'color':     [get_group_color(v) for v in deduped_vars],
        'label':     [clean_label(v) for v in deduped_vars]
    })

    return {
        'panel_a_df': panel_a_df,
        'nm_vals': nm_top15_vals,
        'nm_labels': nm_top15_labels,
        'sm_vals': sm_top15_vals,
        'sm_labels': sm_top15_labels
    }

print("\nStarting interpretability figure generation...")
results = {}
for t, cfg in TASKS.items():
    results[t] = process_task_data(t, cfg)


def render_interpretability_row(fig, gs, task, letters):
    """One task row: linear probes (left), numerical SHAP (middle), semantic SHAP (right)."""
    res = results[task]
    task_label = "Mortality" if task == 'mort_hosp' else "Readmission"

    # ---------------------------------------------------------
    # Probes (Left Column)
    # ---------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    df = res['panel_a_df']
    bar_colors = df['color'].tolist()
    _n = len(df)
    _probe_y_pitch = 1.18  # vertical spacing between probe rows (panel A)
    _yp = np.arange(_n, dtype=float) * _probe_y_pitch
    _half = _probe_y_pitch / 2

    ax_a.barh(y=_yp, width=df['r2_score'], color=bar_colors, edgecolor='white', linewidth=0.5, height=0.56, zorder=3)
    for i in range(_n):
        ax_a.axhspan(_yp[i] - _half, _yp[i] + _half, color='#f7f7f7' if i % 2 == 0 else 'white', zorder=0)

    for i, (_, r) in enumerate(df.iterrows()):
        ax_a.text(r['r2_score'] + 0.003, _yp[i], f"{r['r2_score']:.3f}", va='center', ha='left', fontsize=21, color='#444444')

    ax_a.set_yticks(_yp)
    ax_a.set_yticklabels(df['label'], fontsize=23)
    ax_a.set_ylim(_yp[0] - _half, _yp[-1] + _half)
    ax_a.set_xlabel("Linear Probe R² (5-Fold CV Ridge Regression)", fontsize=30, labelpad=12, color='#444')
    ax_a.set_xlim(0, max(0.9, df['r2_score'].max() * 1.15))
    ax_a.xaxis.grid(True, linestyle='--', linewidth=0.6, color='#dddddd', zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.tick_params(axis='x', labelsize=24)
    ax_a.tick_params(axis='y', pad=6)

    for spine in ['top', 'right', 'left']: ax_a.spines[spine].set_visible(False)
    ax_a.spines['bottom'].set_color('#cccccc')
    
    # Short header to avoid overlap
    ax_a.text(-0.02, 1.01, f"{letters[0]}. {task_label}: Probes", transform=ax_a.transAxes, fontsize=33, fontweight='bold', color='#1a1a2e', va='bottom')

    # Unified legend for panel A (probes)
    group_handles = [mpatches.Patch(facecolor=color, label=group, edgecolor='white') for group, (_, color) in CLINICAL_GROUPS.items() if any(assign_group(v) == group for v in df['variable'])]
    ax_a.legend(handles=group_handles, loc='lower right', bbox_to_anchor=(1.0, 0.05), fontsize=15, framealpha=0.95, edgecolor='#cccccc', title='Clinical Group', title_fontsize=17, ncol=1)
    
    # ---------------------------------------------------------
    # Aggregate Suffix Key (Lower Left Area)
    # ---------------------------------------------------------
    ax_a.text(0.2, 0.01, "(Cnt)= # of Measurements, (Last)=Final Observation", 
              transform=ax_a.transAxes, fontsize=16, ha='left', va='bottom', 
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdfdfd', alpha=0.95, edgecolor='#dddddd'))

    # ---------------------------------------------------------
    # SHAP (Middle & Right Columns)
    # ---------------------------------------------------------
    ax_nm = fig.add_subplot(gs[0, 1])
    ax_sm = fig.add_subplot(gs[0, 2])
    
    max_shap = max(res['nm_vals'].max(), res['sm_vals'].max()) * 1.15
    
    for ax, vals, labels, color, m_name, sub, is_right, panel_letter in zip(
        [ax_nm, ax_sm],
        [res['nm_vals'][::-1], res['sm_vals'][::-1]],
        [res['nm_labels'][::-1], res['sm_labels'][::-1]],
        [C_NM, C_SM],
        ["Numerical Model", "Semantic Model"],
        ["Variables", "Dimensions"],
        [False, True],
        [letters[1], letters[2]]
    ):
        ax.barh(range(len(vals)), vals, color=color, alpha=0.82, edgecolor='white', linewidth=0.5, height=0.6)
        for i in range(len(vals)):
            ax.axhspan(i - 0.5, i + 0.5, color='#f7f7f7' if i % 2 == 0 else 'white', zorder=0)

        if is_right: ax.yaxis.tick_right()
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=32)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=30, labelpad=12)
        
        # Shorter header
        ax.text(-0.02, 1.01, f"{panel_letter}. {m_name}", transform=ax.transAxes, fontsize=33, fontweight='bold', color='#1a1a2e', va='bottom')
            
        ax.xaxis.grid(True, linestyle='--', linewidth=0.6, color='#e0e0e0', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=24)
        if is_right:
            ax.tick_params(axis='y', pad=45) # Avoid overlap with SHAP values
        ax.set_xlim(0, max_shap)

        ax.set_ylim(-1, len(vals))

        for i, v in enumerate(vals):
            ax.text(v + max_shap * 0.02, i, f"{v:.4f}", va='center', ha='left', fontsize=23, color='#444')


def save_interpretability_figure(task, letters, basename):
    """Single-row figure (3 columns); writes PNG (300 dpi) and PDF."""
    fig = plt.figure(figsize=(30, 36), facecolor='white')
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.4, 1, 1], wspace=0.48, hspace=0.10)
    render_interpretability_row(fig, gs, task, letters)
    plt.subplots_adjust(bottom=0.05, top=0.95)
    base = os.path.join(OUTPUT_DIR, basename)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{base}.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)


task_keys = ['mort_hosp', 'readmission_30']
# Each figure uses panels A–C; task distinguishes mortality vs readmission
save_interpretability_figure('mort_hosp', ['A', 'B', 'C'], 'figure_1_interpretability_panels_abc_mortality')
save_interpretability_figure('readmission_30', ['A', 'B', 'C'], 'figure_1_interpretability_panels_abc_readmission')

print("Exporting data to CSV...")
# Export Linear Probes
all_probes = []
for task in task_keys:
    df_p = results[task]['panel_a_df'].copy()
    df_p['task'] = task
    all_probes.append(df_p)
pd.concat(all_probes).to_csv(os.path.join(OUTPUT_DIR, 'linear_probes_data.csv'), index=False)

# Export SHAP data
all_shap = []
for task in task_keys:
    # NM
    df_nm = pd.DataFrame({
        'task': task,
        'model': 'Numerical Model',
        'feature_or_dimension': results[task]['nm_labels'],
        'mean_abs_shap': results[task]['nm_vals'],
        'rank': range(1, len(results[task]['nm_labels']) + 1)
    })
    # SM
    df_sm = pd.DataFrame({
        'task': task,
        'model': 'Semantic Model',
        'feature_or_dimension': results[task]['sm_labels'],
        'mean_abs_shap': results[task]['sm_vals'],
        'rank': range(1, len(results[task]['sm_labels']) + 1)
    })
    all_shap.extend([df_nm, df_sm])
pd.concat(all_shap).to_csv(os.path.join(OUTPUT_DIR, 'shap_importance_data.csv'), index=False)

print(f"Done! Two interpretability figures (each panels A–C) and CSV data saved to {OUTPUT_DIR}")
