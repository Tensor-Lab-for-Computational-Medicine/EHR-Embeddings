"""
Supplementary Figures Script - Combined Edition
Generates a single publication-quality figure combining both tasks.
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
        'nm_model_path': 'notebooks/Phase 1 and 2/phase_1_outputs/mort_hosp/model_1_xgboost_baseline.pkl',
        'sm_model_path': 'notebooks/Phase 5/embedding_model_results/text-embedding-004/mort_hosp/model_F3_P5.pkl',
        'embedding_dir': 'notebooks/Phase 4/embeddings_models_text-embedding-004/F3_P5/test',
        'title': 'In-Hospital Mortality (text-embedding-004)'
    },
    'readmission_30': {
        'nm_model_path': 'notebooks/Phase 1 and 2/phase_1_outputs/readmission_30/model_2_elastic_net_baseline.pkl',
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
        '_mean_count_6h': ' (6h Count)',
        '_mean_count': ' (Count)',
        '_mean_last': ' (Value)',
        '_mean_mean': ' (Mean)',
        '_mean_min_24h': ' (Min)',
        '_mean_max_24h': ' (Max)',
        '_mean_slope_24h': ' (Slope)',
        '_mean_slope_6h': ' (6h Slope)',
        '_mean_stddev_24h': ' (Var)',
        '_mean_value': ' (Value)',
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
        'Tidal Volume Observed': 'TV Observed',
        'Tidal Volume Set': 'TV Set',
        'Tidal Volume Spontaneous': 'TV Spontaneous',
        'Tidal Volume': 'TV',
        'Alkaline Phosphate': 'ALP',
        'Lactate Dehydrogenase': 'LDH',
        'Potassium Serum': 'Potassium'
    }
    for full, abbr in abbrevs.items():
        if full in s:
            s = s.replace(full, abbr)
            
    return s + mod

CLINICAL_GROUPS = {
    "Acid-Base / CO\u2082":       (["ph_", "co2", "carbon dioxide", "bicarbonate", "etco2", "pco2"], "#4e79a7"),
    "Respiratory Mechanics":  (["peep", "positive end-expiratory pressure", "peak inspiratory pressure", "tidal volume", "fio2", "fraction inspired oxygen"], "#59a14f"),
    "Hepatic / Cholestatic":  (["bilirubin", "asparate aminotransferase", "alanine aminotransferase", "alkaline phosphate", "lactate dehydrogenase", "lactate", "fibrinogen", "alt", "ast", "alp", "ldh"], "#e15759"),
    "Immune / Inflammatory":  (["neutrophils", "lymphocytes", "monocytes", "basophils", "wbc", "white blood cells"], "#f28e2b"),
    "Metabolic / Neurologic": (["glascow", "gcs", "glucose", "creatinine", "blood urea nitrogen", "bun", "potassium", "sodium", "calcium", "magnesium", "anion gap", "weight", "respiratory rate", "rr"], "#b07aa1"),
    "Coagulation / Hematologic": (["prothrombin time", "partial thromboplastin", "platelets", "hemoglobin", "hematocrit", "pt inr", "ptt", "rbc", "mcv", "mch"], "#8c564b"),
    "Hemodynamic / Vascular": (["heart rate", "blood pressure", "central venous", "cvp", "cardiac", "vascular", "troponin", "pap", "pcwp"], "#e377c2"),
    "Demographics":           (["age", "gender"], "#7f7f7f"),
}

def assign_group(col_name):
    col_lower = col_name.lower()
    for group, (kws, _) in CLINICAL_GROUPS.items():
        if any(kw in col_lower for kw in kws): return group
    return "Other"

def get_group_color(col_name):
    col_lower = col_name.lower()
    for group, (kws, color) in CLINICAL_GROUPS.items():
        if any(kw in col_lower for kw in kws): return color
    return "#aaaaaa"

CURATED_VARIABLES = [
    'albumin_mean_last', 'albumin_mean_count',
    'bilirubin_mean_last', 'bilirubin_mean_count',
    'alanine aminotransferase_mean_last', 'alanine aminotransferase_mean_count',
    'asparate aminotransferase_mean_last', 'asparate aminotransferase_mean_count',
    'creatinine_mean_last', 'creatinine_mean_count',
    'blood urea nitrogen_mean_last', 'blood urea nitrogen_mean_count',
    'prothrombin time inr_mean_last', 'prothrombin time inr_mean_count',
    'partial thromboplastin time_mean_last', 'partial thromboplastin time_mean_count',
    'hemoglobin_mean_last', 'hemoglobin_mean_count',
    'neutrophils_mean_last', 'neutrophils_mean_count',
    'calcium ionized_mean_last', 'calcium ionized_mean_count',
    'central venous pressure_mean_last', 'central venous pressure_mean_count',
    'peak inspiratory pressure_mean_last', 'peak inspiratory pressure_mean_count',
    'respiratory rate_mean_last', 'respiratory rate_mean_count',
    'lactate_mean_last', 'lactate_mean_count',
    'lactic acid_mean_last', 'lactic acid_mean_count',
    'partial pressure of carbon dioxide_mean_last', 'partial pressure of carbon dioxide_mean_count',
    'glascow coma scale total_mean_last', 'glascow coma scale total_mean_count',
    'potassium_mean_last', 'potassium_mean_count',
    'sodium_mean_last', 'sodium_mean_count',
    'age',
    'gender_encoded'
]

# ==============================================================================
# 1. Main Execution
# ==============================================================================
print("Loading numerical data...")
X_test_full = pd.read_pickle(NUMERICAL_DATA_PATH)
X_test_full = X_test_full.loc[:, X_test_full.std() > 0]
y_test_full = pd.read_pickle(Y_TEST_PATH)

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
    
    top_vars = [v for v in CURATED_VARIABLES if v in r2_scores.index]
    top_vars_sorted = sorted(top_vars, key=lambda v: r2_scores[v])
    
    panel_a_df = pd.DataFrame({
        'variable':  top_vars_sorted,
        'r2_score': r2_scores[top_vars_sorted].values,
        'group':     [assign_group(v) for v in top_vars_sorted],
        'color':     [get_group_color(v) for v in top_vars_sorted],
        'label':     [clean_label(v) for v in top_vars_sorted]
    })
    
    # SHAP
    N_SHAP_SAMPLES = min(500, len(X_test))
    shap_idx = np.random.RandomState(42).choice(len(X_test), N_SHAP_SAMPLES, replace=False)
    
    print("Calculating SHAP...")
    if 'xgboost' in config['nm_model_path']:
        nm_explainer = shap.TreeExplainer(nm_model)
        nm_shap_vals = nm_explainer.shap_values(X_test.iloc[shap_idx])
    else:
        X_bg = X_test.iloc[shap_idx].values
        nm_explainer = shap.LinearExplainer(nm_model, X_test)
        nm_shap_vals = nm_explainer.shap_values(X_bg)
    if isinstance(nm_shap_vals, list): nm_shap_vals = nm_shap_vals[1]

    sm_explainer = shap.TreeExplainer(sm_model)
    sm_shap_vals = sm_explainer.shap_values(E_test[shap_idx])
    if isinstance(sm_shap_vals, list): sm_shap_vals = sm_shap_vals[1]

    nm_mean_abs = np.abs(nm_shap_vals).mean(axis=0)
    unique_nm_labels = set()
    dedup_nm_idx = []
    for idx in np.argsort(nm_mean_abs)[::-1]:
        col_name = X_test.columns[idx]
        if not is_count_or_value(col_name):
            continue
            
        lbl = clean_label(col_name)
        if lbl not in unique_nm_labels:
            unique_nm_labels.add(lbl)
            dedup_nm_idx.append(idx)
        if len(dedup_nm_idx) == 15: break
            
    nm_top15_vals  = nm_mean_abs[dedup_nm_idx]
    nm_top15_labels = [clean_label(X_test.columns[i]) for i in dedup_nm_idx]

    sm_mean_abs = np.abs(sm_shap_vals).mean(axis=0)
    sm_top15_dim_idx = np.argsort(sm_mean_abs)[::-1][:15]
    sm_top15_vals    = sm_mean_abs[sm_top15_dim_idx]
    
    sm_top15_labels = []
    for d in sm_top15_dim_idx:
        dim_corrs = corr_df.iloc[:, d].abs().sort_values(ascending=False)
        seen_base = set()
        top_k = {}
        for var, val in dim_corrs.items():
            if not is_count_or_value(var):
                continue
            s = clean_label(var)
            base_var = s.strip()
            if base_var not in seen_base:
                seen_base.add(base_var)
                top_k[base_var] = val**2
                if len(top_k) == 3: break
        label_parts = []
        for s, r2_val in top_k.items():
            if len(s) > 22: s = s[:19] + "..."
            label_parts.append(f"{s} R²={r2_val:.2f}")
        sm_top15_labels.append(f"Dim {d}  ({', '.join(label_parts)})")
        
    return {
        'panel_a_df': panel_a_df,
        'nm_vals': nm_top15_vals,
        'nm_labels': nm_top15_labels,
        'sm_vals': sm_top15_vals,
        'sm_labels': sm_top15_labels
    }

print("\nStarting combined figure generation...")
results = {}
for t, cfg in TASKS.items():
    results[t] = process_task_data(t, cfg)

fig = plt.figure(figsize=(30, 42), facecolor='white')
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.4, 1, 1], wspace=0.3, hspace=0.3)
fig.suptitle("Embedding Space Interpretability: Mortality vs Readmission", fontsize=28, fontweight='bold', color='#1a1a2e', y=0.92)

task_keys = ['mort_hosp', 'readmission_30']
letters = [['A', 'B', 'C'], ['D', 'E', 'F']]

for row, task in enumerate(task_keys):
    res = results[task]
    title = TASKS[task]['title']
    
    # ---------------------------------------------------------
    # Probes (Left Column)
    # ---------------------------------------------------------
    ax_a = fig.add_subplot(gs[row, 0])
    df = res['panel_a_df']
    bar_colors = df['color'].tolist()
    
    alb_idx = -1
    if 'albumin_mean_last' in df['variable'].tolist():
        alb_idx = df['variable'].tolist().index('albumin_mean_last')
        bar_colors[alb_idx] = '#8b0000'
        
    ax_a.barh(y=range(len(df)), width=df['r2_score'], color=bar_colors, edgecolor='white', linewidth=0.5, height=0.6, zorder=3)
    for i in range(len(df)):
        ax_a.axhspan(i - 0.5, i + 0.5, color='#f7f7f7' if i % 2 == 0 else 'white', zorder=0)

    if alb_idx != -1:
        ax_a.axhspan(alb_idx - 0.5, alb_idx + 0.5, color='#fff3cc', zorder=1, linewidth=0)
        ax_a.axhline(alb_idx - 0.5, color='#e6b800', linewidth=1.2, zorder=2)
        ax_a.axhline(alb_idx + 0.5, color='#e6b800', linewidth=1.2, zorder=2)

    for i, (_, r) in enumerate(df.iterrows()):
        ax_a.text(r['r2_score'] + 0.003, i, f"{r['r2_score']:.3f}", va='center', ha='left', fontsize=14, color='#444444')

    ax_a.set_yticks(range(len(df)))
    ax_a.set_yticklabels(df['label'], fontsize=19)
    ax_a.set_xlabel("Linear Probe R² (5-Fold CV Ridge Regression)", fontsize=20, labelpad=10, color='#444')
    ax_a.set_xlim(0, max(0.9, df['r2_score'].max() * 1.1))
    ax_a.xaxis.grid(True, linestyle='--', linewidth=0.6, color='#dddddd', zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.tick_params(axis='x', labelsize=16)

    for spine in ['top', 'right', 'left']: ax_a.spines[spine].set_visible(False)
    ax_a.spines['bottom'].set_color('#cccccc')
    
    ax_a.set_title(f"{letters[row][0]}. {title}", fontsize=22, pad=14, fontweight='bold', loc='left', color='#1a1a2e')

    if row == 0:
        group_handles = [mpatches.Patch(facecolor=color, label=group, edgecolor='white') for group, (_, color) in CLINICAL_GROUPS.items() if any(assign_group(v) == group for v in df['variable'])]
        if alb_idx != -1: group_handles.append(mpatches.Patch(facecolor='#8b0000', label='Albumin (highlighted)', edgecolor='white'))
        ax_a.legend(handles=group_handles, loc='lower right', fontsize=14, framealpha=0.92, edgecolor='#cccccc', title='Clinical Group', title_fontsize=15, ncol=2)

    # ---------------------------------------------------------
    # SHAP (Middle & Right Columns)
    # ---------------------------------------------------------
    ax_nm = fig.add_subplot(gs[row, 1])
    ax_sm = fig.add_subplot(gs[row, 2])
    
    max_shap = max(res['nm_vals'].max(), res['sm_vals'].max()) * 1.15
    
    for ax, vals, labels, color, m_name, sub, is_right, panel_letter in zip(
        [ax_nm, ax_sm],
        [res['nm_vals'][::-1], res['sm_vals'][::-1]],
        [res['nm_labels'][::-1], res['sm_labels'][::-1]],
        [C_NM, C_SM],
        ["Numerical Model", "Semantic Model"],
        ["Top Variables", "Top Dimensions"],
        [False, True],
        [letters[row][1], letters[row][2]]
    ):
        # We need the SHAP bars to have sensible height. Since the Probes plot has ~40 rows, 
        # and SHAP has 15, the bars will be thicker. That is fine, we just scale height=0.6.
        ax.barh(range(len(vals)), vals, color=color, alpha=0.82, edgecolor='white', linewidth=0.5, height=0.6)
        for i in range(len(vals)):
            ax.axhspan(i - 0.5, i + 0.5, color='#f7f7f7' if i % 2 == 0 else 'white', zorder=0)

        if is_right: ax.yaxis.tick_right()
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=21)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=20, labelpad=8)
        
        ax.set_title(f"{panel_letter}. {m_name}: {sub}", fontsize=22, pad=14, fontweight='bold', loc='left', color='#1a1a2e')
            
        ax.xaxis.grid(True, linestyle='--', linewidth=0.6, color='#e0e0e0', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlim(0, max_shap)

        # Set ylim so that it visually compresses the 15 bars rather than stretching them across 40 rows
        ax.set_ylim(-1, len(vals))

        for i, v in enumerate(vals):
            ax.text(v + max_shap * 0.02, i, f"{v:.4f}", va='center', ha='left', fontsize=15, color='#444')

caption_text = (
    "Figure 1. Embedding Space Interpretability across Clinical Prediction Tasks. (A, D) Out-of-sample R² from Ridge regression linear probes "
    "(5-fold cross-validation) quantifying how faithfully the full embedding space encodes each clinical variable. Variables were curated to "
    "include those defining validated error archetypes, proxies for clinical interventions identified in the comparative failure mode analysis, "
    "and reference variables spanning major physiological domains. Across both embedding models, measurement frequency variables (Count) are "
    "more faithfully encoded than raw physiological values (Value), and demographic variables occupy the highest encoding tier. Albumin Value "
    "is highlighted to emphasize its consistently poor encoding (R² = 0.231, mortality; R² = 0.046, readmission). (B, E) SHAP features importance "
    "for the Numerical Model, which relies on established prognostic variables. (C, F) SHAP features importance for the Semantic Model, with "
    "each embedding dimension annotated by its top three unique clinical correlates (univariate R²). In the mortality task (C), four of fifteen "
    "top dimensions encode gender as a top correlate. In the readmission task (F), six of fifteen top dimensions encode hepatic markers "
    "(ALT Count, AST Count), consistent with the albumin-centered error archetypes reported below."
)
fig.text(0.5, 0.03, caption_text, ha='center', va='top', fontsize=20, color='#333333', linespacing=1.5, wrap=True, bbox=dict(boxstyle='square,pad=1.0', fc='#fcfcfc', ec='#cccccc', lw=1))
plt.subplots_adjust(bottom=0.12)

fig.savefig(os.path.join(OUTPUT_DIR, 'figure_1_interpretability_combined.png'), dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(OUTPUT_DIR, 'figure_1_interpretability_combined.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig)

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

print(f"Done! Combined figures and CSV data saved to {OUTPUT_DIR}")
