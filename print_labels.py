import pandas as pd
import numpy as np

X_test_full = pd.read_pickle('notebooks/Phase 1 and 2/phase_1_outputs/X_test.pkl')
X_test_full = X_test_full.loc[:, X_test_full.std() > 0]

embedding_vectors, valid_indices = [], []
for icustay_id in X_test_full.index:
    import os
    fp = f'notebooks/Phase 4/embeddings_text-embedding-005/F1_P0/test/{icustay_id}.npy'
    if os.path.exists(fp):
        embedding_vectors.append(np.load(fp))
        valid_indices.append(icustay_id)
        
E_test = np.vstack(embedding_vectors)
X_test = X_test_full.loc[valid_indices].copy()

X_vals = X_test.values.astype(float)
E_vals = E_test

X_norm = (X_vals - X_vals.mean(0)) / (X_vals.std(0) + 1e-9)
E_norm = (E_test - E_test.mean(0)) / (E_test.std(0) + 1e-9)
corr_matrix = (X_norm.T @ E_norm) / X_norm.shape[0]

corr_df = pd.DataFrame(corr_matrix, index=X_test.columns, columns=[f"Dim_{i}" for i in range(E_test.shape[1])])

import pickle
import shap

with open(f"notebooks/Phase 1 and 2/model_evaluation_outputs/mort_hosp/xgboost/best_model_xgboost_mort_hosp.pkl", 'rb') as f:
    best_model_data = pickle.load(f)
sm_model = best_model_data['model']

explainer = shap.TreeExplainer(sm_model)
shap_values = explainer.shap_values(E_test)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

sm_top15_dim_idx = np.argsort(mean_abs_shap)[::-1][:15]

for d in sm_top15_dim_idx:
    dim_corrs = corr_df.iloc[:, d].abs().sort_values(ascending=False)
    seen_base = set()
    top_k = {}
    
    def clean_label(l):
        l = l.replace('_mean_last', '').replace('_mean_count', '').replace('_mean_mean', '').replace('_encoded', '')
        l = l.replace('_mean_min_24h', '').replace('_mean_max_24h', '').replace('_mean_slope_24h', '')
        l = l.replace('_mean_slope_6h', '').replace('_mean_count_6h', '').replace('_mean_stddev_24h', '')
        l = l.title()
        return l
        
    for var, val in dim_corrs.items():
        s = clean_label(var)
        reps = {
            'Partial Pressure Of Carbon Dioxide': 'pCO2',
            'Partial Pressure Of Oxygen': 'pO2',
            'Fraction Inspired Oxygen': 'FiO2',
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
            'Mean Corpuscula...': 'MCH',
            'Mean Corpuscular Volume': 'MCV',
            'Red Blood Cell Count': 'RBC'
        }
        for old, new in reps.items():
            s = s.replace(old, new)
        
        base_var = s.strip()
        if base_var not in seen_base:
            seen_base.add(base_var)
            top_k[base_var] = val**2
            if len(top_k) == 2:
                break
    
    label_parts = []
    for s, r2_val in top_k.items():
        if len(s) > 22:
            s = s[:19] + "..."
        label_parts.append(f"{s} R²={r2_val:.2f}")
    print(f"Dim {d}  ({', '.join(label_parts)})")
