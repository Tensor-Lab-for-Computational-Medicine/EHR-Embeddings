import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

X_test_full = pd.read_pickle('notebooks/Phase 1 and 2/phase_1_outputs/X_test.pkl')
X_test_full = X_test_full.loc[:, X_test_full.std() > 0]
y_test_full = pd.read_pickle('notebooks/Phase 1 and 2/phase_1_outputs/y_test.pkl')

task_name = 'readmission_30'
embedding_dir = 'notebooks/Phase 4/embeddings_text-embedding-005/F1_P0/test'

embedding_vectors, valid_indices = [], []
for icustay_id in X_test_full.index:
    fp = f"{embedding_dir}/{icustay_id}.npy"
    import os
    if os.path.exists(fp):
        embedding_vectors.append(np.load(fp))
        valid_indices.append(icustay_id)
        
E_test = np.vstack(embedding_vectors)
X_test = X_test_full.loc[valid_indices].copy()

X_vals = X_test.values.astype(float)
E_vals = E_test

# Try Ridge with CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros_like(X_vals)

for train_idx, test_idx in kf.split(E_vals):
    model = Ridge(alpha=1.0)
    model.fit(E_vals[train_idx], X_vals[train_idx])
    preds[test_idx] = model.predict(E_vals[test_idx])

r2_scores = pd.Series([r2_score(X_vals[:, i], preds[:, i]) for i in range(X_vals.shape[1])], index=X_test.columns)
print(r2_scores.sort_values(ascending=False).head(20))
