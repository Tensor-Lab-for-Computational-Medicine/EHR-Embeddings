import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

X_test_full = pd.read_pickle('notebooks/Phase 1 and 2/phase_1_outputs/X_test.pkl')
X_test_full = X_test_full.loc[:, X_test_full.std() > 0]

# Load actual E_test
embedding_vectors, valid_indices = [], []
for icustay_id in X_test_full.index:
    import os
    fp = f"notebooks/Phase 4/embeddings_text-embedding-005/F1_P0/test/{icustay_id}.npy"
    if os.path.exists(fp):
        embedding_vectors.append(np.load(fp))
        valid_indices.append(icustay_id)
        
E_test = np.vstack(embedding_vectors)
X_test = X_test_full.loc[valid_indices].copy()

X_vals = X_test.values.astype(float)
E_vals = E_test

var_name = 'central venous pressure_mean_last'
idx = X_test.columns.get_loc(var_name)

# 1. Pearson r squared for best dim
X_norm = (X_vals - X_vals.mean(0)) / (X_vals.std(0) + 1e-9)
E_norm = (E_test - E_test.mean(0)) / (E_test.std(0) + 1e-9)
corr = (X_norm[:, idx].T @ E_norm) / X_norm.shape[0]
best_dim = np.argmax(np.abs(corr))
max_r = corr[best_dim]
print(f"{var_name}: max |r| = {np.abs(max_r):.4f}, max r^2 = {max_r**2:.4f} (Dim {best_dim})")

# 2. Ridge CV with unnormalized X
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds_unnorm = np.zeros(X_vals.shape[0])
for train_idx, test_idx in kf.split(E_vals):
    model = Ridge(alpha=1.0)
    model.fit(E_vals[train_idx], X_vals[train_idx, idx])
    preds_unnorm[test_idx] = model.predict(E_vals[test_idx])
print(f"Ridge CV R2 (unnormalized X) = {r2_score(X_vals[:, idx], preds_unnorm):.4f}")

# 3. Ridge CV with normalized X
preds_norm = np.zeros(X_vals.shape[0])
for train_idx, test_idx in kf.split(E_vals):
    model = Ridge(alpha=1.0)
    model.fit(E_vals[train_idx], X_norm[train_idx, idx])
    preds_norm[test_idx] = model.predict(E_vals[test_idx])
print(f"Ridge CV R2 (normalized X) = {r2_score(X_norm[:, idx], preds_norm):.4f}")

# Compare univariate vs multivariate in CV
preds_uni = np.zeros(X_vals.shape[0])
for train_idx, test_idx in kf.split(E_vals):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(E_vals[train_idx, best_dim].reshape(-1, 1), X_vals[train_idx, idx])
    preds_uni[test_idx] = model.predict(E_vals[test_idx, best_dim].reshape(-1, 1))
print(f"Univariate OLS CV R2 (Dim {best_dim}) = {r2_score(X_vals[:, idx], preds_uni):.4f}")

