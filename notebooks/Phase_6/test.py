import pickle
import pandas as pd
import numpy as np

with open(r"D:\Projects\EHR Embeddings\notebooks\Phase 6 - H2 Analysis\h2a\h2_results\h2a_to_h2b_artifact.pkl", 'rb') as f:
    art = pickle.load(f)

y_true = np.array(art['y_true'])
nm_proba = np.array(art['nm_proba'])

# Compute 90% Sensitivity Threshold
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, nm_proba)
idx_90sens = np.where(tpr >= 0.90)[0]
t_90sens = thresholds[idx_90sens[0]] if len(idx_90sens) > 0 else 0.0

# Compute 90% Specificity Threshold
idx_90spec = np.where(1 - fpr >= 0.90)[0]
t_90spec = thresholds[idx_90spec[-1]] if len(idx_90spec) > 0 else 1.0

print(f"Original Youden NM Threshold: {art['thresholds']['nm']}")
print(f"90% Sens Threshold: {t_90sens}")
print(f"90% Spec Threshold: {t_90spec}")