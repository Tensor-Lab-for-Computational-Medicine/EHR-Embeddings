import pickle
import pandas as pd
import os

path = r'd:\Projects\EHR Embeddings\notebooks\Phase_1-2\phase_1_outputs\X_train.pkl'
if os.path.exists(path):
    with open(path, 'rb') as f:
        X = pickle.load(f)
    with open('notebooks/feature_list.txt', 'w') as f:
        f.write("\n".join(X.columns))
    print(f"Shape: {X.shape}")
    print(f"Columns written to notebooks/feature_list.txt")
else:
    print("File not found")
