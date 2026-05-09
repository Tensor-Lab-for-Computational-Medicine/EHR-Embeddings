import pickle
import pandas as pd

y_path = r"d:\Projects\EHR Embeddings\notebooks\Phase_1-2\phase_1_outputs\y_test.pkl"
with open(y_path, 'rb') as f:
    df = pickle.load(f)

print("Columns:", df.columns.tolist())
print("Index name:", df.index.name)
print("Index head:", df.index[:5].tolist())
print("DF head:\n", df.head())
