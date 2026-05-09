import pandas as pd
import pickle
import os

with open('../Phase_1-2/phase_1_outputs/X_test.pkl', 'rb') as f:
    d = pickle.load(f)
    print(f"X_test size: {len(d)}")
