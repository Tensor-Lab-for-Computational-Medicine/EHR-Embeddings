import pandas as pd
import os

f = 'd:/Projects/EHR Embeddings/notebooks/Phase_6/h2b/h2_results/mort_hosp/final_archetypes.csv'
if os.path.exists(f):
    df = pd.read_csv(f)
    print(f"final_archetypes.csv: {len(df)} rows")
    if 'coverage' in df.columns and 'coverage_pct' in df.columns:
        c = df['coverage'].iloc[0]
        p = df['coverage_pct'].iloc[0]
        if p > 0:
            n = c / (p / 100.0)
            print(f"First row: coverage={c}, pct={p} -> inferred N={n:.2f}")
else:
    print(f"File NOT found: {f}")
