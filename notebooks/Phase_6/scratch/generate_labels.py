import os
import pickle
import pandas as pd

# Paths
phase_1_dir = r"d:\Projects\EHR Embeddings\notebooks\Phase_1-2\phase_1_outputs"
phase_3_base = r"d:\Projects\EHR Embeddings\notebooks\Phase_3\phase_3_serialized_data"

target_variables = ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso']

for split in ['train', 'val', 'test']:
    y_path = os.path.join(phase_1_dir, f"y_{split}.pkl")
    print(f"Loading {y_path}")
    with open(y_path, 'rb') as f:
        y_df = pickle.load(f)
        
    for target in target_variables:
        if target in y_df.columns:
            # Create target-specific directory
            target_dir = os.path.join(phase_3_base, target)
            os.makedirs(target_dir, exist_ok=True)
            
            # Save as split_labels.csv (no header, ID and target only)
            out_path = os.path.join(target_dir, f"{split}_labels.csv")
            # We want icustay_id as first column, then the target
            # Drop NaNs for the specific target
            subset = y_df[target].dropna()
            subset.to_csv(out_path, header=False)
            print(f"Saved {out_path} ({len(subset)} rows)")
        else:
            print(f"Target {target} not found in {split}")
