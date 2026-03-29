# save as print_denominator.py and run from repo root:
#   python print_denominator.py
import sys, os, re, pickle
from pathlib import Path
import pandas as pd

# Add h2b to import path
h2b_dir = Path.cwd() / "notebooks" / "Phase 6 - H2 Analysis" / "h2b"
sys.path.insert(0, str(h2b_dir))

from config_h2_readmin30 import ConfigH2 as C  # adjust if you use a different config
cfg = C()

# Load X_test numeric matrix
with open(cfg.X_TEST_NUM_PATH, "rb") as f:
    X = pickle.load(f)
df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

# Count unique feature families from *_mean_count and *_mean_count_6h
cols = [c for c in df.columns if c.endswith("_mean_count") or c.endswith("_mean_count_6h")]
families = {re.sub(r"_mean_count(_6h)?$", "", c) for c in cols}

print(f"Unique feature families (denominator): {len(families)}")
# Optional: print a few names to sanity-check
print("Sample families:", sorted(list(families))[:10])