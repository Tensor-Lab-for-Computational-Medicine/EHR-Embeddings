#!/usr/bin/env python
"""Debug script to test model loading"""

import sys
import os

# Add the Phase 6 directory to path
sys.path.append(r'D:\Projects\EHR Embeddings\notebooks\Phase 6 - H2 Analysis')

try:
    import config_h2
    config = config_h2.ConfigH2()
    print("✓ Config created successfully")
    print(f"Baseline model path: {config.BASELINE_MODEL_PATH}")
    print(f"Champion model path: {config.CHAMPION_MODEL_PATH}")

    print(f"Baseline model exists: {os.path.exists(config.BASELINE_MODEL_PATH)}")
    print(f"Champion model exists: {os.path.exists(config.CHAMPION_MODEL_PATH)}")

    # Try loading both models
    print("\nTrying to load baseline model...")
    import pickle
    import numpy as np

    # Apply the same fix we used before
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = np.core

    try:
        with open(config.BASELINE_MODEL_PATH, 'rb') as f:
            baseline_model = pickle.load(f)
        print(f"✓ Baseline model loaded: {type(baseline_model)}")
    except Exception as e:
        print(f"✗ Baseline model failed: {e}")

    print("\nTrying to load champion model...")
    # Check file size first
    file_size = os.path.getsize(config.CHAMPION_MODEL_PATH)
    print(f"Champion model file size: {file_size} bytes")

    try:
        with open(config.CHAMPION_MODEL_PATH, 'rb') as f:
            champion_model = pickle.load(f)
        print(f"✓ Champion model loaded: {type(champion_model)}")
    except Exception as e:
        print(f"✗ Champion model failed: {e}")
        import traceback
        traceback.print_exc()

        # Try with joblib as fallback
        print("\nTrying joblib fallback...")
        try:
            import joblib
            champion_model = joblib.load(config.CHAMPION_MODEL_PATH)
            print(f"✓ Champion model loaded with joblib: {type(champion_model)}")
        except Exception as e2:
            print(f"✗ Joblib also failed: {e2}")
            import traceback
            traceback.print_exc()

    # Clean up
    if 'numpy._core' in sys.modules and sys.modules['numpy._core'] is np.core:
        del sys.modules['numpy._core']

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
