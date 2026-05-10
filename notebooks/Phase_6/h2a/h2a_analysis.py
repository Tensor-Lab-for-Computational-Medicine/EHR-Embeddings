"""
H2a Analysis (Refactored): Quantifying Error Discordance Between NM and SM Models
This script now delegates to reusable functional modules under lib/ for reusability.
"""

import os
import logging
import argparse
import importlib
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.runner import run_h2a_analysis


def _resolve_config(config_module: str):
    module = importlib.import_module(config_module)
    if hasattr(module, 'ConfigH2'):
        return module.ConfigH2()
    raise AttributeError(f"Config module '{config_module}' does not expose ConfigH2")


def main():
    parser = argparse.ArgumentParser(description="Run H2a analysis with a given config module.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_h2_morthosp",
        help="Python module path (importable) exposing ConfigH2 (e.g., config_h2_morthosp)",
    )
    args = parser.parse_args()

    config = _resolve_config(args.config)
    np.random.seed(42)

    # Force the baseline model to use the xgboost numeric model, rather than the champion numeric model
    if hasattr(config, 'BASELINE_MODEL_PATH'):
        base_dir = os.path.dirname(config.BASELINE_MODEL_PATH)
        # Use the uncalibrated model for readmission as it's the newly improved one
        if "readmission_30" in config.BASELINE_MODEL_PATH:
            config.BASELINE_MODEL_PATH = os.path.join(base_dir, 'model_1_xgboost_baseline_calibrated.pkl')
        else:
            config.BASELINE_MODEL_PATH = os.path.join(base_dir, 'model_1_xgboost_baseline_calibrated.pkl')

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'h2a_analysis.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("H2a ANALYSIS: QUANTIFYING ERROR DISCORDANCE (Refactored)")
    logging.info("="*60)
    
    run_h2a_analysis(config)


if __name__ == "__main__":
    main()