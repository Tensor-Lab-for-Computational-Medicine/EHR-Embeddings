import argparse
import os

from phenotypes import compute_and_save_phenotypes


def main():
    p = argparse.ArgumentParser(description='Compute phenotype features from rules CSV and X_test numeric data')
    p.add_argument('--x_test_path', type=str, required=True, help='Path to X_test.pkl (numeric features)')
    p.add_argument('--scaler_path', type=str, required=True, help='Path to scaler.pkl for inverse transform')
    p.add_argument('--rules_csv', type=str, required=True, help='Path to feature_rules.csv')
    p.add_argument('--out_path', type=str, required=True, help='Output path for X_test_phenotypes.pkl')
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    path = compute_and_save_phenotypes(args.x_test_path, args.scaler_path, args.rules_csv, args.out_path)
    print(f"Wrote phenotypes to: {path}")


if __name__ == '__main__':
    main()


