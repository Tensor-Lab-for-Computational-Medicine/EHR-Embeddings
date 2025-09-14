import argparse
import os
import pickle
import pandas as pd

from phenotypes import compute_and_save_phenotypes, compute_phenotypes


def main():
    p = argparse.ArgumentParser(description='Compute phenotype features from rules CSV and numeric data (train/test)')
    p.add_argument('--x_path', type=str, help='Path to X numeric .pkl (train or test)')
    p.add_argument('--x_test_path', type=str, help='Backward-compat: Path to X_test.pkl (numeric features)')
    p.add_argument('--x2_path', type=str, help='Optional second X path to vertically concatenate (e.g., val)')
    p.add_argument('--scaler_path', type=str, required=True, help='Path to scaler.pkl for inverse transform')
    p.add_argument('--rules_csv', type=str, required=True, help='Path to feature_rules.csv')
    p.add_argument('--out_path', type=str, required=True, help='Output path for phenotypes .pkl')
    args = p.parse_args()

    x_path = args.x_path or args.x_test_path
    if not x_path:
        raise SystemExit('--x_path is required (or use --x_test_path for backward compatibility)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    if args.x2_path:
        # Custom combine: load both, inverse transform, then compute and save phenotypes
        with open(args.scaler_path, 'rb') as f:
            sc = pickle.load(f)
        X1 = pd.read_pickle(x_path)
        X2 = pd.read_pickle(args.x2_path)
        if not isinstance(X1, pd.DataFrame):
            X1 = pd.DataFrame(X1)
        if not isinstance(X2, pd.DataFrame):
            X2 = pd.DataFrame(X2)
        try:
            if hasattr(sc, 'feature_names_in_'):
                X1[sc.feature_names_in_] = sc.inverse_transform(X1[sc.feature_names_in_])
                X2[sc.feature_names_in_] = sc.inverse_transform(X2[sc.feature_names_in_])
            else:
                X1 = pd.DataFrame(sc.inverse_transform(X1), columns=X1.columns, index=X1.index)
                X2 = pd.DataFrame(sc.inverse_transform(X2), columns=X2.columns, index=X2.index)
        except Exception:
            pass
        X = pd.concat([X1, X2], axis=0)
        phenos = compute_phenotypes(X, args.rules_csv)
        phenos.to_pickle(args.out_path)
        try:
            phenos.to_csv(args.out_path.replace('.pkl', '.csv'), index=False)
        except Exception:
            pass
        print(f"Wrote phenotypes to: {args.out_path}")
    else:
        path = compute_and_save_phenotypes(x_path, args.scaler_path, args.rules_csv, args.out_path)
        print(f"Wrote phenotypes to: {path}")


if __name__ == '__main__':
    main()


