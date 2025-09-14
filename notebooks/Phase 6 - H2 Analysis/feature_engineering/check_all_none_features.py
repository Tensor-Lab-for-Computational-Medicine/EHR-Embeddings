import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


def find_all_none_columns(csv_path: Path) -> List[str]:
	missing_tokens = {"", "None", "none", "NA", "N/A", "null", "NULL", "NaN", "nan"}
	# Parse with default NA handling plus explicit tokens
	df = pd.read_csv(csv_path, keep_default_na=True, na_values=list(missing_tokens), skipinitialspace=True)
	# Normalize object columns: strip whitespace and blank-to-NaN
	for col in df.select_dtypes(include=["object"]).columns:
		s = df[col].astype(str).str.strip()
		# Empty strings to NaN
		s = s.mask(s == "", np.nan)
		# Known missing tokens to NaN (post-strip)
		s = s.replace(list(missing_tokens), np.nan)
		df[col] = s
	return [c for c in df.columns if df[c].isna().all()]


def main():
	parser = argparse.ArgumentParser(description="List columns with all None/NaN/empty values in a CSV.")
	default_csv = Path(__file__).parent / "artifacts" / "X_test_phenotypes.csv"
	parser.add_argument("csv", nargs="?", type=Path, default=default_csv, help=f"Path to CSV (default: {default_csv})")
	parser.add_argument("--save", type=Path, default=None, help="Optional path to save the list as CSV")
	args = parser.parse_args()

	all_none_cols = find_all_none_columns(args.csv)
	print(f"Total columns: {len(pd.read_csv(args.csv, nrows=0).columns)}")
	print(f"Columns entirely None/NaN/empty: {len(all_none_cols)}")
	if all_none_cols:
		print("\nColumn names:")
		for name in all_none_cols:
			print(name)

	if args.save:
		pd.Series(all_none_cols, name="all_none_columns").to_csv(args.save, index=False)
		print(f"\nSaved list to: {args.save}")


if __name__ == "__main__":
	main()


