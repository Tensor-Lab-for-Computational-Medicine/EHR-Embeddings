import pandas as pd
import pickle
import numpy as np

# Using raw strings for windows paths
files_to_check = [
    r'notebooks\Phase 1 and 2\phase_1_outputs\preprocessed_mort_hosp_los_3_los_7_trends_True_window_24_gap_6_seed_42_X_test.pkl',
    r'notebooks\Phase 1 and 2\phase_1_outputs\preprocessed_mort_hosp_los_3_los_7_trends_True_window_24_gap_6_seed_42_y_test.pkl',
    r'notebooks\Phase 1 and 2\phase_1_outputs\preprocessed_mort_hosp_los_3_los_7_trends_True_window_24_gap_6_seed_42_y_val.pkl'
]

for file_path in files_to_check:
    print(f"Analyzing file: {file_path}")
    print("-" * 30)
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            if hasattr(data, "toarray"): # For sparse matrices
                data = data.toarray()
            df = pd.DataFrame(data)
        else:
            df = data

        if df.empty:
            print("The file contains an empty DataFrame or data structure.")
        else:
            nan_values = df.isnull().sum()
            
            # Check for infinite values only in numeric columns
            inf_values = np.isinf(df.select_dtypes(include=np.number)).sum()
            inf_values = inf_values.reindex(df.columns, fill_value=0)

            total_missing = nan_values + inf_values.astype(int)
            
            # If it's a multi-index dataframe, this will be a Series.
            if total_missing.sum() == 0:
                 print("No missing values (NaN or Inf) found.")
                 print(df.head())
            else:
                missing_percentage = (total_missing / len(df)) * 100 if len(df) > 0 else 0
            
                result = pd.DataFrame({
                    'NaN Values': nan_values,
                    'Inf Values': inf_values,
                    'Total Missing': total_missing,
                    'Percentage (%)': missing_percentage
                })
                
                # Only show columns with missing values
                result_with_missing = result[result['Total Missing'] > 0]
                
                if result_with_missing.empty:
                    print("No missing values (NaN or Inf) found.")
                else:
                    print("Found NaN or Inf values:")
                    print(result_with_missing)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
    print("\n" + "="*50 + "\n") 