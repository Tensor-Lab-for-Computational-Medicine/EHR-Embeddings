import pickle
import pandas as pd

file_path = 'notebooks/Phase 1 and 2/phase_1_outputs/preprocessed_mort_hosp_trends_True_window_24_gap_6_seed_42_X_train.pkl'
scaler_path = 'notebooks/Phase 1 and 2/phase_1_outputs/preprocessed_mort_hosp_trends_True_window_24_gap_6_seed_42_scaler.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    if isinstance(data, pd.DataFrame):
        print("Columns in the DataFrame:")
        for col in data.columns:
            print(col)
        
        print("\n--- Statistics for weight-related columns (unnormalized) ---")
        weight_cols = [col for col in data.columns if 'weight_mean' in col.lower()]

        if weight_cols:
            # Create a copy to avoid changing the original normalized data
            data_unscaled = data.copy()
            
            # Inverse transform the entire DataFrame
            # The scaler expects a NumPy array, so we convert the DataFrame
            unscaled_values = scaler.inverse_transform(data_unscaled)
            
            # The result is a NumPy array, so we convert it back to a DataFrame
            # with the original columns
            data_unscaled = pd.DataFrame(unscaled_values, columns=data.columns, index=data.index)

            for col in weight_cols:
                print(f"\nStatistics for column: {col}")
                print(data_unscaled[col].describe())
        else:
            print("No weight-related columns found.")
            
    else:
        print("The file does not contain a pandas DataFrame.")

except FileNotFoundError:
    print(f"Error: A file was not found. Searched for:\n- {file_path}\n- {scaler_path}")
except Exception as e:
    print(f"An error occurred: {e}") 