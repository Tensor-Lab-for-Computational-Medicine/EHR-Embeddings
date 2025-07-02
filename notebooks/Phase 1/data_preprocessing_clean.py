# data_preprocessing_clean.py
"""
Streamlined EHR Data Preprocessing Pipeline

Core Features:
- Age filtering (18-125) with consistent application across datasets
- Count-based missingness tracking (no binary indicators)
- Category-specific feature engineering
- Subject-level splitting (no data leakage)
- Demographic integration with proper encoding
- XGBoost-compatible output with preserved NaNs
"""

import pandas as pd
import numpy as np
import logging
import time
import os
import pickle
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configuration
DEFAULT_CONFIG = {
    'HDF_FILE_PATH': '../../data/raw/all_hourly_data.h5',
    'FEATURE_CLASSIFICATION_PATH': '../../data/processed/eda_results_corrected/feature_classification.csv',
    'OUTPUT_DIR': 'phase_1_outputs',
    'DRY_RUN': True,
    'DRY_RUN_PATIENTS': 1000,
    'USE_CACHED_PREPROCESSING': True,
    'CALCULATE_TRENDS': True,
    'WINDOW_SIZE': 24,
    'GAP_TIME': 6,
    'TARGET_VARIABLE': 'mort_hosp',
    'SEED': 42
}

def set_config(config_dict=None):
    """Set global configuration."""
    global CONFIG
    CONFIG = {**DEFAULT_CONFIG, **(config_dict or {})}
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(CONFIG['OUTPUT_DIR'], 'preprocessing_log.txt'), mode='w'),
            logging.StreamHandler()
        ]
    )

def load_data():
    """Load and filter data by age."""
    logging.info("Loading data...")
    
    with pd.HDFStore(CONFIG['HDF_FILE_PATH'], 'r') as store:
        df_patients = store['/patients']
        df_ts = store['/vitals_labs_mean']
        
        if CONFIG['DRY_RUN']:
            np.random.seed(CONFIG['SEED'])
            sampled_ids = np.random.choice(
                df_patients.index.get_level_values('icustay_id').unique(), 
                CONFIG['DRY_RUN_PATIENTS'], replace=False
            )
            df_patients = df_patients[df_patients.index.get_level_values('icustay_id').isin(sampled_ids)]
            df_ts = df_ts[df_ts.index.get_level_values('icustay_id').isin(sampled_ids)]
    
    # Handle MultiIndex columns
    if isinstance(df_ts.columns, pd.MultiIndex):
        df_ts.columns = pd.Index(['_'.join(col).strip() for col in df_ts.columns.values])
    df_ts = df_ts.loc[:,~df_ts.columns.duplicated()]
    
    # Age filtering (18-125) - applies to both datasets
    if 'age' in df_patients.columns:
        valid_subjects = df_patients.groupby('subject_id')['age'].first()
        valid_subjects = valid_subjects[(valid_subjects >= 18) & (valid_subjects <= 125)].index
        
        df_patients = df_patients[df_patients.index.get_level_values('subject_id').isin(valid_subjects)]
        df_ts = df_ts[df_ts.index.get_level_values('icustay_id').isin(df_patients.index.get_level_values('icustay_id'))]
        
        logging.info(f"Age filtered: {len(valid_subjects)} patients, {df_ts.shape[0]} records")
    
    return df_patients, df_ts

def process_demographics(df_patients):
    """Extract and encode demographic features."""
    demographic_cols = ['age', 'gender', 'ethnicity', 'insurance']
    df_demo = df_patients.groupby(level='subject_id').first()[demographic_cols].copy()
    
    # Fill missing values
    df_demo['age'].fillna(df_demo['age'].median(), inplace=True)
    for col in ['gender', 'ethnicity', 'insurance']:
        if col in df_demo.columns:
            mode_val = df_demo[col].mode().iloc[0] if not df_demo[col].mode().empty else 'Unknown'
            df_demo[col].fillna(mode_val, inplace=True)
            df_demo[col] = df_demo[col].astype(str)
    
    # Encode categorical features (fit on all data - no leakage risk)
    categorical_cols = ['gender', 'ethnicity', 'insurance']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_demo.columns:
            le = LabelEncoder()
            le.fit(df_demo[col])  # Fit on all data - LabelEncoder is just a mapping
            df_demo[f'{col}_encoded'] = le.transform(df_demo[col])
            df_demo.drop(columns=[col], inplace=True)
            label_encoders[col] = le
    
    logging.info(f"Demographics processed: {df_demo.shape}")
    return df_demo, label_encoders

def _calculate_slope(series):
    """Calculate slope of time series."""
    y = series.dropna()
    if len(y) < 2:
        return np.nan
    x = y.index.get_level_values('hours_in').values if isinstance(y.index, pd.MultiIndex) else y.index.values
    try:
        return linregress(x, y.values).slope
    except:
        return np.nan

def engineer_features(df_ts, feature_categories):
    """Engineer features by category with count-based missingness."""
    grouped = df_ts.sort_index().groupby('icustay_id')
    features = {}
    
    # Group features by category
    categories = {}
    for feature, info in feature_categories.items():
        if feature in df_ts.columns:
            category = info['category']
            categories.setdefault(category, []).append(feature)
    
    logging.info(f"Engineering features for {len(categories)} categories...")
    
    # Log category breakdown
    for category, feature_list in categories.items():
        logging.info(f"  - {category}: {len(feature_list)} features")
    
    for category, feature_list in categories.items():
        for feature in feature_list:
            fg = grouped[feature]
            
            if category == 'Static':
                features[f'{feature}_value'] = fg.first()
            elif category == 'Event-Driven':
                features[f'{feature}_count'] = fg.count()
                features[f'{feature}_last'] = fg.last()
            elif category == 'High-Frequency Physiological':
                features.update({
                    f'{feature}_last': fg.last(),
                    f'{feature}_mean': fg.mean(),
                    f'{feature}_min_24h': fg.min(),
                    f'{feature}_max_24h': fg.max(),
                    f'{feature}_stddev_24h': fg.std(),
                    f'{feature}_count': fg.count()
                })
                if CONFIG['CALCULATE_TRENDS']:
                    features[f'{feature}_slope_24h'] = fg.apply(_calculate_slope)
            elif category in ['Labile Lab', 'Stable Index', 'Sparse Dynamic']:
                features[f'{feature}_last'] = fg.last()
                features[f'{feature}_mean'] = fg.mean()
                features[f'{feature}_count'] = fg.count()
                if CONFIG['CALCULATE_TRENDS'] and category != 'Stable Index':
                    features[f'{feature}_slope_24h'] = fg.apply(_calculate_slope)
    
    df_features = pd.DataFrame(features)
    logging.info(f"Features engineered: {df_features.shape}")
    return df_features

def merge_demographics(df_features, df_demographics, icustay_to_subject):
    """Merge demographic features with engineered features."""
    df_features['subject_id'] = df_features.index.map(icustay_to_subject)
    df_merged = df_features.merge(df_demographics, left_on='subject_id', right_index=True, how='left')
    return df_merged.drop(columns=['subject_id'])

def encode_all_categorical_features(datasets, existing_encoders=None):
    """Encode any remaining categorical features across all datasets."""
    if existing_encoders is None:
        existing_encoders = {}
    
    # Check for any non-numeric columns in the training set using a more robust method
    X_train = datasets['X_train']
    non_numeric_cols = []
    
    for col in X_train.columns:
        # Check if column contains non-numeric data
        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'object':
            non_numeric_cols.append(col)
    
    if len(non_numeric_cols) > 0:
        logging.info(f"Encoding remaining categorical columns: {non_numeric_cols}")
        
        for col in non_numeric_cols:
            # Fit encoder on combined data to ensure consistency (no data leakage for label encoding)
            combined_values = pd.concat([datasets['X_train'][col], datasets['X_val'][col], datasets['X_test'][col]])
            combined_values = combined_values.dropna()  # Remove NaN for fitting
            
            if len(combined_values) > 0:
                le = LabelEncoder()
                le.fit(combined_values.astype(str))  # Ensure string type for fitting
                existing_encoders[f'feature_{col}'] = le
                
                # Transform each dataset, preserving NaN values
                for dataset_name in ['X_train', 'X_val', 'X_test']:
                    df = datasets[dataset_name]
                    mask = df[col].notna()
                    if mask.any():
                        # Create a new column with encoded values
                        df[col] = df[col].astype('object')  # Ensure object type
                        df.loc[mask, col] = le.transform(df.loc[mask, col].astype(str))
                        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
                        
        logging.info(f"✓ All categorical features encoded successfully")
    else:
        logging.info("✓ No additional categorical features found - all columns are numeric")
    
    return existing_encoders

def get_cache_prefix():
    """Generate cache filename prefix."""
    prefix = f"preprocessed_{CONFIG['TARGET_VARIABLE']}"
    if CONFIG['DRY_RUN']:
        prefix += f"_dryrun_{CONFIG['DRY_RUN_PATIENTS']}"
    prefix += f"_trends_{CONFIG['CALCULATE_TRENDS']}_window_{CONFIG['WINDOW_SIZE']}_gap_{CONFIG['GAP_TIME']}_seed_{CONFIG['SEED']}"
    return prefix

def save_data(datasets, label_encoders=None):
    """Save all datasets."""
    prefix = get_cache_prefix()
    
    for name, data in datasets.items():
        with open(os.path.join(CONFIG['OUTPUT_DIR'], f'{prefix}_{name}.pkl'), 'wb') as f:
            pickle.dump(data, f)
    
    if label_encoders:
        with open(os.path.join(CONFIG['OUTPUT_DIR'], f'{prefix}_label_encoders.pkl'), 'wb') as f:
            pickle.dump(label_encoders, f)
    
    logging.info(f"Data saved with prefix: {prefix}")

def load_cached_data():
    """Check if cached data exists."""
    prefix = get_cache_prefix()
    required_files = [f'{prefix}_{name}.pkl' for name in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']]
    return all(os.path.exists(os.path.join(CONFIG['OUTPUT_DIR'], f)) for f in required_files)

def main(config_dict=None):
    """Main preprocessing pipeline."""
    set_config(config_dict)
    setup_logging()
    
    start_time = time.time()
    
    # Check cache
    if CONFIG['USE_CACHED_PREPROCESSING'] and load_cached_data():
        logging.info("Using cached data - skipping preprocessing!")
        return
    
    # Load data
    df_patients, df_ts_raw = load_data()
    
    # Apply time window filtering
    patient_max_hours = df_ts_raw.groupby(level='icustay_id').apply(
        lambda x: x.index.get_level_values('hours_in').max()
    )
    valid_patients = patient_max_hours[patient_max_hours > (CONFIG['WINDOW_SIZE'] + CONFIG['GAP_TIME'])]
    
    df_patients = df_patients[df_patients.index.get_level_values('icustay_id').isin(valid_patients.index)]
    df_ts_raw = df_ts_raw[
        (df_ts_raw.index.get_level_values('icustay_id').isin(valid_patients.index)) &
        (df_ts_raw.index.get_level_values('hours_in') < CONFIG['WINDOW_SIZE'])
    ]
    
    logging.info(f"Time window filtered: {len(valid_patients)} patients, {df_ts_raw.shape} time-series")
    
    # Load feature classifications
    df_classification = pd.read_csv(CONFIG['FEATURE_CLASSIFICATION_PATH'])
    feature_categories = {row['feature_name']: {'category': row['category']} for _, row in df_classification.iterrows()}
    
    # Subject-level splits (no leakage)
    subject_outcomes = df_patients.groupby('subject_id')[CONFIG['TARGET_VARIABLE']].max()
    subjects, outcomes = subject_outcomes.index.values, subject_outcomes.values
    
    train_val_subjects, test_subjects, _, _ = train_test_split(
        subjects, outcomes, test_size=0.25, random_state=CONFIG['SEED'], stratify=outcomes
    )
    train_subjects, val_subjects, _, _ = train_test_split(
        train_val_subjects, subject_outcomes[train_val_subjects], 
        test_size=0.125, random_state=CONFIG['SEED'], 
        stratify=subject_outcomes[train_val_subjects]
    )
    
    # Get ICU stay IDs for each split
    splits = {}
    for split_name, split_subjects in [('train', train_subjects), ('val', val_subjects), ('test', test_subjects)]:
        icustay_ids = df_patients[df_patients.index.get_level_values('subject_id').isin(split_subjects)].index.get_level_values('icustay_id').unique()
        splits[split_name] = {'subjects': split_subjects, 'icustays': icustay_ids}
    
    logging.info(f"Splits: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)} subjects")
    
    # Process demographics (fit encoders on all data - no leakage risk)
    df_demographics, label_encoders = process_demographics(df_patients)
    
    # Create ICU stay to subject mapping
    icustay_to_subject = df_patients.index.get_level_values('subject_id').to_series(
        index=df_patients.index.get_level_values('icustay_id')
    )
    
    # Engineer features for each split
    datasets = {}
    for split_name, split_info in splits.items():
        # Filter time-series data for this split
        ts_mask = df_ts_raw.index.get_level_values('icustay_id').isin(split_info['icustays'])
        df_ts_split = df_ts_raw[ts_mask]
        
        # Engineer features and merge demographics
        df_features = engineer_features(df_ts_split, feature_categories)
        df_final = merge_demographics(df_features, df_demographics, icustay_to_subject)
        
        # Prepare targets
        y = df_patients.loc[df_patients.index.get_level_values('icustay_id').isin(df_final.index), CONFIG['TARGET_VARIABLE']]
        y = y.groupby('icustay_id').first().reindex(df_final.index)
        
        datasets[f'X_{split_name}'] = df_final
        datasets[f'y_{split_name}'] = y
    
    # Align feature columns across splits
    all_features = set()
    for name in ['X_train', 'X_val', 'X_test']:
        all_features.update(datasets[name].columns)
    
    # Reindex with proper dtype handling
    sorted_features = sorted(all_features)
    for name in ['X_train', 'X_val', 'X_test']:
        # Create missing columns first with proper dtypes
        missing_cols = set(sorted_features) - set(datasets[name].columns)
        for col in missing_cols:
            datasets[name][col] = np.nan
        # Reindex to ensure column order
        datasets[name] = datasets[name][sorted_features]
    
    # Final categorical encoding for any remaining non-numeric columns
    label_encoders = encode_all_categorical_features(datasets, label_encoders)
    
    # Create scaler and imputation values
    datasets['scaler'] = StandardScaler()
    # Get numeric columns using a more robust method
    numeric_cols = []
    for col in datasets['X_train'].columns:
        # Check if column contains numeric data
        if (datasets['X_train'][col].dtype in ['int64', 'float64', 'int32', 'float32'] or 
            'int' in str(datasets['X_train'][col].dtype) or 
            'float' in str(datasets['X_train'][col].dtype)):
            numeric_cols.append(col)
    
    if len(numeric_cols) > 0:
        datasets['scaler'].fit(datasets['X_train'][numeric_cols].fillna(0))
    datasets['imputation_values'] = datasets['X_train'].median()
    
    # Save everything
    save_data(datasets, label_encoders)
    
    # Summary
    total_time = time.time() - start_time
    logging.info(f"Preprocessing completed in {total_time/60:.2f} minutes")
    logging.info(f"Final shapes: X_train={datasets['X_train'].shape}, X_val={datasets['X_val'].shape}, X_test={datasets['X_test'].shape}")
    logging.info(f"Mortality rates: train={datasets['y_train'].mean():.3f}, val={datasets['y_val'].mean():.3f}, test={datasets['y_test'].mean():.3f}")

if __name__ == "__main__":
    main()
