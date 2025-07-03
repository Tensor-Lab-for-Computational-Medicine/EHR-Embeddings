# data_preprocessing_clean.py
"""
Streamlined EHR Data Preprocessing Pipeline

Core Features:
- Age filtering (18-125) with consistent application across datasets
- Category-specific feature engineering with proper 6h/24h windows
- Subject-level splitting (no data leakage)
- Demographic integration with proper encoding and reversible label encoders
- Percentile-based outlier handling (1st-99th percentiles) to prevent skewed imputation
- Advanced imputation scheme for derived features:
  * Value-based features (_last, _mean, etc.): Training set median
  * Variation features (_stddev_24h): 0 (no variation)
  * Trend features (_slope_6h, _slope_24h): 0 (no trend)
- XGBoost-compatible output with zero missing values
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
    'SEED': 42,
    'HANDLE_OUTLIERS': True,
    'OUTLIER_PERCENTILES': (1, 99)  # (lower, upper) percentiles for clipping
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
    
    # Handle MultiIndex columns and duplicates
    if isinstance(df_ts.columns, pd.MultiIndex):
        df_ts.columns = pd.Index(['_'.join(col).strip() for col in df_ts.columns.values])
    df_ts = df_ts.loc[:,~df_ts.columns.duplicated()]
    
    # Age filtering (18-125)
    if 'age' in df_patients.columns:
        valid_subjects = df_patients.groupby('subject_id')['age'].first()
        valid_subjects = valid_subjects[(valid_subjects >= 18) & (valid_subjects <= 125)].index
        
        df_patients = df_patients[df_patients.index.get_level_values('subject_id').isin(valid_subjects)]
        df_ts = df_ts[df_ts.index.get_level_values('icustay_id').isin(df_patients.index.get_level_values('icustay_id'))]
        
        logging.info(f"Age filtered: {len(valid_subjects)} patients, {df_ts.shape[0]} records")
    
    return df_patients, df_ts

def process_demographics(df_patients):
    """Extract and encode demographic features with reversible encoders."""
    demographic_cols = ['age', 'gender', 'ethnicity', 'insurance']
    df_demo = df_patients.groupby(level='subject_id').first()[demographic_cols].copy()
    
    # Fill missing age with median
    df_demo['age'].fillna(df_demo['age'].median(), inplace=True)
    
    # Encode categorical columns
    categorical_cols = ['gender', 'ethnicity', 'insurance']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_demo.columns:
            mode_val = df_demo[col].mode().iloc[0] if not df_demo[col].mode().empty else 'Unknown'
            df_demo[col].fillna(mode_val, inplace=True)
            
            # Encode and replace original column
            le = LabelEncoder()
            le.fit(df_demo[col].astype(str))
            df_demo[f'{col}_encoded'] = le.transform(df_demo[col].astype(str))
            df_demo.drop(columns=[col], inplace=True)
            
            # Store encoder with clear mapping for reversal
            label_encoders[f'{col}_encoded'] = {
                'encoder': le,
                'classes': le.classes_.tolist(),
                'mapping': {i: cls for i, cls in enumerate(le.classes_)}
            }
    
    logging.info(f"Demographics processed: {df_demo.shape}")
    logging.info(f"Demographic encoders: {list(label_encoders.keys())}")
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

def _engineer_static_features(fg, feature, features):
    """Engineer static features."""
    features[f'{feature}_value'] = fg.first()
    features[f'{feature}_count'] = fg.count()

def _engineer_event_driven_features(fg, feature, features):
    """Engineer event-driven features."""
    features[f'{feature}_count'] = fg.count()
    features[f'{feature}_last'] = fg.last()

def _engineer_high_freq_features(fg, fg_6h, feature, features):
    """Engineer high-frequency physiological features."""
    features.update({
        f'{feature}_last': fg.last(),
        f'{feature}_mean': fg.mean(),
        f'{feature}_min_24h': fg.min(),
        f'{feature}_max_24h': fg.max(),
        f'{feature}_stddev_24h': fg.std(),
        f'{feature}_count': fg.count()
    })
    if fg_6h is not None:
        features[f'{feature}_count_6h'] = fg_6h.count()
    if CONFIG['CALCULATE_TRENDS']:
        features[f'{feature}_slope_24h'] = fg.apply(_calculate_slope)
        if fg_6h is not None:
            features[f'{feature}_slope_6h'] = fg_6h.apply(_calculate_slope)

def _engineer_labile_lab_features(fg, feature, features):
    """Engineer labile lab features."""
    features.update({
        f'{feature}_last': fg.last(),
        f'{feature}_mean': fg.mean(),
        f'{feature}_stddev_24h': fg.std(),
        f'{feature}_count': fg.count()
    })
    if CONFIG['CALCULATE_TRENDS']:
        features[f'{feature}_slope_24h'] = fg.apply(_calculate_slope)

def _engineer_stable_index_features(fg, feature, features):
    """Engineer stable index features."""
    features[f'{feature}_last'] = fg.last()
    features[f'{feature}_count'] = fg.count()

def _engineer_sparse_dynamic_features(fg, feature, features):
    """Engineer sparse dynamic features."""
    features.update({
        f'{feature}_last': fg.last(),
        f'{feature}_mean': fg.mean(),
        f'{feature}_count': fg.count()
    })
    if CONFIG['CALCULATE_TRENDS']:
        features[f'{feature}_slope_24h'] = fg.apply(_calculate_slope)

def engineer_features(df_ts, feature_categories):
    """Engineer features by category with count-based missingness."""
    grouped = df_ts.sort_index().groupby('icustay_id')
    df_ts_6h = df_ts[df_ts.index.get_level_values('hours_in') < 6]
    grouped_6h = df_ts_6h.sort_index().groupby('icustay_id') if not df_ts_6h.empty else None
    
    # Group features by category
    categories = {}
    for feature, info in feature_categories.items():
        if feature in df_ts.columns:
            categories.setdefault(info['category'], []).append(feature)
    
    logging.info(f"Engineering features for {len(categories)} categories")
    
    # Feature engineering dispatch
    engineers = {
        'Static': _engineer_static_features,
        'Event-Driven': _engineer_event_driven_features,
        'High-Frequency Physiological': _engineer_high_freq_features,
        'Labile Lab': _engineer_labile_lab_features,
        'Stable Index': _engineer_stable_index_features,
        'Sparse Dynamic': _engineer_sparse_dynamic_features
    }
    
    features = {}
    for category, feature_list in categories.items():
        engineer_func = engineers.get(category)
        if not engineer_func:
            continue
            
        for feature in feature_list:
            fg = grouped[feature]
            fg_6h = grouped_6h[feature] if grouped_6h is not None and feature in df_ts_6h.columns else None
            
            if category == 'High-Frequency Physiological':
                engineer_func(fg, fg_6h, feature, features)
            else:
                engineer_func(fg, feature, features)
    
    df_features = pd.DataFrame(features)
    logging.info(f"Features engineered: {df_features.shape}")
    return df_features

def merge_demographics(df_features, df_demographics, icustay_to_subject):
    """Merge demographic features with engineered features."""
    # Store original icustay_id index
    original_index = df_features.index
    
    # Reset index to make icustay_id a column
    df_features_reset = df_features.reset_index()
    df_features_reset['subject_id'] = df_features_reset['icustay_id'].map(icustay_to_subject)
    
    # Merge with demographics
    df_merged = df_features_reset.merge(df_demographics, left_on='subject_id', right_index=True, how='left')
    
    # Restore icustay_id as index and drop temporary columns
    df_merged = df_merged.set_index('icustay_id').drop(columns=['subject_id'])
    
    return df_merged

def verify_no_categorical_features(datasets):
    """Verify that all features are numeric (no categorical features remain)."""
    X_train = datasets['X_train']
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
    
    if categorical_cols:
        logging.warning(f"Unexpected categorical features found: {categorical_cols}")
        return False
    else:
        logging.info("✓ All features are numeric - no categorical encoding needed")
        return True

def handle_outliers_percentile_clipping(datasets):
    """Handle extreme outliers using percentile-based clipping to prevent skewed imputation."""
    if not CONFIG['HANDLE_OUTLIERS']:
        logging.info("Outlier handling disabled")
        return datasets
    
    X_train = datasets['X_train']
    lower_pct, upper_pct = CONFIG['OUTLIER_PERCENTILES']
    
    # Calculate bounds from training data only (prevent data leakage)
    numeric_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    bounds = {}
    total_clipped = 0
    
    for col in numeric_cols:
        values = X_train[col].dropna()
        if len(values) > 0:
            lower, upper = values.quantile([lower_pct/100, upper_pct/100])
            if lower != upper:  # Skip if no variation
                bounds[col] = (lower, upper)
    
    # Apply clipping to all datasets
    for dataset_name in ['X_train', 'X_val', 'X_test']:
        dataset = datasets[dataset_name]
        for col, (lower, upper) in bounds.items():
            if col in dataset.columns:
                clipped = ((dataset[col] < lower) | (dataset[col] > upper)).sum()
                dataset[col] = dataset[col].clip(lower=lower, upper=upper)
                total_clipped += clipped
    
    # Store bounds for reproducibility
    datasets['outlier_bounds'] = pd.DataFrame.from_dict(bounds, orient='index', columns=['lower', 'upper'])
    
    logging.info(f"✓ Outlier handling: {len(bounds)} features, {total_clipped} values clipped ({lower_pct}-{upper_pct} percentiles)")
    return datasets

def impute_derived_features(datasets):
    """Apply the specific imputation scheme for derived features."""
    logging.info("Applying derived feature imputation scheme...")
    
    X_train = datasets['X_train']
    
    # Define imputation rules
    rules = {
        'value_based': (['_last', '_mean', '_min', '_max', '_min_24h', '_max_24h', '_value'], 'median'),
        'variation_based': (['_stddev_24h'], 0.0),
        'trend_based': (['_slope_6h', '_slope_24h'], 0.0)
    }
    
    imputation_values = {}
    counts = {rule: 0 for rule in rules}
    other_count = 0
    
    for col in X_train.columns:
        rule_applied = False
        for rule_name, (suffixes, value) in rules.items():
            if any(col.endswith(suffix) for suffix in suffixes):
                if value == 'median':
                    median_val = X_train[col].median()
                    # Handle case where median is NaN (all values missing)
                    imputation_values[col] = 0.0 if pd.isna(median_val) else median_val
                else:
                    imputation_values[col] = value
                counts[rule_name] += 1
                rule_applied = True
                break
        
        if not rule_applied:
            median_val = X_train[col].median()
            imputation_values[col] = 0.0 if pd.isna(median_val) else median_val
            other_count += 1
    
    logging.info(f"Imputation: Value={counts['value_based']}, Variation={counts['variation_based']}, "
                f"Trend={counts['trend_based']}, Other={other_count}")
    
    # Apply imputation
    for dataset_name in ['X_train', 'X_val', 'X_test']:
        dataset = datasets[dataset_name]
        original_na = dataset.isna().sum().sum()
        
        for col in dataset.columns:
            if col in imputation_values:
                dataset[col].fillna(imputation_values[col], inplace=True)
        
        final_na = dataset.isna().sum().sum()
        logging.info(f"{dataset_name}: {original_na} → {final_na} NaN values")
        
        if final_na > 0:
            na_cols = dataset.columns[dataset.isna().any()].tolist()
            logging.warning(f"Warning: {final_na} NaN values remain in {dataset_name}: {na_cols[:5]}...")
            # Fill any remaining NaN with 0
            dataset.fillna(0.0, inplace=True)
            logging.info(f"✓ Remaining NaN values filled with 0.0")
    
    datasets['imputation_values'] = pd.Series(imputation_values)
    logging.info("✓ Derived feature imputation completed")
    return datasets

def get_cache_prefix():
    """Generate cache filename prefix."""
    prefix = f"preprocessed_{CONFIG['TARGET_VARIABLE']}"
    if CONFIG['DRY_RUN']:
        prefix += f"_dryrun_{CONFIG['DRY_RUN_PATIENTS']}"
    prefix += f"_trends_{CONFIG['CALCULATE_TRENDS']}_window_{CONFIG['WINDOW_SIZE']}_gap_{CONFIG['GAP_TIME']}_seed_{CONFIG['SEED']}"
    return prefix

def save_data(datasets, label_encoders=None):
    """Save all datasets and label encoders."""
    prefix = get_cache_prefix()
    
    for name, data in datasets.items():
        with open(os.path.join(CONFIG['OUTPUT_DIR'], f'{prefix}_{name}.pkl'), 'wb') as f:
            pickle.dump(data, f)
    
    if label_encoders:
        with open(os.path.join(CONFIG['OUTPUT_DIR'], f'{prefix}_label_encoders.pkl'), 'wb') as f:
            pickle.dump(label_encoders, f)
        logging.info(f"Saved label encoders for: {list(label_encoders.keys())}")
    
    logging.info(f"Data saved with prefix: {prefix}")
    if 'outlier_bounds' in datasets:
        logging.info("Outlier bounds saved for reproducibility")

def load_cached_data():
    """Check if cached data exists."""
    prefix = get_cache_prefix()
    required_files = [f'{prefix}_{name}.pkl' for name in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']]
    return all(os.path.exists(os.path.join(CONFIG['OUTPUT_DIR'], f)) for f in required_files)

def create_splits(df_patients):
    """Create subject-level train/val/test splits."""
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
    return splits

def process_split_data(df_ts_raw, df_patients, df_demographics, feature_categories, icustay_to_subject, splits):
    """Process data for each split."""
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
    all_features = sorted(set().union(*(datasets[f'X_{name}'].columns for name in ['train', 'val', 'test'])))
    for name in ['train', 'val', 'test']:
        dataset_name = f'X_{name}'
        missing_cols = set(all_features) - set(datasets[dataset_name].columns)
        for col in missing_cols:
            datasets[dataset_name][col] = np.nan
        datasets[dataset_name] = datasets[dataset_name][all_features]
    
    return datasets

def main(config_dict=None):
    """Main preprocessing pipeline."""
    set_config(config_dict)
    setup_logging()
    start_time = time.time()
    
    # Check cache
    if CONFIG['USE_CACHED_PREPROCESSING'] and load_cached_data():
        logging.info("Using cached data - skipping preprocessing!")
        return
    
    # Load and filter data
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
    
    # Load feature classifications and process demographics
    df_classification = pd.read_csv(CONFIG['FEATURE_CLASSIFICATION_PATH'])
    feature_categories = {row['feature_name']: {'category': row['category']} for _, row in df_classification.iterrows()}
    df_demographics, label_encoders = process_demographics(df_patients)
    
    # Create splits and process data
    splits = create_splits(df_patients)
    icustay_to_subject = df_patients.index.get_level_values('subject_id').to_series(
        index=df_patients.index.get_level_values('icustay_id')
    )
    datasets = process_split_data(df_ts_raw, df_patients, df_demographics, feature_categories, icustay_to_subject, splits)
    
    # Verify no unexpected categorical features exist
    verify_no_categorical_features(datasets)
    
    # Final processing steps (no categorical encoding needed)
    datasets = handle_outliers_percentile_clipping(datasets)
    datasets = impute_derived_features(datasets)
    
    # Create scaler and normalize data
    datasets['scaler'] = StandardScaler()
    numeric_cols = [col for col in datasets['X_train'].columns 
                   if datasets['X_train'][col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    if numeric_cols:
        datasets['scaler'].fit(datasets['X_train'][numeric_cols])
        
        # Apply normalization to all datasets
        for split in ['train', 'val', 'test']:
            datasets[f'X_{split}'][numeric_cols] = datasets['scaler'].transform(datasets[f'X_{split}'][numeric_cols])
        
        logging.info(f"✓ Data normalized: {len(numeric_cols)} numeric features")
    else:
        logging.info("No numeric columns found for scaling")
    
    # Save and summarize
    save_data(datasets, label_encoders)
    
    total_time = time.time() - start_time
    logging.info(f"Preprocessing completed in {total_time/60:.2f} minutes")
    logging.info(f"Final shapes: X_train={datasets['X_train'].shape}, X_val={datasets['X_val'].shape}, X_test={datasets['X_test'].shape}")
    logging.info(f"Mortality rates: train={datasets['y_train'].mean():.3f}, val={datasets['y_val'].mean():.3f}, test={datasets['y_test'].mean():.3f}")
    if CONFIG['HANDLE_OUTLIERS']:
        logging.info(f"Outlier handling: {CONFIG['OUTLIER_PERCENTILES']} percentile clipping")

if __name__ == "__main__":
    main()