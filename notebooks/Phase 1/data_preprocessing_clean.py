# data_preprocessing_clean.py
"""
Clean Data Preprocessing Pipeline for EHR Embeddings Project

This script provides streamlined data preprocessing for medical time-series data,
using count-based missingness tracking instead of redundant binary indicators.

KEY FEATURES:
=============

1. COUNT-BASED MISSINGNESS TRACKING
   - Uses existing count features to track measurement frequency
   - feature_count = 0 → completely missing
   - feature_count = 1-5 → sparsely measured  
   - feature_count = 18-24 → frequently measured
   - More informative than binary missing indicators

2. CATEGORY-SPECIFIC FEATURE ENGINEERING
   - Static features: First value
   - Event-Driven: Count, last value, binary presence
   - High-Frequency Physiological: Mean, min, max, std, trends, count
   - Labile Lab: Mean, std, trends, binary presence, count
   - Stable Index: Last value, binary presence, count
   - Sparse Dynamic: Mean, trends, binary presence, count

3. DEMOGRAPHIC FEATURE INTEGRATION
   - Age, gender, ethnicity, insurance from patients table
   - Proper categorical encoding with train-only fitting
   - No feature leakage through subject-level splitting
   - Integrated with time-series features

4. DUAL DATASET STRATEGY
   - Main cached files: XGBoost-compatible data preserving NaNs
   - Elastic Net dataset: Fully imputed and scaled for traditional ML
   - Both use count features to capture missingness patterns

5. XGBOOST COMPATIBILITY
   - Main cache files use NaN-preserving data for XGBoost analysis
   - XGBoost script applies its own cleaning and handles missing values
   - Elastic Net data saved separately for traditional ML models
   - Compatible with notebooks/Phase 1/xgboost_analysis.py
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

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

# Default configuration values (can be overridden by passing config dict)
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

# Global configuration variables (will be set by main function)
HDF_FILE_PATH = None
FEATURE_CLASSIFICATION_PATH = None
OUTPUT_DIR = None
DRY_RUN = None
DRY_RUN_PATIENTS = None
USE_CACHED_PREPROCESSING = None
CALCULATE_TRENDS = None
WINDOW_SIZE = None
GAP_TIME = None
TARGET_VARIABLE = None
SEED = None

def set_config(config_dict=None):
    """Set global configuration from dictionary or use defaults."""
    global HDF_FILE_PATH, FEATURE_CLASSIFICATION_PATH, OUTPUT_DIR
    global DRY_RUN, DRY_RUN_PATIENTS, USE_CACHED_PREPROCESSING, CALCULATE_TRENDS
    global WINDOW_SIZE, GAP_TIME, TARGET_VARIABLE, SEED
    
    # Use provided config or defaults
    config = config_dict if config_dict else DEFAULT_CONFIG
    
    # Set global variables
    HDF_FILE_PATH = config.get('HDF_FILE_PATH', DEFAULT_CONFIG['HDF_FILE_PATH'])
    FEATURE_CLASSIFICATION_PATH = config.get('FEATURE_CLASSIFICATION_PATH', DEFAULT_CONFIG['FEATURE_CLASSIFICATION_PATH'])
    OUTPUT_DIR = config.get('OUTPUT_DIR', DEFAULT_CONFIG['OUTPUT_DIR'])
    DRY_RUN = config.get('DRY_RUN', DEFAULT_CONFIG['DRY_RUN'])
    DRY_RUN_PATIENTS = config.get('DRY_RUN_PATIENTS', DEFAULT_CONFIG['DRY_RUN_PATIENTS'])
    USE_CACHED_PREPROCESSING = config.get('USE_CACHED_PREPROCESSING', DEFAULT_CONFIG['USE_CACHED_PREPROCESSING'])
    CALCULATE_TRENDS = config.get('CALCULATE_TRENDS', DEFAULT_CONFIG['CALCULATE_TRENDS'])
    WINDOW_SIZE = config.get('WINDOW_SIZE', DEFAULT_CONFIG['WINDOW_SIZE'])
    GAP_TIME = config.get('GAP_TIME', DEFAULT_CONFIG['GAP_TIME'])
    TARGET_VARIABLE = config.get('TARGET_VARIABLE', DEFAULT_CONFIG['TARGET_VARIABLE'])
    SEED = config.get('SEED', DEFAULT_CONFIG['SEED'])
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Set up logging after configuration is initialized."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'preprocessing_log.txt'), mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Output directory: {OUTPUT_DIR}")

def load_feature_classification():
    """Load feature categories from the CSV file."""
    try:
        df_classification = pd.read_csv(FEATURE_CLASSIFICATION_PATH)
        feature_categories = {}
        for _, row in df_classification.iterrows():
            feature_categories[row['feature_name']] = {
                'category': row['category'],
                'category_num': row['category_num']
            }
        logging.info(f"✓ Loaded feature classifications for {len(feature_categories)} features")
        return feature_categories
    except Exception as e:
        logging.error(f"Error loading feature classification: {e}")
        raise

def load_data(hdf_path, n_samples=None):
    """Loads patient and time-series data from the MIMIC-Extract HDF5 store."""
    logging.info(f"Starting data loading from: {hdf_path}")
    
    with pd.HDFStore(hdf_path, 'r') as store:
        df_patients = store['/patients']
        
        if n_samples:
            logging.warning(f"--- DRY RUN MODE: Sampling {n_samples} patients ---")
            all_patient_ids = df_patients.index.get_level_values('icustay_id').unique()
            np.random.seed(SEED)
            sampled_ids = np.random.choice(all_patient_ids, n_samples, replace=False)
            df_patients = df_patients[df_patients.index.get_level_values('icustay_id').isin(sampled_ids)]
            
            df_ts = store['/vitals_labs_mean']
            df_ts = df_ts[df_ts.index.get_level_values('icustay_id').isin(sampled_ids)]
        else:
            df_ts = store['/vitals_labs_mean']
        
        # Handle MultiIndex columns
        if isinstance(df_ts.columns, pd.MultiIndex):
            new_columns = ['_'.join(col).strip() for col in df_ts.columns.values]
            df_ts.columns = pd.Index(new_columns)
        
        df_ts = df_ts.loc[:,~df_ts.columns.duplicated()]
        logging.info(f"✓ Data loaded: patients={df_patients.shape}, time-series={df_ts.shape}")

    return df_patients, df_ts

def extract_demographic_features(df_patients):
    """Extract and process demographic features from patients table."""
    logging.info("Extracting demographic features...")
    
    # Get unique patients (one row per subject_id)
    df_demographics = df_patients.groupby(level='subject_id').first().copy()
    
    # Select demographic columns
    demographic_cols = ['age', 'gender', 'ethnicity', 'insurance']
    available_cols = [col for col in demographic_cols if col in df_demographics.columns]
    missing_cols = [col for col in demographic_cols if col not in df_demographics.columns]
    
    if missing_cols:
        logging.warning(f"⚠️  Missing demographic columns: {missing_cols}")
    
    df_demographics = df_demographics[available_cols]
    
    # Handle missing values
    for col in df_demographics.columns:
        if df_demographics[col].isna().any():
            if df_demographics[col].dtype == 'object':
                # For categorical, use most frequent value
                fill_value = df_demographics[col].mode().iloc[0] if not df_demographics[col].mode().empty else 'Unknown'
            else:
                # For numeric, use median
                fill_value = df_demographics[col].median()
            df_demographics[col] = df_demographics[col].fillna(fill_value)
            logging.info(f"  - Filled {df_demographics[col].isna().sum()} missing values in {col}")
    
    # Create age categories for better modeling
    df_demographics['age_category'] = pd.cut(
        df_demographics['age'], 
        bins=[0, 30, 50, 70, 90, 120], 
        labels=['18-30', '31-50', '51-70', '71-90', '90+'],
        include_lowest=True
    )
    
    # Create simplified ethnicity categories
    if 'ethnicity' in df_demographics.columns:
        ethnicity_mapping = {
            'WHITE': 'WHITE',
            'BLACK/AFRICAN AMERICAN': 'BLACK',
            'HISPANIC/LATINO': 'HISPANIC',
            'ASIAN': 'ASIAN',
            'UNKNOWN/NOT SPECIFIED': 'UNKNOWN',
            'OTHER': 'OTHER'
        }
        df_demographics['ethnicity_simplified'] = df_demographics['ethnicity'].map(
            lambda x: ethnicity_mapping.get(x, 'OTHER')
        )
    
    # Create simplified insurance categories
    if 'insurance' in df_demographics.columns:
        insurance_mapping = {
            'Medicare': 'MEDICARE',
            'Medicaid': 'MEDICAID',
            'Private': 'PRIVATE',
            'Government': 'GOVERNMENT',
            'Self Pay': 'SELF_PAY'
        }
        df_demographics['insurance_simplified'] = df_demographics['insurance'].map(
            lambda x: insurance_mapping.get(x, 'OTHER')
        )
    
    # Ensure all categorical columns are properly handled
    categorical_cols = ['gender', 'ethnicity_simplified', 'insurance_simplified', 'age_category']
    for col in categorical_cols:
        if col in df_demographics.columns:
            # Convert to string to ensure consistent handling
            df_demographics[col] = df_demographics[col].astype(str)
    
    logging.info(f"✓ Extracted demographic features: {df_demographics.shape}")
    logging.info(f"  - Age range: {df_demographics['age'].min():.1f} - {df_demographics['age'].max():.1f}")
    if 'gender' in df_demographics.columns:
        logging.info(f"  - Gender distribution: {df_demographics['gender'].value_counts().to_dict()}")
    
    return df_demographics

def encode_categorical_features(df_demographics, train_subjects=None):
    """Encode categorical demographic features, fitting only on training data to prevent leakage."""
    logging.info("Encoding categorical demographic features...")
    
    df_encoded = df_demographics.copy()
    label_encoders = {}
    
    # Categorical columns to encode
    categorical_cols = ['gender', 'ethnicity_simplified', 'insurance_simplified', 'age_category']
    available_categorical = [col for col in categorical_cols if col in df_encoded.columns]
    
    for col in available_categorical:
        if col in df_encoded.columns:
            le = LabelEncoder()
            
            # Ensure the column contains only string values
            df_encoded[col] = df_encoded[col].astype(str)
            
            if train_subjects is not None:
                # Fit only on training data to prevent leakage
                train_mask = df_encoded.index.isin(train_subjects)
                train_values = df_encoded.loc[train_mask, col]
                le.fit(train_values)
                logging.info(f"  - Fitted {col} encoder on {len(train_values)} training samples")
                logging.info(f"    - Unique values: {list(le.classes_)}")
            else:
                # Fit on all data (for initial processing)
                le.fit(df_encoded[col])
                logging.info(f"  - Fitted {col} encoder on all {len(df_encoded[col])} samples")
                logging.info(f"    - Unique values: {list(le.classes_)}")
            
            # Transform all data
            df_encoded[f'{col}_encoded'] = le.transform(df_encoded[col])
            label_encoders[col] = le
            
            # Drop original categorical column
            df_encoded = df_encoded.drop(columns=[col])
    
    logging.info(f"✓ Encoded {len(available_categorical)} categorical features")
    return df_encoded, label_encoders

def _calculate_slope(series):
    """Helper to calculate slope of a time series, handling NaNs."""
    y = series.dropna()
    if len(y) < 2:
        return np.nan
    
    if isinstance(y.index, pd.MultiIndex):
        x = y.index.get_level_values('hours_in').values
    else:
        x = y.index.values
    
    try:
        return linregress(x, y.values).slope
    except Exception:
        return np.nan

def engineer_features_by_category(df_ts, feature_categories):
    """Engineers features using count-based missingness tracking."""
    logging.info(f"Starting feature engineering for {df_ts.index.get_level_values('icustay_id').nunique()} stays...")
    
    df_ts_sorted = df_ts.sort_index()
    grouped = df_ts_sorted.groupby('icustay_id')
    all_features = {}
    
    available_features = [col for col in df_ts_sorted.columns if col in feature_categories]
    missing_features = [col for col in df_ts_sorted.columns if col not in feature_categories]
    
    if missing_features:
        logging.error(f"❌ Found {len(missing_features)} features not in classification file:")
        for feature in missing_features[:10]:  # Show first 10
            logging.error(f"  - {feature}")
        if len(missing_features) > 10:
            logging.error(f"  ... and {len(missing_features) - 10} more")
        raise ValueError(f"All {len(df_ts_sorted.columns)} features must be categorized. Missing {len(missing_features)} classifications.")
    
    logging.info(f"✓ Processing {len(available_features)} classified features...")
    
    # Group features by category
    features_by_category = {}
    for feature in available_features:
        category = feature_categories[feature]['category']
        if category not in features_by_category:
            features_by_category[category] = []
        features_by_category[category].append(feature)
    
    # Process each category
    for category, features in features_by_category.items():
        logging.info(f"Processing {len(features)} features in category: {category}")
        
        if category == 'Static':
            for feature in features:
                all_features[f'{feature}_value'] = grouped[feature].first()
        
        elif category == 'Event-Driven':
            for feature in features:
                feature_grouped = grouped[feature]
                all_features[f'{feature}_count'] = feature_grouped.count()
                all_features[f'{feature}_last'] = feature_grouped.last()
        
        elif category == 'High-Frequency Physiological':
            for feature in features:
                feature_grouped = grouped[feature]
                all_features[f'{feature}_last'] = feature_grouped.last()
                all_features[f'{feature}_mean'] = feature_grouped.mean()
                all_features[f'{feature}_min_24h'] = feature_grouped.min()
                all_features[f'{feature}_max_24h'] = feature_grouped.max()
                all_features[f'{feature}_stddev_24h'] = feature_grouped.std()
                all_features[f'{feature}_count'] = feature_grouped.count()  # KEY: Count-based missingness (24h)
                
                # Add 6-hour count for recent measurement frequency (clinically relevant for high-freq data)
                last_6h_data = df_ts_sorted[df_ts_sorted.index.get_level_values('hours_in') >= 18][feature]
                if not last_6h_data.empty:
                    all_features[f'{feature}_count_6h'] = last_6h_data.groupby('icustay_id').count()
                else:
                    all_features[f'{feature}_count_6h'] = pd.Series(0, index=grouped.groups.keys())
                
                if CALCULATE_TRENDS:
                    all_features[f'{feature}_slope_24h'] = feature_grouped.apply(_calculate_slope)
                    if not last_6h_data.empty:
                        all_features[f'{feature}_slope_6h'] = last_6h_data.groupby('icustay_id').apply(_calculate_slope)
                    else:
                        all_features[f'{feature}_slope_6h'] = pd.Series(np.nan, index=grouped.groups.keys())
        
        elif category == 'Labile Lab':
            for feature in features:
                feature_grouped = grouped[feature]
                all_features[f'{feature}_last'] = feature_grouped.last()
                all_features[f'{feature}_mean'] = feature_grouped.mean()
                all_features[f'{feature}_stddev_24h'] = feature_grouped.std()
                all_features[f'{feature}_count'] = feature_grouped.count()  # KEY: Count-based missingness
                
                if CALCULATE_TRENDS:
                    all_features[f'{feature}_slope_24h'] = feature_grouped.apply(_calculate_slope)
        
        elif category == 'Stable Index':
            for feature in features:
                feature_grouped = grouped[feature]
                all_features[f'{feature}_last'] = feature_grouped.last()
                all_features[f'{feature}_count'] = feature_grouped.count()  # KEY: Count-based missingness
        
        elif category == 'Sparse Dynamic':
            for feature in features:
                feature_grouped = grouped[feature]
                all_features[f'{feature}_last'] = feature_grouped.last()
                all_features[f'{feature}_mean'] = feature_grouped.mean()
                all_features[f'{feature}_count'] = feature_grouped.count()  # KEY: Count-based missingness
                
                if CALCULATE_TRENDS:
                    all_features[f'{feature}_slope_24h'] = feature_grouped.apply(_calculate_slope)
        
        else:
            # All features should be categorized - this indicates missing classifications
            uncategorized_features = [f for f in features if f not in feature_categories]
            if uncategorized_features:
                logging.error(f"❌ Uncategorized features found: {uncategorized_features}")
                logging.error("All features must be categorized in the feature classification CSV file.")
                raise ValueError(f"Found {len(uncategorized_features)} uncategorized features. Please update the feature classification file.")
            else:
                logging.warning(f"⚠️  Features in unknown category: {features}")
                # Apply default feature engineering for unknown categories
                for feature in features:
                    feature_grouped = grouped[feature]
                    all_features[f'{feature}_mean'] = feature_grouped.mean()
                    all_features[f'{feature}_last'] = feature_grouped.last()
                    all_features[f'{feature}_count'] = feature_grouped.count()
    
    features_df = pd.DataFrame(all_features)
    logging.info(f"✓ Feature engineering complete. Shape: {features_df.shape}")
    
    # Count features that end with "_count" (true count-based missingness features)
    count_features = [col for col in features_df.columns if col.endswith('_count')]
    logging.info(f"✓ Count-based missingness tracking: {len(count_features)} features")
    
    return create_model_specific_datasets(features_df)

def create_model_specific_datasets(df_features):
    """Create optimized datasets for different model types."""
    logging.info("Creating model-specific datasets...")
    
    # Elastic Net dataset (fully imputed)
    df_elastic = df_features.copy()
    for col in df_elastic.columns:
        if df_elastic[col].isna().any():
            fill_value = df_elastic[col].median() if df_elastic[col].dtype != 'object' else 'Unknown'
            df_elastic[col] = df_elastic[col].fillna(fill_value)
    
    # XGBoost dataset (minimal imputation)
    df_xgboost = df_features.copy()
    # Only impute extremely sparse features (>95% missing)
    for col in df_xgboost.columns:
        if df_xgboost[col].isna().mean() > 0.95:
            fill_value = df_xgboost[col].median() if df_xgboost[col].dtype != 'object' else 0
            df_xgboost[col] = df_xgboost[col].fillna(fill_value)
    
    logging.info(f"✓ Elastic Net dataset: {df_elastic.shape} (fully imputed)")
    logging.info(f"✓ XGBoost dataset: {df_xgboost.shape} (preserves NaNs)")
    
    return {'elastic_net': df_elastic, 'xgboost': df_xgboost}

def merge_demographic_features(df_features, df_demographics, icustay_to_subject_mapping):
    """Merge demographic features with engineered features using ICU stay to subject mapping."""
    logging.info("Merging demographic features with engineered features...")
    
    # Create mapping from icustay_id to demographic features
    df_features_with_demo = df_features.copy()
    
    # Add subject_id to features dataframe
    df_features_with_demo['subject_id'] = df_features_with_demo.index.map(icustay_to_subject_mapping)
    # Remove any rows where mapping failed (shouldn't happen, but just in case)
    df_features_with_demo = df_features_with_demo.dropna(subset=['subject_id'])
    
    # Merge with demographics
    df_features_with_demo = df_features_with_demo.merge(
        df_demographics, 
        left_on='subject_id', 
        right_index=True, 
        how='left'
    )
    
    # Drop the subject_id column (keep only demographic features)
    df_features_with_demo = df_features_with_demo.drop(columns=['subject_id'])
    
    # Check for any missing demographic data
    demo_cols = [col for col in df_demographics.columns if col in df_features_with_demo.columns]
    missing_demo = df_features_with_demo[demo_cols].isna().sum()
    if missing_demo.sum() > 0:
        logging.warning(f"⚠️  Missing demographic data: {missing_demo[missing_demo > 0].to_dict()}")
    
    logging.info(f"✓ Merged demographic features. Final shape: {df_features_with_demo.shape}")
    logging.info(f"  - Added {len(demo_cols)} demographic features")
    
    return df_features_with_demo

def get_cache_filename_prefix():
    """Generate cache filename prefix that matches xgboost_analysis.py expectations."""
    prefix = f"preprocessed_{TARGET_VARIABLE}"
    if DRY_RUN:
        prefix += f"_dryrun_{DRY_RUN_PATIENTS}"
    prefix += f"_trends_{CALCULATE_TRENDS}"
    prefix += f"_window_{WINDOW_SIZE}_gap_{GAP_TIME}"
    prefix += f"_seed_{SEED}"
    prefix += "_with_demographics"  # Add flag for demographic features
    return prefix

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputation_values,
                          X_train_unnormalized=None, X_val_unnormalized=None, X_test_unnormalized=None,
                          label_encoders=None):
    """Save preprocessed data to cache files."""
    prefix = get_cache_filename_prefix()
    
    cache_files = {
        'X_train': os.path.join(OUTPUT_DIR, f'{prefix}_X_train.pkl'),
        'X_val': os.path.join(OUTPUT_DIR, f'{prefix}_X_val.pkl'),
        'X_test': os.path.join(OUTPUT_DIR, f'{prefix}_X_test.pkl'),
        'y_train': os.path.join(OUTPUT_DIR, f'{prefix}_y_train.pkl'),
        'y_val': os.path.join(OUTPUT_DIR, f'{prefix}_y_val.pkl'),
        'y_test': os.path.join(OUTPUT_DIR, f'{prefix}_y_test.pkl'),
        'scaler': os.path.join(OUTPUT_DIR, f'{prefix}_scaler.pkl'),
        'imputation_values': os.path.join(OUTPUT_DIR, f'{prefix}_imputation_values.pkl')
    }
    
    # Save all files
    with open(cache_files['X_train'], 'wb') as f: pickle.dump(X_train, f)
    with open(cache_files['X_val'], 'wb') as f: pickle.dump(X_val, f)
    with open(cache_files['X_test'], 'wb') as f: pickle.dump(X_test, f)
    with open(cache_files['y_train'], 'wb') as f: pickle.dump(y_train, f)
    with open(cache_files['y_val'], 'wb') as f: pickle.dump(y_val, f)
    with open(cache_files['y_test'], 'wb') as f: pickle.dump(y_test, f)
    with open(cache_files['scaler'], 'wb') as f: pickle.dump(scaler, f)
    with open(cache_files['imputation_values'], 'wb') as f: pickle.dump(imputation_values, f)
    
    # Save label encoders if provided
    if label_encoders:
        with open(os.path.join(OUTPUT_DIR, f'{prefix}_label_encoders.pkl'), 'wb') as f:
            pickle.dump(label_encoders, f)
    
    # Save unnormalized data if provided
    if X_train_unnormalized is not None:
        with open(os.path.join(OUTPUT_DIR, f'{prefix}_X_train_unnormalized.pkl'), 'wb') as f:
            pickle.dump(X_train_unnormalized, f)
        with open(os.path.join(OUTPUT_DIR, f'{prefix}_X_val_unnormalized.pkl'), 'wb') as f:
            pickle.dump(X_val_unnormalized, f)
        with open(os.path.join(OUTPUT_DIR, f'{prefix}_X_test_unnormalized.pkl'), 'wb') as f:
            pickle.dump(X_test_unnormalized, f)
    
    logging.info(f"✓ Data cached with prefix: {prefix}")
    return cache_files

def load_preprocessed_data():
    """Load preprocessed data from cache files if they exist."""
    prefix = get_cache_filename_prefix()
    
    cache_files = {
        'X_train': os.path.join(OUTPUT_DIR, f'{prefix}_X_train.pkl'),
        'X_val': os.path.join(OUTPUT_DIR, f'{prefix}_X_val.pkl'),
        'X_test': os.path.join(OUTPUT_DIR, f'{prefix}_X_test.pkl'),
        'y_train': os.path.join(OUTPUT_DIR, f'{prefix}_y_train.pkl'),
        'y_val': os.path.join(OUTPUT_DIR, f'{prefix}_y_val.pkl'),
        'y_test': os.path.join(OUTPUT_DIR, f'{prefix}_y_test.pkl'),
        'scaler': os.path.join(OUTPUT_DIR, f'{prefix}_scaler.pkl'),
        'imputation_values': os.path.join(OUTPUT_DIR, f'{prefix}_imputation_values.pkl')
    }
    
    if not all(os.path.exists(f) for f in cache_files.values()):
        return None
    
    try:
        with open(cache_files['X_train'], 'rb') as f: X_train = pickle.load(f)
        with open(cache_files['X_val'], 'rb') as f: X_val = pickle.load(f)
        with open(cache_files['X_test'], 'rb') as f: X_test = pickle.load(f)
        with open(cache_files['y_train'], 'rb') as f: y_train = pickle.load(f)
        with open(cache_files['y_val'], 'rb') as f: y_val = pickle.load(f)
        with open(cache_files['y_test'], 'rb') as f: y_test = pickle.load(f)
        with open(cache_files['scaler'], 'rb') as f: scaler = pickle.load(f)
        with open(cache_files['imputation_values'], 'rb') as f: imputation_values = pickle.load(f)
        
        logging.info(f"✓ Loaded cached data: X_train={X_train.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputation_values
    except Exception as e:
        logging.warning(f"Failed to load cached data: {e}")
        return None

def main(config_dict=None):
    """Main preprocessing function."""
    # Set configuration first
    set_config(config_dict)
    
    # Set up logging after configuration is initialized
    setup_logging()
    
    start_time = time.time()
    
    # Check for cached data
    if USE_CACHED_PREPROCESSING:
        cached_data = load_preprocessed_data()
        if cached_data is not None:
            logging.info("✓ Using cached data - skipping preprocessing!")
            return
    
    # Load data
    df_patients, df_ts_raw = load_data(HDF_FILE_PATH, n_samples=DRY_RUN_PATIENTS if DRY_RUN else None)
    
    # Debug: Check the structure of df_patients
    logging.info(f"✓ Patients DataFrame structure:")
    logging.info(f"  - Index levels: {df_patients.index.names}")
    logging.info(f"  - Columns: {list(df_patients.columns)}")
    logging.info(f"  - Shape: {df_patients.shape}")
    
    # Extract demographic features
    df_demographics = extract_demographic_features(df_patients)
    
    # Apply time window filtering
    patient_max_hours = df_ts_raw.groupby(level='icustay_id').apply(
        lambda x: x.index.get_level_values('hours_in').max()
    )
    valid_patients = patient_max_hours[patient_max_hours > (WINDOW_SIZE + GAP_TIME)]
    
    df_patients = df_patients[df_patients.index.get_level_values('icustay_id').isin(valid_patients.index)]
    df_ts_raw = df_ts_raw[
        (df_ts_raw.index.get_level_values('icustay_id').isin(valid_patients.index)) &
        (df_ts_raw.index.get_level_values('hours_in') < WINDOW_SIZE)
    ]
    
    logging.info(f"✓ Time window filtering: {len(valid_patients)} patients, {df_ts_raw.shape} time-series")
    
    # Load feature classifications
    feature_categories = load_feature_classification()
    
    # Split data by subjects (no leakage)
    subject_outcomes = df_patients.groupby('subject_id')[TARGET_VARIABLE].max()
    subjects, outcomes = subject_outcomes.index.values, subject_outcomes.values
    
    train_val_subjects, test_subjects, train_val_outcomes, test_outcomes = train_test_split(
        subjects, outcomes, test_size=0.25, random_state=SEED, stratify=outcomes
    )
    train_subjects, val_subjects, train_outcomes, val_outcomes = train_test_split(
        train_val_subjects, train_val_outcomes, test_size=0.125, random_state=SEED, stratify=train_val_outcomes
    )
    
    # Create ICU stay masks
    train_icustay_ids = df_patients[df_patients.index.get_level_values('subject_id').isin(train_subjects)].index.get_level_values('icustay_id').unique()
    val_icustay_ids = df_patients[df_patients.index.get_level_values('subject_id').isin(val_subjects)].index.get_level_values('icustay_id').unique()
    test_icustay_ids = df_patients[df_patients.index.get_level_values('subject_id').isin(test_subjects)].index.get_level_values('icustay_id').unique()
    
    logging.info(f"✓ Train: {len(train_subjects)} subjects, {len(train_icustay_ids)} stays")
    logging.info(f"✓ Val: {len(val_subjects)} subjects, {len(val_icustay_ids)} stays")  
    logging.info(f"✓ Test: {len(test_subjects)} subjects, {len(test_icustay_ids)} stays")
    
    # Encode categorical features (fit only on training data to prevent leakage)
    df_demographics_encoded, label_encoders = encode_categorical_features(df_demographics, train_subjects)
    
    # Create ICU stay to subject mapping
    icustay_to_subject = df_patients.index.get_level_values('subject_id').to_series(index=df_patients.index.get_level_values('icustay_id'))
    logging.info(f"✓ Created ICU stay to subject mapping: {len(icustay_to_subject)} mappings")
    logging.info(f"  - Sample mappings: {dict(list(icustay_to_subject.head().items()))}")
    
    # Feature engineering for each split
    train_ts_mask = df_ts_raw.index.get_level_values('icustay_id').isin(train_icustay_ids)
    val_ts_mask = df_ts_raw.index.get_level_values('icustay_id').isin(val_icustay_ids)
    test_ts_mask = df_ts_raw.index.get_level_values('icustay_id').isin(test_icustay_ids)
    
    train_datasets = engineer_features_by_category(df_ts_raw[train_ts_mask], feature_categories)
    val_datasets = engineer_features_by_category(df_ts_raw[val_ts_mask], feature_categories)
    test_datasets = engineer_features_by_category(df_ts_raw[test_ts_mask], feature_categories)
    
    # Merge demographic features with each split
    train_datasets['elastic_net'] = merge_demographic_features(
        train_datasets['elastic_net'], df_demographics_encoded, icustay_to_subject
    )
    train_datasets['xgboost'] = merge_demographic_features(
        train_datasets['xgboost'], df_demographics_encoded, icustay_to_subject
    )
    
    val_datasets['elastic_net'] = merge_demographic_features(
        val_datasets['elastic_net'], df_demographics_encoded, icustay_to_subject
    )
    val_datasets['xgboost'] = merge_demographic_features(
        val_datasets['xgboost'], df_demographics_encoded, icustay_to_subject
    )
    
    test_datasets['elastic_net'] = merge_demographic_features(
        test_datasets['elastic_net'], df_demographics_encoded, icustay_to_subject
    )
    test_datasets['xgboost'] = merge_demographic_features(
        test_datasets['xgboost'], df_demographics_encoded, icustay_to_subject
    )
    
    # Extract Elastic Net datasets (fully imputed)
    X_train_elastic = train_datasets['elastic_net']
    X_val_elastic = val_datasets['elastic_net']
    X_test_elastic = test_datasets['elastic_net']
    
    # Debug: Check data types
    logging.info(f"✓ Elastic Net dataset data types:")
    # Use a more compatible approach for older pandas versions
    numeric_cols = []
    object_cols = []
    for col in X_train_elastic.columns:
        if X_train_elastic[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
        elif X_train_elastic[col].dtype == 'object':
            object_cols.append(col)
    
    logging.info(f"  - Numeric columns: {len(numeric_cols)}")
    logging.info(f"  - Object columns: {len(object_cols)}")
    if len(object_cols) > 0:
        logging.info(f"  - Object column names: {object_cols}")
        # Show sample values from first object column
        first_obj_col = object_cols[0]
        logging.info(f"  - Sample values from {first_obj_col}: {X_train_elastic[first_obj_col].value_counts().head().to_dict()}")
    
    # Extract XGBoost datasets (preserves NaNs)
    X_train_xgb = train_datasets['xgboost']
    X_val_xgb = val_datasets['xgboost']
    X_test_xgb = test_datasets['xgboost']
    
    # Align columns across splits
    all_cols_elastic = X_train_elastic.columns.union(X_val_elastic.columns).union(X_test_elastic.columns)
    X_train_elastic = X_train_elastic.reindex(columns=all_cols_elastic, fill_value=0)
    X_val_elastic = X_val_elastic.reindex(columns=all_cols_elastic, fill_value=0)
    X_test_elastic = X_test_elastic.reindex(columns=all_cols_elastic, fill_value=0)
    
    all_cols_xgb = X_train_xgb.columns.union(X_val_xgb.columns).union(X_test_xgb.columns)
    X_train_xgb = X_train_xgb.reindex(columns=all_cols_xgb, fill_value=np.nan)
    X_val_xgb = X_val_xgb.reindex(columns=all_cols_xgb, fill_value=np.nan)
    X_test_xgb = X_test_xgb.reindex(columns=all_cols_xgb, fill_value=np.nan)
    
    # Save unnormalized data for embeddings (using elastic net data which is fully imputed)
    X_train_unnormalized = X_train_elastic.copy()
    X_val_unnormalized = X_val_elastic.copy()
    X_test_unnormalized = X_test_elastic.copy()
    
    # For XGBoost compatibility, use the NaN-preserving datasets as the main cached data
    # XGBoost analysis script expects to handle NaNs itself
    X_train = X_train_xgb.copy()
    X_val = X_val_xgb.copy() 
    X_test = X_test_xgb.copy()
    
    # Scale features for traditional ML (elastic net version)
    # Only scale numeric features, exclude any remaining categorical features
    # Use the same approach as above for compatibility
    numeric_cols_for_scaling = []
    categorical_cols_for_scaling = []
    for col in X_train_elastic.columns:
        if X_train_elastic[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols_for_scaling.append(col)
        elif X_train_elastic[col].dtype == 'object':
            categorical_cols_for_scaling.append(col)
    
    if len(categorical_cols_for_scaling) > 0:
        logging.warning(f"⚠️  Found {len(categorical_cols_for_scaling)} categorical columns that will not be scaled: {categorical_cols_for_scaling}")
    
    scaler = StandardScaler()
    X_train_elastic_scaled = X_train_elastic.copy()
    X_val_elastic_scaled = X_val_elastic.copy()
    X_test_elastic_scaled = X_test_elastic.copy()
    
    # Scale only numeric features
    if len(numeric_cols_for_scaling) > 0:
        X_train_elastic_scaled[numeric_cols_for_scaling] = scaler.fit_transform(X_train_elastic[numeric_cols_for_scaling])
        X_val_elastic_scaled[numeric_cols_for_scaling] = scaler.transform(X_val_elastic[numeric_cols_for_scaling])
        X_test_elastic_scaled[numeric_cols_for_scaling] = scaler.transform(X_test_elastic[numeric_cols_for_scaling])
        logging.info(f"✓ Scaled {len(numeric_cols_for_scaling)} numeric features")
    else:
        logging.warning("⚠️  No numeric features found for scaling")
    
    # Prepare target variables
    y_train = df_patients.loc[df_patients.index.get_level_values('icustay_id').isin(X_train.index), TARGET_VARIABLE]
    y_train = y_train.groupby('icustay_id').first().reindex(X_train.index)
    
    y_val = df_patients.loc[df_patients.index.get_level_values('icustay_id').isin(X_val.index), TARGET_VARIABLE]
    y_val = y_val.groupby('icustay_id').first().reindex(X_val.index)
    
    y_test = df_patients.loc[df_patients.index.get_level_values('icustay_id').isin(X_test.index), TARGET_VARIABLE]
    y_test = y_test.groupby('icustay_id').first().reindex(X_test.index)
    
    # Verify alignment
    logging.info(f"✓ Target alignment: train={X_train.index.equals(y_train.index)}, val={X_val.index.equals(y_val.index)}, test={X_test.index.equals(y_test.index)}")
    logging.info(f"✓ Mortality rates: train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")
    
    # Create imputation values record
    imputation_values = X_train_elastic.median()
    
    # Save data (main cache files use XGBoost-compatible data with NaNs)
    save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputation_values,
                          X_train_unnormalized, X_val_unnormalized, X_test_unnormalized, label_encoders)
    
    # Save Elastic Net scaled datasets separately (for traditional ML models)
    prefix = get_cache_filename_prefix()
    with open(os.path.join(OUTPUT_DIR, f'{prefix}_X_train_elastic_scaled.pkl'), 'wb') as f:
        pickle.dump(X_train_elastic_scaled, f)
    with open(os.path.join(OUTPUT_DIR, f'{prefix}_X_val_elastic_scaled.pkl'), 'wb') as f:
        pickle.dump(X_val_elastic_scaled, f)
    with open(os.path.join(OUTPUT_DIR, f'{prefix}_X_test_elastic_scaled.pkl'), 'wb') as f:
        pickle.dump(X_test_elastic_scaled, f)
    
    # Save CSV files for easy access
    X_train_unnormalized.to_csv(os.path.join(OUTPUT_DIR, 'X_train_elastic_unnormalized.csv'))
    X_val_unnormalized.to_csv(os.path.join(OUTPUT_DIR, 'X_val_elastic_unnormalized.csv'))
    X_test_unnormalized.to_csv(os.path.join(OUTPUT_DIR, 'X_test_elastic_unnormalized.csv'))
    
    X_train_elastic_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_train_elastic_scaled.csv'))
    X_val_elastic_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_val_elastic_scaled.csv'))
    X_test_elastic_scaled.to_csv(os.path.join(OUTPUT_DIR, 'X_test_elastic_scaled.csv'))
    
    X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train_xgboost.csv'))
    X_val.to_csv(os.path.join(OUTPUT_DIR, 'X_val_xgboost.csv'))
    X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test_xgboost.csv'))
    
    y_train.to_csv(os.path.join(OUTPUT_DIR, 'y_train.csv'))
    y_val.to_csv(os.path.join(OUTPUT_DIR, 'y_val.csv'))
    y_test.to_csv(os.path.join(OUTPUT_DIR, 'y_test.csv'))
    
    # Count different types of features
    count_features = [col for col in X_train.columns if col.endswith('_count') or col.endswith('_count_6h')]
    demographic_features = [col for col in X_train.columns if col in df_demographics_encoded.columns]
    time_series_features = [col for col in X_train.columns if col not in demographic_features]
    
    logging.info(f"✓ All datasets saved successfully")
    logging.info(f"✓ Main cached data (XGBoost-compatible with NaNs): X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    logging.info(f"✓ Elastic Net scaled data shapes: X_train={X_train_elastic_scaled.shape}, X_val={X_val_elastic_scaled.shape}, X_test={X_test_elastic_scaled.shape}")
    logging.info(f"✓ Count-based missingness tracking: {len(count_features)} features (including 6h counts for high-freq data)")
    logging.info(f"✓ Demographic features: {len(demographic_features)} features")
    logging.info(f"✓ Time-series features: {len(time_series_features)} features")
    logging.info(f"✓ Main cache files compatible with xgboost_analysis.py script")
    
    total_time = time.time() - start_time
    logging.info(f"--- Clean preprocessing completed in {total_time/60:.2f} minutes ---")

if __name__ == "__main__":
    main()
