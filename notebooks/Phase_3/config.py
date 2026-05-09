# config.py
"""
Final configuration settings for Phase III text representation generation.
This version includes the corrected suffix map to handle 'weight_mean_value'.
"""
import os

# --- Core Paths ---
# Directory where the preprocessed pickle files from Phase 1 are stored.
PREPROCESSED_DATA_DIR = os.path.join('notebooks', 'Phase_1-2', 'phase_1_outputs')

# Directory where the reference ranges file is located.
REFERENCE_RANGES_PATH = os.path.join('data', 'Lab_reference_ranges.csv')

# Base directory where all serialized text files will be saved.
SERIALIZED_OUTPUT_DIR = os.path.join('notebooks', 'Phase_3', 'phase_3_serialized_data')

# --- Experiment Definitions ---

# Factor 2: Revised Prompting Strategies (P0-P5)
PROMPTS = {
    'P0': "", # Control - Null Prompt
    'P1': "The following is a summary of the first 24 hours of a patient's ICU stay. Their lab values and vitals were averaged over each hour of their first 24 hours. This data was then processed to extract features.  Generate a patient state embedding optimized for predicting in-hospital mortality from this data.",
    'P2': "The following is a summary of the first 24 hours of a patient's ICU stay. Their lab values and vitals were averaged over each hour of their first 24 hours. This data was then processed to extract features.  From the perspective of an experienced ICU physician, generate a clinical embedding that synthesizes this data to capture the patient's overall severity and primary risk factors.",
    'P3': "The following is a summary of the first 24 hours of a patient's ICU stay. Their lab values and vitals were averaged over each hour of their first 24 hours. This data was then processed to extract features.  Generate a clinical embedding that focuses on the relationships between physiological systems and the temporal evolution of key variables within this period.",
    'P4': "The following is a summary of the first 24 hours of a patient's ICU stay. Their lab values and vitals were averaged over each hour of their first 24 hours. This data was then processed to extract features.  Generate a clinical embedding that captures the patient's integrated state of physiological dysregulation. The embedding should represent the severity and dynamics of the acute illness presented in this initial window.",
    'P5': "The following is a summary of the first 24 hours of a patient's ICU stay. Their lab values and vitals were averaged over each hour of their first 24 hours. This data was then processed to extract features.  Generate a clinical embedding that is oriented around the patient's dominant pathophysiological process. Determine if the primary driver of risk appears to be cardiovascular, respiratory, septic, or metabolic in nature, and encode this assessment into the embedding."
}

# --- Feature Representation Settings ---

# Suffixes used to parse feature names (e.g., 'creatinine_last').
_MEASUREMENT_SUFFIXES = sorted([
    'mean_slope_24h', 'mean_slope_6h', 'mean_stddev', 'mean_count', 'mean_slope',
    'mean_last', 'mean_mean', 'mean_min', 'mean_max', 'mean_value', 'mean_count_6h',
    'stddev_24h', 'stddev_6h', 'slope_24h', 'slope_6h', 'count_24h', 'count_6h', 
    'max_24h', 'max_6h', 'min_24h', 'min_6h', 'stddev', 'slope', 'count', 
    'last', 'mean', 'min', 'max'
], key=len, reverse=True)

# Mapping from feature suffixes to human-readable labels.
FEATURE_LABEL_MAP = {
    'last': 'Most Recent Value',
    'mean': 'Average Value',
    'min': 'Lowest Value',
    'max': 'Highest Value',
    'stddev': 'Variability (Std Dev)',
    'count': 'Hours with Measurements',
    'slope': 'Overall Trend (Slope)',
    'min_24h': 'Lowest Value (24h)',
    'max_24h': 'Highest Value (24h)',
    'stddev_24h': 'Variability (Std Dev, 24h)',
    'count_24h': 'Measurement Count (24h)',
    'slope_24h': 'Trend (Slope, 24h)',
    'min_6h': 'Lowest Value (6h)',
    'max_6h': 'Highest Value (6h)',
    'count_6h': 'Hours with Measurements in 6h',
    'slope_6h': 'Trend (Slope, 6h)',
    'mean_mean': 'Overall 24h Average',
    'mean_last': 'Most Recent Hourly Average',
    'mean_count': 'Hourly Measurement Count',
    'mean_value': 'Mean Value',
    'mean_count_6h': 'Hours with Measurements in 6h', # FIX: Added explicit label
    'value': 'Value'
}

# Define all target variables used in the project
TARGET_VARIABLES = ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso']

# --- Cache Prefix ---
def get_cache_prefix(
    dry_run=False,
    dry_run_patients=1000,
    calculate_trends=True,
    window_size=24,
    gap_time=6,
    seed=42
):
    """
    Generates the cache filename prefix consistent with the preprocessing script.
    """
    prefix = f"preprocessed_{'_'.join(TARGET_VARIABLES)}"
    if dry_run:
        prefix += f"_dryrun_{dry_run_patients}"
    prefix += f"_trends_{calculate_trends}_window_{window_size}_gap_{gap_time}_seed_{seed}"
    return prefix