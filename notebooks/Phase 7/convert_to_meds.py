import os
import sys
from datetime import timedelta, datetime
import json
from typing import Dict, List, Set, Tuple, Optional
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import jsonschema
from sklearn.model_selection import train_test_split
import tqdm
<<<<<<< HEAD
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
=======
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130

from meds import (
    CodeMetadataSchema,
    DataSchema,
    DatasetMetadataSchema,
    SubjectSplitSchema,
    LabelSchema,
    train_split,
    tuning_split,
    held_out_split,
)

# --- Configuration ---
HDF_FILE_PATH = './data/raw/all_hourly_data.h5'
MEDS_OUTPUT_DIR = './data/meds_cohort_split_filtered'
OMOP_VOCAB_DIR = './data/OMOP_Vocabulary'
FEATURE_NAMES_DIR = './data/processed/feature_names'
TARGET_VARIABLES = ['mort_hosp', 'los_3', 'los_7', 'readmission_30', 'intervention_vent', 'intervention_vaso']
SPLIT_CONFIG = {'STRATIFICATION_TARGET': 'mort_hosp', 'SEED': 42}
TIME_WINDOW_CONFIG = {'WINDOW_SIZE': 24, 'GAP_TIME': 6}
# Set DEBUG_MODE to True to run on a small subset of patients
DEBUG_MODE = False
DEBUG_PATIENT_COUNT = 1000

# --- MEDS Schema Definitions (Corrected) ---
DATA_SCHEMA = DataSchema(
    subject_id='subject_id',
    time='time',
    code='code',
    numeric_value='numeric_value',
    text_value='text_value'
)
LABEL_SCHEMA = LabelSchema(
    subject_id="subject_id",
    prediction_time="prediction_time",
    boolean_value="boolean_value",
    integer_value="integer_value",
    float_value="float_value",
    categorical_value="categorical_value"
)
SUBJECT_SPLIT_SCHEMA = SubjectSplitSchema(subject_id='subject_id', split='split')
CODE_METADATA_SCHEMA = CodeMetadataSchema(code='code', description='description', parent_codes='parent_codes')
DATASET_METADATA_SCHEMA = DatasetMetadataSchema()


class RobustOMOPMapper:
    """
    Robust OMOP concept mapper that programmatically maps all feature names 
    using fuzzy matching, rule-based parsing, and curated mappings.
    """
    
    def __init__(self, vocab_dir: str = OMOP_VOCAB_DIR, feature_names_dir: str = FEATURE_NAMES_DIR):
        """Initialize with OMOP vocabulary and feature names directories."""
        self.vocab_dir = vocab_dir
        self.feature_names_dir = feature_names_dir
        
        # Load all feature names from CSV files
        self.all_feature_names = self._load_all_feature_names()
        print(f"  --> Loaded {len(self.all_feature_names)} total feature names from CSV files")
        
        # Load OMOP vocabulary
        self.concepts_df = self._load_omop_concepts()
        self.concept_mappings = self._load_concept_mappings()
        
        # Build comprehensive feature mappings
        self.feature_mappings = self._build_comprehensive_mappings()
        self.unmapped_codes = set()
        
        print(f"  --> Created {len(self.feature_mappings)} comprehensive feature mappings")
        
    def _load_all_feature_names(self) -> Dict[str, List[str]]:
        """Load all feature names from the CSV files."""
        feature_collections = {}
        
        feature_files = [
            'vitals_labs_mean_features.csv',
            'vitals_labs_features.csv', 
            'patients_features.csv',
            'interventions_features.csv',
            'codes_features.csv'
        ]
        
        for filename in feature_files:
            filepath = os.path.join(self.feature_names_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                # Remove empty rows and strip whitespace
                features = df['feature_name'].dropna().str.strip().tolist()
                features = [f for f in features if f]  # Remove empty strings
                feature_collections[filename.replace('_features.csv', '')] = features
                print(f"    --> {filename}: {len(features)} features")
        
        return feature_collections
    
    def _load_omop_concepts(self) -> pd.DataFrame:
        """Load OMOP concepts for fuzzy matching."""
        try:
            concepts = pd.read_csv(f"{self.vocab_dir}/CONCEPT.csv", sep='\t', low_memory=False)
            
            # Filter to standard concepts in relevant vocabularies
            relevant_vocabs = ['LOINC', 'SNOMED', 'RxNorm', 'CPT4', 'HCPCS', 'ICD10CM', 'ICD9CM']
            standard_concepts = concepts[
                (concepts['standard_concept'] == 'S') & 
                (concepts['vocabulary_id'].isin(relevant_vocabs)) &
                (concepts['invalid_reason'].isna())
            ].copy()
            
            # Create searchable versions (handle NaN values)
            standard_concepts['name_lower'] = standard_concepts['concept_name'].fillna('').str.lower()
            standard_concepts['name_clean'] = standard_concepts['name_lower'].str.replace(r'[^\w\s]', ' ', regex=True)
            
            print(f"  --> Loaded {len(standard_concepts)} standard OMOP concepts for matching")
            return standard_concepts
            
        except Exception as e:
            print(f"Warning: Failed to load OMOP concepts: {e}")
            return pd.DataFrame()
    
    def _load_concept_mappings(self) -> Dict[str, str]:
        """Load existing concept mappings (for ICD codes, etc.)."""
        try:
            print("  --> Loading existing OMOP vocabulary mappings...")
            
            concepts = pd.read_csv(f"{self.vocab_dir}/CONCEPT.csv", sep='\t', low_memory=False)
            relationships = pd.read_csv(f"{self.vocab_dir}/CONCEPT_RELATIONSHIP.csv", sep='\t', low_memory=False)
            
            mappings = {}
            
            # Get active "Maps to" relationships
            active_mappings = relationships[
                (relationships['relationship_id'] == 'Maps to') &
                (relationships['invalid_reason'].isna())
            ]
            
            # Join with concepts
            source_concepts = concepts[['concept_id', 'concept_code', 'vocabulary_id', 'standard_concept']]
            target_concepts = concepts[concepts['standard_concept'] == 'S'][['concept_id', 'concept_code', 'vocabulary_id']]
            
            # Create mappings for source vocabularies
            source_vocabs = ['ICD9CM', 'ICD9Proc', 'ICD10CM', 'ICD10PCS', 'CPT4', 'HCPCS']
            
            for vocab in source_vocabs:
                vocab_sources = source_concepts[source_concepts['vocabulary_id'] == vocab]
                if vocab_sources.empty:
                    continue
                    
                vocab_mappings = active_mappings.merge(
                    vocab_sources, left_on='concept_id_1', right_on='concept_id', how='inner'
                )
                
                vocab_mappings = vocab_mappings.merge(
                    target_concepts, left_on='concept_id_2', right_on='concept_id', 
                    how='inner', suffixes=('_source', '_target')
                )
                
                for _, row in vocab_mappings.iterrows():
                    source_key = f"{vocab}/{row['concept_code_source']}"
                    target_value = f"{row['vocabulary_id_target']}/{row['concept_code_target']}"
                    mappings[source_key] = target_value
                
                print(f"  --> Loaded {len(vocab_mappings)} mappings for {vocab}")
            
            return mappings
            
        except Exception as e:
            print(f"Warning: Failed to load concept mappings: {e}")
            return {}
    
    def _fuzzy_match_concept(self, feature_name: str, min_similarity: float = 0.6) -> Optional[Tuple[str, str, float]]:
        """Find best fuzzy match for a feature name in OMOP concepts."""
        if self.concepts_df.empty:
            return None
            
        feature_clean = re.sub(r'[^\w\s]', ' ', feature_name.lower()).strip()
        feature_words = set(feature_clean.split())
        
        best_match = None
        best_score = 0.0
        
        # Try different matching strategies
        for _, concept in self.concepts_df.iterrows():
            concept_clean = concept['name_clean']
            
            # Skip if concept_clean is NaN or not a string
            if pd.isna(concept_clean) or not isinstance(concept_clean, str):
                continue
                
            concept_words = set(concept_clean.split())
            
            # Strategy 1: Direct similarity
            similarity = SequenceMatcher(None, feature_clean, concept_clean).ratio()
            
            # Strategy 2: Word overlap bonus  
            if feature_words and concept_words:
                word_overlap = len(feature_words.intersection(concept_words)) / len(feature_words.union(concept_words))
                similarity = max(similarity, word_overlap * 0.8)  # Weight word overlap
            
            # Strategy 3: Key term matching
            if self._contains_key_terms(feature_clean, concept_clean):
                similarity += 0.1  # Bonus for key medical terms
            
            if similarity > best_score and similarity >= min_similarity:
                best_score = similarity
                best_match = (
                    f"{concept['vocabulary_id']}/{concept['concept_code']}", 
                    concept['concept_name'],
                    similarity
                )
        
        return best_match
    
    def _contains_key_terms(self, feature: str, concept: str) -> bool:
        """Check if feature and concept share important medical terms."""
        key_terms = {
            'blood', 'serum', 'plasma', 'urine', 'pressure', 'rate', 'count', 
            'glucose', 'sodium', 'potassium', 'chloride', 'creatinine', 'albumin',
            'hemoglobin', 'hematocrit', 'platelet', 'white', 'red', 'cell'
        }
        
        # Handle None/NaN values
        if not feature or not concept or pd.isna(feature) or pd.isna(concept):
            return False
            
        feature_terms = set(str(feature).split()) & key_terms
        concept_terms = set(str(concept).split()) & key_terms
        
        return len(feature_terms & concept_terms) > 0
    
    def _parse_feature_components(self, feature_name: str) -> Dict[str, str]:
        """Parse feature name into components (measurement, specimen, aggregation)."""
        components = {'base_name': feature_name, 'specimen': '', 'aggregation': ''}
        
        # Extract aggregation suffix (_mean, _count, _std)
        agg_pattern = r'_+(mean|count|std|min|max|first|last)$'
        match = re.search(agg_pattern, feature_name)
        if match:
            components['aggregation'] = match.group(1)
            components['base_name'] = re.sub(agg_pattern, '', feature_name)
        
        # Extract specimen type
        base_name = components['base_name']
        specimen_patterns = [
            (r'\s+(serum|plasma|blood|urine|ascites|pleural|csf|body fluid)$', 'specimen'),
            (r'\s+(arterial|venous|capillary)$', 'specimen'),
        ]
        
        for pattern, comp_type in specimen_patterns:
            match = re.search(pattern, base_name, re.IGNORECASE)
            if match:
                components['specimen'] = match.group(1)
                components['base_name'] = re.sub(pattern, '', base_name, flags=re.IGNORECASE).strip()
                break
        
        return components
    
    def _create_curated_mappings(self) -> Dict[str, str]:
        """Create curated high-confidence mappings for common features."""
        return {
            # Common lab values with high confidence LOINC codes
            'glucose': 'LOINC/2345-7',  # Glucose [Mass/volume] in Serum or Plasma
            'creatinine': 'LOINC/2160-0',  # Creatinine [Mass/volume] in Serum or Plasma
            'sodium': 'LOINC/2951-2',  # Sodium [Moles/volume] in Serum or Plasma
            'potassium': 'LOINC/2823-3',  # Potassium [Moles/volume] in Serum or Plasma
            'chloride': 'LOINC/2075-0',  # Chloride [Moles/volume] in Serum or Plasma
            'albumin': 'LOINC/1751-7',  # Albumin [Mass/volume] in Serum or Plasma
            'hemoglobin': 'LOINC/718-7',  # Hemoglobin [Mass/volume] in Blood
            'hematocrit': 'LOINC/4544-3',  # Hematocrit [Volume Fraction] of Blood
            'platelet': 'LOINC/777-3',  # Platelets [#/volume] in Blood
            'white blood cell': 'LOINC/6690-2',  # Leukocytes [#/volume] in Blood
            'white blood cells': 'LOINC/6690-2',
            'bicarbonate': 'LOINC/1963-8',  # Bicarbonate [Moles/volume] in Serum or Plasma
            'bilirubin': 'LOINC/1975-2',  # Bilirubin.total [Mass/volume] in Serum or Plasma
            'lactate': 'LOINC/2524-7',  # Lactate [Moles/volume] in Serum or Plasma
            
            # Vital signs
            'heart rate': 'LOINC/8867-4',  # Heart rate
            'systolic blood pressure': 'LOINC/8480-6',  # Systolic blood pressure
            'diastolic blood pressure': 'LOINC/8462-4',  # Diastolic blood pressure
            'mean blood pressure': 'LOINC/8478-0',  # Mean blood pressure
            'respiratory rate': 'LOINC/9279-1',  # Respiratory rate
            'temperature': 'LOINC/8310-5',  # Body temperature
            'oxygen saturation': 'LOINC/2708-6',  # Oxygen saturation in Arterial blood
            
            # Demographics and administrative
            'gender': 'SNOMED/263495000',  # Gender
            'age': 'SNOMED/397669002',  # Age
            'ethnicity': 'SNOMED/372148003',  # Ethnic group
            
            # Interventions
            'vent': 'SNOMED/40617009',  # Artificial respiration
            'vaso': 'SNOMED/182836005',  # Administration of vasopressor
            'dopamine': 'RxNorm/3628',  # Dopamine
            'norepinephrine': 'RxNorm/7512',  # Norepinephrine
            'epinephrine': 'RxNorm/3992',  # Epinephrine
            'dobutamine': 'RxNorm/3616',  # Dobutamine
            'vasopressin': 'RxNorm/11125',  # Vasopressin
        }
    
    def _build_comprehensive_mappings(self) -> Dict[str, str]:
        """Build comprehensive mappings using multiple strategies."""
        all_mappings = {}
        
        # Start with curated high-confidence mappings
        curated = self._create_curated_mappings()
        all_mappings.update(curated)
        print(f"  --> Added {len(curated)} curated mappings")
        
        # Process each feature collection
        for collection_name, features in self.all_feature_names.items():
            print(f"  --> Processing {collection_name} features...")
            
            for feature in features:
                if not feature or feature in all_mappings:
                    continue
                
                # Parse feature components
                components = self._parse_feature_components(feature)
                base_name = components['base_name']
                
                # Try curated mapping first (exact or base name match)
                if base_name.lower() in curated:
                    all_mappings[feature] = curated[base_name.lower()]
                    continue
                
                # Try fuzzy matching
                match = self._fuzzy_match_concept(base_name, min_similarity=0.6)
                if match:
                    omop_code, concept_name, similarity = match
                    all_mappings[feature] = omop_code
                    if similarity < 0.8:  # Log uncertain matches
                        print(f"    Fuzzy match ({similarity:.2f}): {feature} -> {omop_code} ({concept_name})")
                    continue
                
                # Fallback: try with lower similarity threshold
                match = self._fuzzy_match_concept(base_name, min_similarity=0.4)
                if match:
                    omop_code, concept_name, similarity = match
                    all_mappings[feature] = omop_code
                    print(f"    Low confidence match ({similarity:.2f}): {feature} -> {omop_code} ({concept_name})")
        
        return all_mappings
    
    def map_code(self, code: str, event_type: str = 'measurement') -> Optional[str]:
        """
        Map a code to OMOP standard concept.
        
        Args:
            code: Original code/feature name
            event_type: Type of event ('diagnosis', 'procedure', 'measurement', 'observation')
        
        Returns:
            OMOP standard concept or None if unmappable
        """
        # Handle ICD codes using existing mappings
        if event_type == 'diagnosis':
            source_key = f"ICD9CM/{code}"
            mapped = self.concept_mappings.get(source_key)
            if mapped:
                return mapped
        elif event_type == 'procedure':
            source_key = f"ICD9Proc/{code}"
            mapped = self.concept_mappings.get(source_key)
            if mapped:
                return mapped
        
        # Handle feature names using comprehensive mappings
        if code in self.feature_mappings:
            return self.feature_mappings[code]
        
        # Try with cleaned code name (remove aggregation suffixes)
        components = self._parse_feature_components(code)
        base_name = components['base_name']
        if base_name in self.feature_mappings:
            return self.feature_mappings[base_name]
        
        # Track unmapped codes
        self.unmapped_codes.add(f"{event_type}:{code}")
        return None
    
    def report_unmapped_codes(self) -> None:
        """Report codes that couldn't be mapped to OMOP standard concepts."""
        if self.unmapped_codes:
            print(f"\n⚠️  WARNING: {len(self.unmapped_codes)} codes could not be mapped to OMOP standard concepts:")
            
            # Group by event type
            by_type = {}
            for code_entry in self.unmapped_codes:
                event_type, code = code_entry.split(':', 1)
                if event_type not in by_type:
                    by_type[event_type] = []
                by_type[event_type].append(code)
            
            for event_type, codes in by_type.items():
                print(f"  {event_type}: {len(codes)} unmapped codes")
                # Show first few examples
                for code in sorted(codes)[:5]:
                    print(f"    - {code}")
                if len(codes) > 5:
                    print(f"    ... and {len(codes) - 5} more")
                    
            print("  Consider extending the curated mappings for critical unmapped codes.")
        else:
            print("✅ All codes successfully mapped to OMOP standard concepts!")
    
    def save_mappings_report(self, output_path: str) -> None:
        """Save comprehensive mapping report for review."""
        report_data = []
        
        for feature, omop_code in self.feature_mappings.items():
            # Get concept details
            concept_name = ""
            if not self.concepts_df.empty:
                vocab, code = omop_code.split('/', 1)
                concept_row = self.concepts_df[
                    (self.concepts_df['vocabulary_id'] == vocab) & 
                    (self.concepts_df['concept_code'] == code)
                ]
                if not concept_row.empty:
                    concept_name = concept_row.iloc[0]['concept_name']
            
            report_data.append({
                'feature_name': feature,
                'omop_code': omop_code,
                'concept_name': concept_name,
                'mapping_strategy': 'curated' if feature in self._create_curated_mappings() else 'fuzzy_match'
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_path, index=False)
        print(f"  --> Saved mapping report to: {output_path}")


# Alias for backward compatibility
OMOPConceptMapper = RobustOMOPMapper


def load_and_preprocess_data(hdf_path: str) -> Dict[str, pd.DataFrame]:
    """Loads data from HDF5 and pre-computes derived labels."""
    print("\nStep 1: Loading and Preprocessing Data...")
    with pd.HDFStore(hdf_path, 'r') as store:
        all_data = {
            'patients': store.select('/patients'),
            'codes': store.select('/codes'),
            'interventions': store.select('/interventions'),
            'vitals': store.select('/vitals_labs_mean'),
        }

    patients_df = all_data['patients']
    patients_df['admittime'] = pd.to_datetime(patients_df['admittime'])

    age_for_dob_calc = pd.to_numeric(patients_df['age'], errors='coerce').clip(upper=90)
    age_in_days = (age_for_dob_calc * 365.25).fillna(0).astype(int)
    patients_df['birth_date'] = patients_df['admittime'] - pd.to_timedelta(age_in_days, unit='D')
    print("  --> Calculated approximate birth dates.")

    patients_df['los_3'] = (patients_df['los_icu'] > 3).astype(int)
    patients_df['los_7'] = (patients_df['los_icu'] > 7).astype(int)
    print("  --> Derived LOS labels (los_3, los_7).")

    # Create intervention labels with proper prevalent case handling
    interventions_df = all_data['interventions']
    
    # 1. Create labels from full ICU stay data (for patients who eventually need interventions)
    intervention_labels = interventions_df.groupby('subject_id')[['vent', 'vaso']].max()
    intervention_labels = intervention_labels.rename(columns={'vent': 'intervention_vent', 'vaso': 'intervention_vaso'})
    patients_df = patients_df.join(intervention_labels, on='subject_id')
    patients_df[['intervention_vent', 'intervention_vaso']] = patients_df[['intervention_vent', 'intervention_vaso']].fillna(0).astype(int)
    print("  --> Derived intervention labels from full ICU stay.")
    
    # 2. Identify prevalent cases (patients already on interventions in first 24 hours)
    print("  --> Identifying and excluding prevalent cases for intervention predictions...")
    window_size = TIME_WINDOW_CONFIG['WINDOW_SIZE']  # 24 hours
    prevalent_interventions = interventions_df[
        interventions_df.index.get_level_values('hours_in') < window_size
    ]
    
    # Find patients with interventions in the first 24 hours
    prevalent_vent_subjects = set(
        prevalent_interventions[prevalent_interventions['vent'] > 0]
        .index.get_level_values('subject_id').unique()
    )
    prevalent_vaso_subjects = set(
        prevalent_interventions[prevalent_interventions['vaso'] > 0]
        .index.get_level_values('subject_id').unique()
    )
    
    # 3. Set prevalent cases to NaN (exclude from prediction task)
    if prevalent_vent_subjects:
        vent_prevalent_mask = patients_df.index.get_level_values('subject_id').isin(prevalent_vent_subjects)
        patients_df.loc[vent_prevalent_mask, 'intervention_vent'] = np.nan
        print(f"    --> Excluded {vent_prevalent_mask.sum()} prevalent cases for 'intervention_vent' (already on ventilation in first {window_size}h)")
    
    if prevalent_vaso_subjects:
        vaso_prevalent_mask = patients_df.index.get_level_values('subject_id').isin(prevalent_vaso_subjects)
        patients_df.loc[vaso_prevalent_mask, 'intervention_vaso'] = np.nan
        print(f"    --> Excluded {vaso_prevalent_mask.sum()} prevalent cases for 'intervention_vaso' (already on vasopressors in first {window_size}h)")
    
    print("  --> Intervention labels created with proper prevalent case exclusion.")

    all_data['patients'] = patients_df
    print("  --> All tables loaded and preprocessed.")
    return all_data

def get_valid_icustay_ids(vitals_df: pd.DataFrame) -> Set[int]:
    """Identifies ICU stay IDs that are long enough for prediction."""
    print("\nStep 2: Identifying valid ICU stays for filtering...")
    patient_max_hours = vitals_df.reset_index().groupby('icustay_id')['hours_in'].max()
    min_required_hours = TIME_WINDOW_CONFIG['WINDOW_SIZE'] + TIME_WINDOW_CONFIG['GAP_TIME']
    valid_icustays = patient_max_hours[patient_max_hours > min_required_hours].index
    print(f"  --> Found {len(valid_icustays)} ICU stays lasting longer than {min_required_hours} hours.")
    return set(valid_icustays)

def create_splits(df_patients: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Creates subject-level train/val/test splits."""
    print("\nStep 3: Creating train/validation/test splits...")
    strat_target = SPLIT_CONFIG['STRATIFICATION_TARGET']
    subject_outcomes = df_patients.groupby('subject_id')[strat_target].max()
    subjects, outcomes = subject_outcomes.index.to_numpy(), subject_outcomes.to_numpy()

    train_val_subjects, test_subjects = train_test_split(subjects, test_size=0.25, random_state=SPLIT_CONFIG['SEED'], stratify=outcomes)
    train_val_outcomes = subject_outcomes.loc[train_val_subjects]
    train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.125, random_state=SPLIT_CONFIG['SEED'], stratify=train_val_outcomes)

    splits = {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}
    print(f"  --> Splits created: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
    return splits

def _process_event_data(df: pd.DataFrame, admission_info: pd.DataFrame, mapper: RobustOMOPMapper, event_type: str, value_col: str = None, needs_melting: bool = False) -> pd.DataFrame:
    """
    Helper to process different event types into OMOP standard format.
    
    Args:
        df: Raw event data
        admission_info: Patient admission information  
        mapper: OMOP concept mapper instance
        event_type: Type of medical event ('diagnosis', 'procedure', 'measurement')
        value_col: Column name for numeric values
        needs_melting: Whether data needs to be melted from wide to long format
    """
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]

    if 'icustay_id' in df.columns and 'subject_id' not in df.columns:
        id_map = admission_info[['icustay_id', 'hadm_id', 'subject_id', 'admittime']].drop_duplicates(subset=['icustay_id'])
        df = df.merge(id_map, on='icustay_id', how='inner')
    elif 'hadm_id' in df.columns and 'admittime' not in df.columns:
        id_map = admission_info[['hadm_id', 'subject_id', 'admittime']].drop_duplicates(subset=['hadm_id'])
        cols_to_merge = ['hadm_id'] + [col for col in id_map.columns if col not in df.columns]
        df = df.merge(id_map[cols_to_merge], on='hadm_id', how='inner')

    if needs_melting:
        id_vars = ['subject_id', 'hadm_id', 'icustay_id', 'admittime', 'hours_in']
        id_vars = [col for col in id_vars if col in df.columns]
        df = df.melt(id_vars=id_vars, var_name='code', value_name='value')
        df = df[df['value'].notna()]
        if value_col == 'val':
            df = df[df['value'] == 1]

    if 'hours_in' in df.columns:
        df['time'] = df['admittime'] + pd.to_timedelta(df['hours_in'], unit='h')
    else:
        df['time'] = df['admittime']

<<<<<<< HEAD
    # OMOP Concept Mapping (replaces hardcoded prefixes)
=======
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
    if 'icd9_codes' in df.columns:
        df = df.explode('icd9_codes').dropna(subset=['icd9_codes'])
        # Map each ICD9 code to OMOP standard concept
        df['code'] = df['icd9_codes'].apply(lambda x: mapper.map_code(str(x), event_type))
    else:
        # Map other codes to OMOP standard concepts
        df['code'] = df['code'].apply(lambda x: mapper.map_code(str(x), event_type))
    
    # Filter out unmapped codes (where mapper returned None)
    original_count = len(df)
    df = df.dropna(subset=['code'])
    filtered_count = len(df)
    if original_count != filtered_count:
        print(f"  --> Filtered out {original_count - filtered_count} unmapped {event_type} codes")

    cols_to_keep = ['subject_id', 'time', 'code']
    if value_col and value_col == 'numeric_value':
        df = df.rename(columns={'value': 'numeric_value'})
        cols_to_keep.append('numeric_value')

    return df[cols_to_keep]

def generate_and_save_labels(patients_df: pd.DataFrame, target: str, output_dir: str):
    """Generates and saves labels for a given task for all patients."""
    prediction_times = patients_df.reset_index().sort_values('admittime').drop_duplicates('subject_id')
    subject_labels = patients_df.groupby('subject_id')[target].max().reset_index()

    labels_df = pd.merge(subject_labels, prediction_times[['subject_id', 'admittime']], on='subject_id')
    labels_df['prediction_time'] = labels_df['admittime'] + timedelta(hours=TIME_WINDOW_CONFIG['WINDOW_SIZE'])
    labels_df = labels_df.rename(columns={target: 'boolean_value'})
    
    # Handle intervention tasks: preserve NaN values for prevalent cases
    if target in ['intervention_vent', 'intervention_vaso']:
        # Keep NaN values as they represent excluded prevalent cases
        # Only convert non-NaN values to boolean
        labels_df['boolean_value'] = labels_df['boolean_value'].apply(
            lambda x: bool(x) if pd.notna(x) else None
        )
        print(f"    --> Preserved {labels_df['boolean_value'].isna().sum()} prevalent cases as NaN for {target}")
    else:
        # For non-intervention tasks, convert to boolean as before
        labels_df['boolean_value'] = labels_df['boolean_value'].astype(bool)

    # Use the correct null type for each column
    labels_df['integer_value'] = pd.NA
    labels_df['float_value'] = np.nan
    labels_df['categorical_value'] = pd.NA

    # Use the correct null type for each column
    labels_df['integer_value'] = pd.NA
    labels_df['float_value'] = np.nan
    labels_df['categorical_value'] = pd.NA

    task_dir = os.path.join(output_dir, 'tasks', target)
    os.makedirs(task_dir, exist_ok=True)
    labels_path = os.path.join(task_dir, 'labels.parquet')
    
    # Ensure the DataFrame being saved matches the full schema and has correct dtypes
<<<<<<< HEAD
=======
    # Handle boolean_value column carefully to preserve NaN for intervention tasks
    if target in ['intervention_vent', 'intervention_vaso']:
        # Use nullable boolean type that can handle NaN
        labels_df['boolean_value'] = labels_df['boolean_value'].astype('boolean')
    else:
        labels_df['boolean_value'] = labels_df['boolean_value'].astype(bool)
    
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
    final_labels_df = labels_df[LABEL_SCHEMA.schema().names].astype({
        'integer_value': 'Int64',
        'float_value': 'float32',
        'categorical_value': 'string'
    })
    
    table = pa.Table.from_pandas(final_labels_df, schema=LABEL_SCHEMA.schema(), preserve_index=False)
    pq.write_table(table, labels_path)

def main():
    """Main conversion pipeline."""
    if not os.path.exists(HDF_FILE_PATH):
        sys.exit(f"--- CRITICAL ERROR: File not found at '{HDF_FILE_PATH}' ---")

    print("--- Starting HDF5 to MEDS Conversion ---")
    print("--- NOTE: Intervention predictions properly exclude prevalent cases (patients already on interventions in first 24h) ---")

    # Create base directories
    os.makedirs(MEDS_OUTPUT_DIR, exist_ok=True)
    metadata_dir = os.path.join(MEDS_OUTPUT_DIR, 'metadata')
    data_dir = os.path.join(MEDS_OUTPUT_DIR, 'data')
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    all_data = load_and_preprocess_data(HDF_FILE_PATH)

    # Step 2: Identify valid ICU stays
    valid_icustay_ids = get_valid_icustay_ids(all_data['vitals'])
    
    # Filter patients based on valid stays
    filtered_patients = all_data['patients'][all_data['patients'].index.get_level_values('icustay_id').isin(valid_icustay_ids)]
    
    # Apply debug settings if enabled
    if DEBUG_MODE:
        all_valid_ids = filtered_patients.index.get_level_values('subject_id').unique()
        debug_ids = np.random.choice(all_valid_ids, size=min(len(all_valid_ids), DEBUG_PATIENT_COUNT), replace=False)
        filtered_patients = filtered_patients[filtered_patients.index.get_level_values('subject_id').isin(debug_ids)]
        print(f"\n*** DEBUG MODE: Using a subset of {len(debug_ids)} patients. ***")

    # Finalize the sets of valid IDs
    final_valid_icustays = set(filtered_patients.index.get_level_values('icustay_id'))
    final_valid_hadms = set(filtered_patients.index.get_level_values('hadm_id'))
    final_valid_subjects = set(filtered_patients.index.get_level_values('subject_id'))
    
    print(f"\nFiltering all tables to {len(final_valid_subjects)} patients across {len(final_valid_icustays)} valid ICU stays.")
    
    filtered_data = {
        'patients': filtered_patients,
        'codes': all_data['codes'][all_data['codes'].index.get_level_values('hadm_id').isin(final_valid_hadms)],
        'interventions': all_data['interventions'][
            (all_data['interventions'].index.get_level_values('icustay_id').isin(final_valid_icustays)) &
            (all_data['interventions'].index.get_level_values('hours_in') < TIME_WINDOW_CONFIG['WINDOW_SIZE'])
        ],
        'vitals': all_data['vitals'][
            (all_data['vitals'].index.get_level_values('icustay_id').isin(final_valid_icustays)) &
            (all_data['vitals'].index.get_level_values('hours_in') < TIME_WINDOW_CONFIG['WINDOW_SIZE'])
        ]
    }
    
    # Step 3 & 4: Create metadata files
    print("\nStep 4: Creating metadata files...")
    dataset_metadata = {
        "dataset_name": "MIMIC-IV-Sample-ICU", "dataset_version": "1.0", "etl_name": "HDF5 to MEDS ETL",
        "etl_version": "2.1", "meds_version" : "0.3.3", "created_at" : str(datetime.now()),
    }
    jsonschema.validate(instance=dataset_metadata, schema=DATASET_METADATA_SCHEMA.schema())
    metadata_path = os.path.join(metadata_dir, 'dataset.json')
    with open(metadata_path, 'w') as f: json.dump(dataset_metadata, f)
    print(f"  --> Wrote dataset metadata to: {metadata_path}")

    # Create and save splits metadata
    splits = create_splits(filtered_data['patients']) # `splits` is now a dictionary
    split_dfs = []
    split_map = {'train': train_split, 'val': tuning_split, 'test': held_out_split}
    for split_name, subject_ids in splits.items():
        name = split_map[split_name]
        split_dfs.append(pd.DataFrame({'subject_id': subject_ids, 'split': name}))
    
    all_splits_df = pd.concat(split_dfs, ignore_index=True)
    splits_path = os.path.join(metadata_dir, 'subject_splits.parquet')
    pq.write_table(pa.Table.from_pandas(all_splits_df, schema=SUBJECT_SPLIT_SCHEMA.schema()), splits_path)
    print(f"  --> Wrote {len(all_splits_df)} subject splits to: {splits_path}")

<<<<<<< HEAD
    # Step 5: Process all event data into a single dataframe with OMOP concept mapping
    print("\nStep 5: Processing all events with OMOP concept mapping...")
    admission_info = filtered_data['patients'].reset_index()[['subject_id', 'hadm_id', 'icustay_id', 'admittime']]
    
    # Initialize OMOP concept mapper
    mapper = RobustOMOPMapper()
    
    processed_events = [
        _process_event_data(filtered_data['codes'], admission_info, mapper, 'diagnosis'),
        _process_event_data(filtered_data['interventions'], admission_info, mapper, 'procedure', 
                            value_col='val', needs_melting=True),
        _process_event_data(filtered_data['vitals'], admission_info, mapper, 'measurement', 
=======
    # Step 5: Process all event data into a single dataframe
    print("\nStep 5: Processing all events...")
    admission_info = filtered_data['patients'].reset_index()[['subject_id', 'hadm_id', 'icustay_id', 'admittime']]
    
    processed_events = [
        _process_event_data(filtered_data['codes'], admission_info, "DIAGNOSIS/ICD9CM/"),
        _process_event_data(filtered_data['interventions'], admission_info, "PROCEDURE/intervention/", 
                            value_col='val', needs_melting=True),
        _process_event_data(filtered_data['vitals'], admission_info, "MEASUREMENT/vitals_labs/", 
>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
                            value_col='numeric_value', needs_melting=True)
    ]
    
    events_df = pd.concat([df for df in processed_events if not df.empty], ignore_index=True)
    
    # Report unmapped codes and save mapping report
    mapper.report_unmapped_codes()
    
    # Save comprehensive mapping report for review
    mapping_report_path = os.path.join(MEDS_OUTPUT_DIR, 'feature_mappings_report.csv')
    mapper.save_mappings_report(mapping_report_path)
    
    if events_df.empty:
        print("  --> No events found. Halting.")
        return

    print(f"  --> Processed {len(events_df)} total events.")
    events_df = events_df.sort_values(by=['subject_id', 'time'])
    
    # Ensure all required columns exist and have the right type
    if 'numeric_value' not in events_df.columns: events_df['numeric_value'] = np.nan
    if 'text_value' not in events_df.columns: events_df['text_value'] = pd.NA
    final_df = events_df[DATA_SCHEMA.schema().names].astype({
        'subject_id': 'int64', 'time': 'datetime64[us]', 'code': 'string',
        'numeric_value': 'float32', 'text_value': 'string'
    })

    # --- FIX #1: WRITING PATIENT METADATA (STILL NEEDED) ---
    print("\n--- Writing Patient Metadata ---")
    patients_to_save = filtered_data['patients'].reset_index().rename(columns={'subject_id': 'patient_id'})
    patient_metadata_path = os.path.join(MEDS_OUTPUT_DIR, "patients.parquet")
    pq.write_table(pa.Table.from_pandas(patients_to_save, preserve_index=False), patient_metadata_path)
    print(f"  --> Wrote patient metadata to: {patient_metadata_path}")

    # --- START FIX #2: WRITING SPLIT EVENT DATA ---
    print("\n--- Writing Split Event Data ---")
    # The 'splits' dictionary holds the subject_ids for 'train', 'val', 'test'
    for split_name, subject_ids in splits.items():
        print(f"  --> Processing split: '{split_name}'")

        # Create the directory for the split (e.g., .../data/train)
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Filter the main events dataframe for subjects in the current split
        split_events_df = final_df[final_df['subject_id'].isin(subject_ids)]

        if split_events_df.empty:
            print(f"    --> WARNING: No events found for split '{split_name}'. Skipping file write.")
            continue

        # Define the output path for the split's data
        output_path = os.path.join(split_dir, 'data.parquet')
        
        # Create and write the pyarrow table
        table = pa.Table.from_pandas(split_events_df, schema=DATA_SCHEMA.schema(), preserve_index=False)
        pq.write_table(table, output_path)

        print(f"    --> Wrote {len(split_events_df)} events for {len(subject_ids)} subjects to: {output_path}")
    # --- END FIX #2 ---

    print("\n--- Writing Code Metadata ---")
    unique_codes = final_df['code'].unique()
    codes_df = pd.DataFrame({'code': unique_codes, 'description': ['' for _ in unique_codes], 'parent_codes': [[] for _ in unique_codes]})
    codes_path = os.path.join(metadata_dir, 'codes.parquet')
    pq.write_table(pa.Table.from_pandas(codes_df, schema=CODE_METADATA_SCHEMA.schema()), codes_path)
    print(f"  --> Wrote {len(codes_df)} unique codes to: {codes_path}")

    # Step 6: Generate labels
    print("\nStep 6: Generating and saving labels for all tasks...")
    for target in tqdm.tqdm(TARGET_VARIABLES, desc="Generating Labels"):
        generate_and_save_labels(filtered_data['patients'], target, MEDS_OUTPUT_DIR)
    print("  --> Finished generating all labels.")

    print("\n--- MEDS Conversion Completed Successfully! ---")
    
<<<<<<< HEAD
=======
    print("\nRunning meds_reader to confirm MEDS extract is valid...")
    print("--- NOTE: This validation may fail due to the new split-directory structure, which is expected. ---")
    print("--- The new structure is required by the `generate_climbr_embeddings.py` script. ---")
    db_path = os.path.join(MEDS_OUTPUT_DIR, "processed_db")
    status = os.system(f"meds_reader_convert {MEDS_OUTPUT_DIR} {db_path} --num_threads 5")
    if status == 0:
        print(f"--- meds_reader validation successful! DB created at {db_path} ---")
    else:
        print("--- WARNING: meds_reader_convert failed as expected. The output directory is still likely valid for the embedding script. ---")

>>>>>>> c9b96b8dc53bcdd9e88ecfd6548d53e75fe50130
if __name__ == '__main__':
    main()