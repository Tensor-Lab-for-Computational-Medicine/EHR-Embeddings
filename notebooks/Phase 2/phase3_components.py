# phase3_components.py
"""
Modular components for Phase III factorial evaluation.
Implements the four orthogonal information components: B, S, R, I
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import yaml
from dataclasses import dataclass


@dataclass
class ReferenceRange:
    """Structure to hold reference range information"""
    feature_name: str
    male_lower: Optional[float]
    male_upper: Optional[float]
    female_lower: Optional[float]
    female_upper: Optional[float]
    units: str
    notes: str

    def get_interpretation(self, value: float, gender: str = 'unknown') -> str:
        """
        Get clinical interpretation of a value based on reference ranges.
        
        Args:
            value: The numerical value to interpret
            gender: 'male', 'female', or 'unknown'
            
        Returns:
            'Normal', 'High', 'Low', or 'Unknown' if no reference range available
        """
        if pd.isna(value):
            return 'N/A'
            
        # Select appropriate bounds based on gender
        if gender.lower() == 'male':
            lower_bound = self.male_lower
            upper_bound = self.male_upper
        elif gender.lower() == 'female':
            lower_bound = self.female_lower
            upper_bound = self.female_upper
        else:
            # Use male ranges as default or take the most restrictive range
            lower_bound = self.male_lower
            upper_bound = self.male_upper
            
        # If no valid bounds available, return Unknown
        if pd.isna(lower_bound) and pd.isna(upper_bound):
            return 'Unknown'
            
        # Determine interpretation
        if lower_bound is not None and not pd.isna(lower_bound) and value < lower_bound:
            return 'Low'
        elif upper_bound is not None and not pd.isna(upper_bound) and value > upper_bound:
            return 'High'
        else:
            return 'Normal'


class ReferenceRangeManager:
    """Manages loading and lookup of reference ranges"""
    
    def __init__(self, reference_file_path: str):
        self.ranges = self._load_reference_ranges(reference_file_path)
        
    def _load_reference_ranges(self, file_path: str) -> Dict[str, ReferenceRange]:
        """Load reference ranges from CSV file"""
        df = pd.read_csv(file_path)
        ranges = {}
        
        for _, row in df.iterrows():
            feature_name = row['feature_name']
            ranges[feature_name] = ReferenceRange(
                feature_name=feature_name,
                male_lower=pd.to_numeric(row['Male Lower Bound'], errors='coerce'),
                male_upper=pd.to_numeric(row['Male Upper Bound'], errors='coerce'),
                female_lower=pd.to_numeric(row['Female Lower Bound'], errors='coerce'),
                female_upper=pd.to_numeric(row['Female Upper Bound'], errors='coerce'),
                units=str(row['Units']) if pd.notna(row['Units']) else '',
                notes=str(row['Notes/Context']) if pd.notna(row['Notes/Context']) else ''
            )
            
        return ranges
    
    def get_interpretation(self, feature_name: str, value: float, gender: str = 'unknown') -> str:
        """Get clinical interpretation for a feature value"""
        # Try exact match first
        if feature_name in self.ranges:
            return self.ranges[feature_name].get_interpretation(value, gender)
            
        # Try to find base feature name (remove suffixes like _last, _mean, etc.)
        base_name = self._extract_base_name(feature_name)
        if base_name in self.ranges:
            return self.ranges[base_name].get_interpretation(value, gender)
            
        return 'Unknown'
    
    def _extract_base_name(self, feature_name: str) -> str:
        """Extract base feature name by removing common suffixes"""
        suffixes = ['_last', '_mean', '_min_24h', '_max_24h', '_stddev_24h', 
                   '_count', '_count_6h', '_slope_24h', '_slope_6h', '_value']
        
        for suffix in suffixes:
            if feature_name.endswith(suffix):
                return feature_name[:-len(suffix)]
                
        return feature_name


def categorize_features_by_patterns(feature_names: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Categorize features based on predefined patterns.
    
    Categories and their patterns:
    - Static Features: {feature}_value, {feature}_count
    - Event-Driven Features: {feature}_count, {feature}_last  
    - High-Frequency Physiological: _last, _mean, _min_24h, _max_24h, _stddev_24h, _count, _count_6h, _slope_24h, _slope_6h
    - Labile Lab Features: _last, _mean, _stddev_24h, _count, _slope_24h
    - Stable Index Features: _last, _count
    - Sparse Dynamic Features: _last, _mean, _count, _slope_24h
    """
    
    # Define the suffix patterns for each category
    category_patterns = {
        'Static Features': ['_value', '_count'],
        'Event-Driven Features': ['_count', '_last'],
        'High-Frequency Physiological': ['_last', '_mean', '_min_24h', '_max_24h', '_stddev_24h', '_count', '_count_6h', '_slope_24h', '_slope_6h'],
        'Labile Lab Features': ['_last', '_mean', '_stddev_24h', '_count', '_slope_24h'],
        'Stable Index Features': ['_last', '_count'],
        'Sparse Dynamic Features': ['_last', '_mean', '_count', '_slope_24h']
    }
    
    # Initialize result structure
    categorized = {}
    for category in category_patterns:
        categorized[category] = {}
    
    # Extract base features and their measurements
    for feature in feature_names:
        # Find which category this feature belongs to
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if feature.endswith(pattern):
                    base_name = feature[:-len(pattern)]
                    measurement_type = pattern[1:]  # Remove leading underscore
                    
                    if base_name not in categorized[category]:
                        categorized[category][base_name] = {}
                    
                    categorized[category][base_name][measurement_type] = feature
                    break
    
    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


class ComponentB:
    """Component B: Base Data - Minimalist key-value pairs with abbreviated names"""
    
    @staticmethod
    def abbreviate_feature_name(feature_name: str) -> str:
        """Abbreviate feature names for token efficiency"""
        # Common abbreviations mapping
        abbreviations = {
            'creatinine': 'creat',
            'blood_urea_nitrogen': 'bun',
            'heart_rate': 'hr',
            'blood_pressure': 'bp',
            'respiratory_rate': 'rr',
            'temperature': 'temp',
            'oxygen_saturation': 'o2sat',
            'systolic_blood_pressure': 'sbp',
            'diastolic_blood_pressure': 'dbp',
            'mean_blood_pressure': 'mbp',
            'white_blood_cell_count': 'wbc',
            'red_blood_cell_count': 'rbc',
            'hemoglobin': 'hgb',
            'hematocrit': 'hct',
            'potassium': 'k',
            'sodium': 'na',
            'chloride': 'cl',
            'glucose': 'gluc',
            'partial_pressure_of_oxygen': 'po2',
            'partial_pressure_of_carbon_dioxide': 'pco2',
            'bicarbonate': 'hco3',
            'lactate': 'lac',
            'alanine_aminotransferase': 'alt',
            'asparate_aminotransferase': 'ast',
            'alkaline_phosphate': 'alkp',
            'bilirubin': 'bili',
            'albumin': 'alb',
            'magnesium': 'mg',
            'phosphate': 'phos',
            'calcium': 'ca'
        }
        
        # Start with the original name
        abbreviated = feature_name.lower()
        
        # Apply abbreviations to any matching substrings
        for full_name, abbrev in abbreviations.items():
            abbreviated = abbreviated.replace(full_name, abbrev)
        
        return abbreviated
    
    @staticmethod
    def serialize(patient_data: pd.Series, max_decimals: int = 3) -> str:
        """Serialize patient data as abbreviated key-value pairs"""
        abbreviated_pairs = []
        for feature_name, value in patient_data.items():
            if pd.notna(value):
                abbreviated_name = ComponentB.abbreviate_feature_name(feature_name)
                formatted_value = f"{float(value):.{max_decimals}f}"
                abbreviated_pairs.append(f"{abbreviated_name}: {formatted_value}")
                
        return "; ".join(abbreviated_pairs)


class ComponentS:
    """Component S: Structural Hierarchy - YAML-like clinical categorization"""
    
    @staticmethod
    def serialize(patient_data: pd.Series, max_decimals: int = 3) -> str:
        """Serialize with clinical structural hierarchy"""
        feature_names = list(patient_data.index)
        categorized = categorize_features_by_patterns(feature_names)
        
        yaml_structure = []
        
        for category, base_features in categorized.items():
            yaml_structure.append(f"{category}:")
            
            for base_name, measurements in base_features.items():
                yaml_structure.append(f"  {base_name}:")
                
                for measurement_type, feature_name in measurements.items():
                    if feature_name in patient_data and pd.notna(patient_data[feature_name]):
                        value = patient_data[feature_name]
                        formatted_value = f"{float(value):.{max_decimals}f}"
                        yaml_structure.append(f"    {measurement_type}: {formatted_value}")
        
        return "\n".join(yaml_structure)


class ComponentR:
    """Component R: Relational-Temporal Context - Explicit statements about trends and physiological relationships"""
    
    def __init__(self, reference_manager: ReferenceRangeManager):
        self.reference_manager = reference_manager
    
    def serialize(self, patient_data: pd.Series, max_decimals: int = 3, gender: str = 'unknown') -> str:
        """Generate relational-temporal context statements"""
        statements = []
        
        # Generate trend statements for time-series variables
        trend_statements = self._generate_trend_statements(patient_data)
        if trend_statements:
            statements.extend(trend_statements)
        
        # Generate physiological relationship statements
        physio_statements = self._generate_physiological_relationships(patient_data, gender)
        if physio_statements:
            statements.extend(physio_statements)
        
        return "\n".join(statements) if statements else "No significant relational patterns identified."
    
    def _generate_trend_statements(self, patient_data: pd.Series) -> List[str]:
        """Generate qualitative trend statements for key time-series variables"""
        statements = []
        
        # Define key variables to analyze for trends
        trend_variables = {
            'heart_rate': ['hr_slope_24h', 'hr_slope_6h'],
            'blood_pressure': ['sbp_slope_24h', 'dbp_slope_24h'],
            'temperature': ['temp_slope_24h', 'temp_slope_6h'],
            'respiratory_rate': ['rr_slope_24h', 'rr_slope_6h'],
            'creatinine': ['creat_slope_24h'],
            'lactate': ['lac_slope_24h']
        }
        
        for variable, slope_features in trend_variables.items():
            for slope_feature in slope_features:
                if slope_feature in patient_data and pd.notna(patient_data[slope_feature]):
                    slope_value = patient_data[slope_feature]
                    trend_desc = self._classify_trend(slope_value)
                    time_window = "24h" if "24h" in slope_feature else "6h"
                    
                    if trend_desc != "Stable":
                        statements.append(f"Trend ({time_window}): {variable.replace('_', ' ').title()} - {trend_desc}")
        
        return statements
    
    def _classify_trend(self, slope_value: float) -> str:
        """Classify trend based on slope magnitude"""
        abs_slope = abs(slope_value)
        
        if abs_slope < 0.1:
            return "Stable"
        elif abs_slope < 0.5:
            direction = "Increasing" if slope_value > 0 else "Decreasing"
            return f"Gradually {direction}"
        elif abs_slope < 1.0:
            direction = "Increasing" if slope_value > 0 else "Decreasing"
            return f"Moderately {direction}"
        else:
            direction = "Increasing" if slope_value > 0 else "Decreasing"
            return f"Rapidly {direction}"
    
    def _generate_physiological_relationships(self, patient_data: pd.Series, gender: str) -> List[str]:
        """Generate statements about critical physiological pairings"""
        statements = []
        
        # Define physiological relationship patterns
        relationships = [
            {
                'name': 'Shock Physiology',
                'condition': self._check_shock_physiology,
                'description': 'Evidence of Shock Physiology: Concurrent Hypotension and Tachycardia'
            },
            {
                'name': 'Respiratory Distress',
                'condition': self._check_respiratory_distress,
                'description': 'Evidence of Respiratory Distress: Tachypnea with Abnormal Gas Exchange'
            },
            {
                'name': 'Renal Dysfunction',
                'condition': self._check_renal_dysfunction,
                'description': 'Evidence of Renal Dysfunction: Elevated Creatinine with Oliguria'
            },
            {
                'name': 'Sepsis Pattern',
                'condition': self._check_sepsis_pattern,
                'description': 'Evidence of Sepsis Pattern: Fever/Hypothermia with Leukocytosis/Leukopenia'
            }
        ]
        
        for relationship in relationships:
            if relationship['condition'](patient_data, gender):
                statements.append(relationship['description'])
        
        return statements
    
    def _check_shock_physiology(self, patient_data: pd.Series, gender: str) -> bool:
        """Check for concurrent hypotension and tachycardia"""
        hr_high = False
        bp_low = False
        
        # Check for tachycardia
        hr_features = ['hr_last', 'hr_mean']
        for feature in hr_features:
            if feature in patient_data and pd.notna(patient_data[feature]):
                if patient_data[feature] > 100:  # Tachycardia threshold
                    hr_high = True
                    break
        
        # Check for hypotension
        bp_features = ['sbp_last', 'sbp_mean', 'mbp_last', 'mbp_mean']
        for feature in bp_features:
            if feature in patient_data and pd.notna(patient_data[feature]):
                threshold = 90 if 'sbp' in feature else 65  # SBP vs MAP thresholds
                if patient_data[feature] < threshold:
                    bp_low = True
                    break
        
        return hr_high and bp_low
    
    def _check_respiratory_distress(self, patient_data: pd.Series, gender: str) -> bool:
        """Check for tachypnea with abnormal gas exchange"""
        rr_high = False
        gas_abnormal = False
        
        # Check for tachypnea
        rr_features = ['rr_last', 'rr_mean']
        for feature in rr_features:
            if feature in patient_data and pd.notna(patient_data[feature]):
                if patient_data[feature] > 20:  # Tachypnea threshold
                    rr_high = True
                    break
        
        # Check for abnormal gas exchange
        gas_features = ['po2_last', 'pco2_last', 'o2sat_last']
        for feature in gas_features:
            if feature in patient_data and pd.notna(patient_data[feature]):
                if 'po2' in feature and patient_data[feature] < 80:
                    gas_abnormal = True
                elif 'pco2' in feature and (patient_data[feature] < 35 or patient_data[feature] > 45):
                    gas_abnormal = True
                elif 'o2sat' in feature and patient_data[feature] < 95:
                    gas_abnormal = True
        
        return rr_high and gas_abnormal
    
    def _check_renal_dysfunction(self, patient_data: pd.Series, gender: str) -> bool:
        """Check for elevated creatinine with oliguria"""
        creat_high = False
        
        # Check for elevated creatinine
        creat_features = ['creat_last', 'creat_mean']
        for feature in creat_features:
            if feature in patient_data and pd.notna(patient_data[feature]):
                interpretation = self.reference_manager.get_interpretation(feature, patient_data[feature], gender)
                if interpretation == 'High':
                    creat_high = True
                    break
        
        # For now, assume oliguria if creatinine is elevated (could be refined with urine output data)
        return creat_high
    
    def _check_sepsis_pattern(self, patient_data: pd.Series, gender: str) -> bool:
        """Check for fever/hypothermia with leukocytosis/leukopenia"""
        temp_abnormal = False
        wbc_abnormal = False
        
        # Check for abnormal temperature
        temp_features = ['temp_last', 'temp_mean']
        for feature in temp_features:
            if feature in patient_data and pd.notna(patient_data[feature]):
                if patient_data[feature] > 38.3 or patient_data[feature] < 36.0:  # Fever or hypothermia
                    temp_abnormal = True
                    break
        
        # Check for abnormal WBC
        wbc_features = ['wbc_last', 'wbc_mean']
        for feature in wbc_features:
            if feature in patient_data and pd.notna(patient_data[feature]):
                if patient_data[feature] > 12 or patient_data[feature] < 4:  # Leukocytosis or leukopenia
                    wbc_abnormal = True
                    break
        
        return temp_abnormal and wbc_abnormal


class ComponentI:
    """Component I: Clinical Interpretations - Qualitative assessments"""
    
    def __init__(self, reference_manager: ReferenceRangeManager):
        self.reference_manager = reference_manager
    
    def serialize(self, patient_data: pd.Series, max_decimals: int = 3, 
                  gender: str = 'unknown') -> str:
        """Serialize with clinical interpretations"""
        interpreted_pairs = []
        for feature_name, value in patient_data.items():
            if pd.notna(value):
                formatted_value = f"{float(value):.{max_decimals}f}"
                interpretation = self.reference_manager.get_interpretation(
                    feature_name, value, gender
                )
                
                if interpretation != 'Unknown':
                    interpreted_pairs.append(f"{feature_name}: {formatted_value} ({interpretation})")
                else:
                    interpreted_pairs.append(f"{feature_name}: {formatted_value}")
                    
        return "; ".join(interpreted_pairs)


class FormatGenerator:
    """Generates all 8 format combinations for the factorial experiment"""
    
    def __init__(self, reference_manager: ReferenceRangeManager):
        self.reference_manager = reference_manager
        self.component_b = ComponentB()
        self.component_s = ComponentS()
        self.component_r = ComponentR(reference_manager)
        self.component_i = ComponentI(reference_manager)
    
    def generate_format(self, patient_data: pd.Series, format_id: str, 
                       max_decimals: int = 3, gender: str = 'unknown') -> str:
        """
        Generate specific format based on format ID.
        
        Format IDs match the new Phase III plan:
        F1: B (Base Data Only)
        F2: B + S (Structured Hierarchy)
        F3: B + R (Relational-Temporal Context)
        F4: B + I (Clinical Interpretations)
        F5: B + S + R (Structured Hierarchy & Relational Context)
        F6: B + S + I (Structured Hierarchy & Interpretations)
        F7: B + R + I (Relational Context & Interpretations)
        F8: B + S + R + I (Full Contextual Information)
        """
        
        if format_id == 'F1':
            return self.component_b.serialize(patient_data, max_decimals)
        
        elif format_id == 'F2':
            return self._combine_b_s(patient_data, max_decimals)
        
        elif format_id == 'F3':
            return self._combine_b_r(patient_data, max_decimals, gender)
        
        elif format_id == 'F4':
            return self._combine_b_i(patient_data, max_decimals, gender)
        
        elif format_id == 'F5':
            return self._combine_b_s_r(patient_data, max_decimals, gender)
        
        elif format_id == 'F6':
            return self._combine_b_s_i(patient_data, max_decimals, gender)
        
        elif format_id == 'F7':
            return self._combine_b_r_i(patient_data, max_decimals, gender)
        
        elif format_id == 'F8':
            return self._combine_b_s_r_i(patient_data, max_decimals, gender)
        
        else:
            raise ValueError(f"Unknown format ID: {format_id}")
    
    def _combine_b_s(self, patient_data: pd.Series, max_decimals: int) -> str:
        """Combine Base + Structural"""
        base = self.component_b.serialize(patient_data, max_decimals)
        structural = self.component_s.serialize(patient_data, max_decimals)
        return f"Base Data: {base}\n\nStructural Organization:\n{structural}"
    
    def _combine_b_r(self, patient_data: pd.Series, max_decimals: int, gender: str) -> str:
        """Combine Base + Relational-Temporal Context"""
        base = self.component_b.serialize(patient_data, max_decimals)
        relational = self.component_r.serialize(patient_data, max_decimals, gender)
        return f"Base Data: {base}\n\nRelational-Temporal Context:\n{relational}"
    
    def _combine_b_i(self, patient_data: pd.Series, max_decimals: int, gender: str) -> str:
        """Combine Base + Clinical Interpretations"""
        base = self.component_b.serialize(patient_data, max_decimals)
        interpreted = self.component_i.serialize(patient_data, max_decimals, gender)
        return f"Raw Values: {base}\n\nClinical Assessment: {interpreted}"
    
    def _combine_b_s_r(self, patient_data: pd.Series, max_decimals: int, gender: str) -> str:
        """Combine Base + Structural + Relational-Temporal"""
        base = self.component_b.serialize(patient_data, max_decimals)
        structural = self.component_s.serialize(patient_data, max_decimals)
        relational = self.component_r.serialize(patient_data, max_decimals, gender)
        return f"Base Data: {base}\n\nStructural Organization:\n{structural}\n\nRelational-Temporal Context:\n{relational}"
    
    def _combine_b_s_i(self, patient_data: pd.Series, max_decimals: int, gender: str) -> str:
        """Combine Base + Structural + Clinical Interpretations"""
        base = self.component_b.serialize(patient_data, max_decimals)
        structural = self.component_s.serialize(patient_data, max_decimals)
        interpreted = self.component_i.serialize(patient_data, max_decimals, gender)
        return f"Raw Values: {base}\n\nStructural Organization:\n{structural}\n\nClinical Assessment: {interpreted}"
    
    def _combine_b_r_i(self, patient_data: pd.Series, max_decimals: int, gender: str) -> str:
        """Combine Base + Relational-Temporal + Clinical Interpretations"""
        base = self.component_b.serialize(patient_data, max_decimals)
        relational = self.component_r.serialize(patient_data, max_decimals, gender)
        interpreted = self.component_i.serialize(patient_data, max_decimals, gender)
        return f"Base Data: {base}\n\nRelational-Temporal Context:\n{relational}\n\nClinical Assessment: {interpreted}"
    
    def _combine_b_s_r_i(self, patient_data: pd.Series, max_decimals: int, gender: str) -> str:
        """Combine all components"""
        base = self.component_b.serialize(patient_data, max_decimals)
        structural = self.component_s.serialize(patient_data, max_decimals)
        relational = self.component_r.serialize(patient_data, max_decimals, gender)
        interpreted = self.component_i.serialize(patient_data, max_decimals, gender)
        
        return f"""Base Data: {base}

Structural Organization:
{structural}

Relational-Temporal Context:
{relational}

Clinical Assessment: {interpreted}"""


def get_all_format_ids() -> List[str]:
    """Return list of all format IDs"""
    return ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']


def get_format_description(format_id: str) -> str:
    """Get human-readable description of a format matching the new Phase III plan"""
    descriptions = {
        'F1': 'Base Data Only',
        'F2': 'Structured Hierarchy',
        'F3': 'Relational-Temporal Context', 
        'F4': 'Clinical Interpretations',
        'F5': 'Structured Hierarchy & Relational Context',
        'F6': 'Structured Hierarchy & Interpretations',
        'F7': 'Relational Context & Interpretations',
        'F8': 'Full Contextual Information'
    }
    return descriptions.get(format_id, 'Unknown format') 