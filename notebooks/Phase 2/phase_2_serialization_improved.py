# phase_2_serialization_improved.py

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# =============================================================================
# SERIALIZATION CONFIGURATION
# =============================================================================

@dataclass
class SerializationConfig:
    """Configuration for different serialization formats."""
    name: str
    description: str
    include_feature_names: bool = True
    include_units: bool = False
    group_by_category: bool = False
    use_structured_format: bool = True
    max_decimal_places: int = 3

# Define available serialization formats
SERIALIZATION_FORMATS = {
    'markdown_structured': SerializationConfig(
        name='markdown_structured',
        description='Structured Markdown with clear sections and headers',
        include_feature_names=True,
        include_units=True,
        group_by_category=True,
        use_structured_format=True,
        max_decimal_places=3
    ),
    'markdown_dense': SerializationConfig(
        name='markdown_dense',
        description='Dense Markdown format with minimal structure',
        include_feature_names=True,
        include_units=False,
        group_by_category=False,
        use_structured_format=False,
        max_decimal_places=2
    ),
    'json_structured': SerializationConfig(
        name='json_structured',
        description='Structured JSON with nested categories',
        include_feature_names=True,
        include_units=True,
        group_by_category=True,
        use_structured_format=True,
        max_decimal_places=4
    ),
    'json_flat': SerializationConfig(
        name='json_flat',
        description='Flat JSON structure',
        include_feature_names=True,
        include_units=False,
        group_by_category=False,
        use_structured_format=False,
        max_decimal_places=3
    ),
    'natural_language': SerializationConfig(
        name='natural_language',
        description='Natural language narrative format',
        include_feature_names=False,
        include_units=True,
        group_by_category=True,
        use_structured_format=True,
        max_decimal_places=2
    )
}

# =============================================================================
# IMPROVED FEATURE CATEGORIZATION
# =============================================================================

def extract_feature_measurements(feature_names: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extract and organize all seven types of measurements for each base feature.
    
    Expected measurement types from Phase 1:
    1. Most Recent Value (last_in_24h or similar)
    2. 6-hour Trend (slope_6h or 6h_trend)
    3. 24-hour Trend (slope_24h or 24h_trend) 
    4. 24-hour Volatility (std)
    5. 24-hour Absolute Minimum (min)
    6. 24-hour Absolute Maximum (max)
    7. Mean value (mean)
    
    Args:
        feature_names: List of feature column names
        
    Returns:
        Dictionary mapping base feature names to measurement dictionaries
    """
    
    # Mapping of suffix patterns to measurement types
    measurement_patterns = {
        'mean': 'Mean Value',
        'std': '24h Volatility (Std Dev)',
        'min': '24h Absolute Minimum', 
        'max': '24h Absolute Maximum',
        'last_in_24h': 'Most Recent Value',
        'last': 'Most Recent Value',
        'slope_6h': '6h Trend (Slope)',
        '6h_trend': '6h Trend (Slope)',
        'slope_24h': '24h Trend (Slope)', 
        '24h_trend': '24h Trend (Slope)',
        'trend_6h': '6h Trend (Slope)',
        'trend_24h': '24h Trend (Slope)',
        '6h_slope': '6h Trend (Slope)',
        '24h_slope': '24h Trend (Slope)'
    }
    
    base_features = {}
    
    for feature in feature_names:
        # Find the base feature name and measurement type
        base_name = None
        measurement_type = None
        
        # Check each pattern
        for suffix, measurement in measurement_patterns.items():
            if feature.endswith(f'_{suffix}'):
                base_name = feature[:-len(f'_{suffix}')]
                measurement_type = measurement
                break
        
        # If no suffix match, treat as base feature (possibly static)
        if base_name is None:
            base_name = feature
            measurement_type = 'Base Value'
            
        # Initialize base feature if not seen
        if base_name not in base_features:
            base_features[base_name] = {}
            
        # Store the measurement
        base_features[base_name][measurement_type] = feature
    
    return base_features

def categorize_base_features(base_feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize base features by medical domain.
    
    Args:
        base_feature_names: List of base feature names
        
    Returns:
        Dictionary mapping categories to lists of base feature names
    """
    categories = {
        'Vital Signs': [],
        'Cardiovascular': [], 
        'Respiratory': [],
        'Neurological': [],
        'Laboratory Values': [],
        'Blood Chemistry': [],
        'Hematology': [],
        'Coagulation': [],
        'Static Demographics': [],
        'Other': []
    }
    
    # Define keywords for each category
    category_keywords = {
        'Vital Signs': ['heart_rate', 'temperature', 'respiratory_rate'],
        'Cardiovascular': ['blood_pressure', 'systolic', 'diastolic', 'cardiac_output', 'central_venous', 'pulmonary_artery'],
        'Respiratory': ['oxygen_saturation', 'tidal_volume', 'peep', 'inspired_oxygen', 'respiratory'],
        'Neurological': ['gcs', 'glascow', 'coma'],
        'Laboratory Values': ['glucose', 'creatinine', 'urea', 'sodium', 'potassium', 'chloride', 'calcium'],
        'Blood Chemistry': ['ph', 'co2', 'bicarbonate', 'lactate', 'anion_gap', 'pco2', 'po2'],
        'Hematology': ['hematocrit', 'hemoglobin', 'platelets', 'red_blood_cell'],
        'Coagulation': ['thromboplastin', 'prothrombin', 'inr', 'fibrinogen'],
        'Static Demographics': ['age', 'gender', 'ethnicity', 'weight', 'height']
    }
    
    for base_feature in base_feature_names:
        feature_lower = base_feature.lower()
        categorized = False
        
        for category, keywords in category_keywords.items():
            if any(keyword in feature_lower for keyword in keywords):
                categories[category].append(base_feature)
                categorized = True
                break
        
        if not categorized:
            categories['Other'].append(base_feature)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

# =============================================================================
# SERIALIZATION FUNCTIONS
# =============================================================================

def format_value(value: float, config: SerializationConfig) -> str:
    """Format a numerical value according to the configuration."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{config.max_decimal_places}f}"

def clean_feature_name(feature_name: str) -> str:
    """Clean feature names for better readability."""
    # Replace underscores with spaces
    clean_name = feature_name.replace('_', ' ')
    # Capitalize words
    clean_name = ' '.join(word.capitalize() for word in clean_name.split())
    return clean_name

def serialize_patient_data(patient_data: pd.Series, format_name: str) -> str:
    """
    Main function to serialize patient data using the specified format.
    
    Args:
        patient_data: Series with feature values for one patient
        format_name: Name of the serialization format to use
        
    Returns:
        Serialized string representation of the patient data
        
    Raises:
        ValueError: If format_name is not recognized
    """
    if format_name not in SERIALIZATION_FORMATS:
        raise ValueError(f"Unknown serialization format: {format_name}. "
                        f"Available formats: {list(SERIALIZATION_FORMATS.keys())}")
    
    config = SERIALIZATION_FORMATS[format_name]
    feature_names = patient_data.index.tolist()
    base_features = extract_feature_measurements(feature_names)
    base_feature_names = list(base_features.keys())
    categories = categorize_base_features(base_feature_names)
    
    if format_name == 'markdown_structured':
        return _serialize_to_markdown_structured(patient_data, config, categories, base_features)
    elif format_name == 'markdown_dense':
        return _serialize_to_markdown_dense(patient_data, config, base_features)
    elif format_name == 'json_structured':
        return _serialize_to_json_structured(patient_data, config, categories, base_features)
    elif format_name == 'json_flat':
        return _serialize_to_json_flat(patient_data, config, base_features)
    elif format_name == 'natural_language':
        return _serialize_to_natural_language(patient_data, config, categories, base_features)
    else:
        raise ValueError(f"Serialization function not implemented for format: {format_name}")

def _serialize_to_markdown_structured(patient_data: pd.Series, config: SerializationConfig, 
                                    categories: Dict[str, List[str]], base_features: Dict[str, Dict[str, str]]) -> str:
    """Serialize to structured Markdown format with all measurement types."""
    markdown_parts = ["# ICU Patient Clinical Data Summary\n"]
    
    for category, base_names in categories.items():
        if not base_names:
            continue
            
        markdown_parts.append(f"## {category}\n")
        
        for base_name in base_names:
            if base_name in base_features:
                clean_base_name = clean_feature_name(base_name)
                markdown_parts.append(f"### {clean_base_name}\n")
                
                measurements = base_features[base_name]
                
                # Define preferred order for measurements
                measurement_order = [
                    'Mean Value',
                    'Most Recent Value', 
                    '24h Absolute Minimum',
                    '24h Absolute Maximum',
                    '24h Volatility (Std Dev)',
                    '6h Trend (Slope)',
                    '24h Trend (Slope)',
                    'Base Value'
                ]
                
                for measurement_type in measurement_order:
                    if measurement_type in measurements:
                        feature_name = measurements[measurement_type]
                        value = patient_data[feature_name]
                        formatted_value = format_value(value, config)
                        markdown_parts.append(f"- **{measurement_type}**: {formatted_value}")
                
                # Add any remaining measurements not in the standard order
                for measurement_type, feature_name in measurements.items():
                    if measurement_type not in measurement_order:
                        value = patient_data[feature_name]
                        formatted_value = format_value(value, config)
                        markdown_parts.append(f"- **{measurement_type}**: {formatted_value}")
                
                markdown_parts.append("")  # Empty line between features
        
        markdown_parts.append("")  # Empty line between categories
    
    return "\n".join(markdown_parts)

def _serialize_to_markdown_dense(patient_data: pd.Series, config: SerializationConfig, 
                                base_features: Dict[str, Dict[str, str]]) -> str:
    """Serialize to dense Markdown format with abbreviated measurement types."""
    markdown_parts = ["**ICU Patient Data**: "]
    
    value_strings = []
    
    # Abbreviation mapping for dense format
    measurement_abbrev = {
        'Mean Value': 'Mean',
        'Most Recent Value': 'Last', 
        '24h Absolute Minimum': 'Min',
        '24h Absolute Maximum': 'Max',
        '24h Volatility (Std Dev)': 'Std',
        '6h Trend (Slope)': '6h-Trend',
        '24h Trend (Slope)': '24h-Trend',
        'Base Value': 'Base'
    }
    
    for base_name, measurements in base_features.items():
        clean_base_name = clean_feature_name(base_name)
        
        # Group measurements for this feature
        feature_measurements = []
        for measurement_type, feature_name in measurements.items():
            value = patient_data[feature_name]
            formatted_value = format_value(value, config)
            abbrev = measurement_abbrev.get(measurement_type, measurement_type)
            feature_measurements.append(f"{abbrev}: {formatted_value}")
        
        if feature_measurements:
            value_strings.append(f"{clean_base_name} ({', '.join(feature_measurements)})")
    
    markdown_parts.append("; ".join(value_strings))
    
    return "".join(markdown_parts)

def _serialize_to_json_structured(patient_data: pd.Series, config: SerializationConfig,
                                 categories: Dict[str, List[str]], base_features: Dict[str, Dict[str, str]]) -> str:
    """Serialize to structured JSON format with nested measurements."""
    json_data = {
        "patient_type": "ICU_patient",
        "clinical_data": {}
    }
    
    for category, base_names in categories.items():
        if not base_names:
            continue
            
        category_data = {}
        
        for base_name in base_names:
            if base_name in base_features:
                measurements = base_features[base_name]
                
                feature_data = {}
                for measurement_type, feature_name in measurements.items():
                    value = patient_data[feature_name]
                    
                    if pd.isna(value):
                        formatted_value = None
                    else:
                        formatted_value = round(float(value), config.max_decimal_places)
                    
                    # Clean measurement type name for JSON key
                    clean_measurement = measurement_type.replace(' ', '_').replace('(', '').replace(')', '').lower()
                    feature_data[clean_measurement] = formatted_value
                
                clean_base_name = clean_feature_name(base_name).replace(' ', '_').lower()
                category_data[clean_base_name] = feature_data
        
        if category_data:
            json_data["clinical_data"][category.lower().replace(' ', '_')] = category_data
    
    return json.dumps(json_data, indent=2)

def _serialize_to_json_flat(patient_data: pd.Series, config: SerializationConfig,
                           base_features: Dict[str, Dict[str, str]]) -> str:
    """Serialize to flat JSON format with descriptive keys."""
    json_data = {"patient_type": "ICU_patient"}
    
    for base_name, measurements in base_features.items():
        clean_base_name = clean_feature_name(base_name).replace(' ', '_')
        
        for measurement_type, feature_name in measurements.items():
            value = patient_data[feature_name]
            
            if pd.isna(value):
                formatted_value = None
            else:
                formatted_value = round(float(value), config.max_decimal_places)
            
            # Create descriptive key
            clean_measurement = measurement_type.replace(' ', '_').replace('(', '').replace(')', '').lower()
            key = f"{clean_base_name}_{clean_measurement}"
            json_data[key] = formatted_value
    
    return json.dumps(json_data, indent=2)

def _serialize_to_natural_language(patient_data: pd.Series, config: SerializationConfig,
                                  categories: Dict[str, List[str]], base_features: Dict[str, Dict[str, str]]) -> str:
    """Serialize to natural language narrative format."""
    narrative_parts = ["This ICU patient presents with the following clinical characteristics:"]
    
    for category, base_names in categories.items():
        if not base_names:
            continue
            
        category_narratives = []
        
        for base_name in base_names:
            if base_name in base_features:
                measurements = base_features[base_name]
                clean_base_name = clean_feature_name(base_name).lower()
                
                # Create narrative for this feature
                feature_parts = []
                
                if 'Mean Value' in measurements:
                    value = patient_data[measurements['Mean Value']]
                    if not pd.isna(value):
                        formatted_value = format_value(value, config)
                        feature_parts.append(f"average {clean_base_name} of {formatted_value}")
                
                if 'Most Recent Value' in measurements:
                    value = patient_data[measurements['Most Recent Value']]
                    if not pd.isna(value):
                        formatted_value = format_value(value, config)
                        feature_parts.append(f"most recent {clean_base_name} of {formatted_value}")
                
                if '24h Volatility (Std Dev)' in measurements:
                    value = patient_data[measurements['24h Volatility (Std Dev)']]
                    if not pd.isna(value):
                        formatted_value = format_value(value, config)
                        if float(formatted_value) > 0:
                            feature_parts.append(f"variability in {clean_base_name} (std: {formatted_value})")
                
                if '24h Absolute Minimum' in measurements and '24h Absolute Maximum' in measurements:
                    min_val = patient_data[measurements['24h Absolute Minimum']]
                    max_val = patient_data[measurements['24h Absolute Maximum']]
                    if not pd.isna(min_val) and not pd.isna(max_val):
                        min_formatted = format_value(min_val, config)
                        max_formatted = format_value(max_val, config)
                        feature_parts.append(f"{clean_base_name} ranging from {min_formatted} to {max_formatted}")
                
                # Add trend information
                trends = []
                if '6h Trend (Slope)' in measurements:
                    value = patient_data[measurements['6h Trend (Slope)']]
                    if not pd.isna(value) and abs(value) > 0.01:  # Only mention significant trends
                        direction = "increasing" if value > 0 else "decreasing"
                        trends.append(f"6-hour {direction} trend")
                
                if '24h Trend (Slope)' in measurements:
                    value = patient_data[measurements['24h Trend (Slope)']]
                    if not pd.isna(value) and abs(value) > 0.01:  # Only mention significant trends
                        direction = "increasing" if value > 0 else "decreasing"
                        trends.append(f"24-hour {direction} trend")
                
                if trends:
                    feature_parts.append(f"{clean_base_name} showing {' and '.join(trends)}")
                
                if feature_parts:
                    category_narratives.append(f"{clean_base_name}: {', '.join(feature_parts)}")
        
        if category_narratives:
            category_description = f"{category.lower()} show " + "; ".join(category_narratives)
            narrative_parts.append(category_description)
    
    return " ".join(narrative_parts) + "."

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_formats() -> List[str]:
    """Return list of available serialization formats."""
    return list(SERIALIZATION_FORMATS.keys())

def get_format_description(format_name: str) -> str:
    """Get description of a serialization format."""
    if format_name not in SERIALIZATION_FORMATS:
        return "Unknown format"
    return SERIALIZATION_FORMATS[format_name].description

def preview_serializations(patient_data: pd.Series, max_length: int = 500) -> Dict[str, str]:
    """Generate previews of all serialization formats for a given patient."""
    previews = {}
    
    for format_name in get_available_formats():
        try:
            serialized = serialize_patient_data(patient_data, format_name)
            preview = serialized[:max_length]
            if len(serialized) > max_length:
                preview += "..."
            previews[format_name] = preview
        except Exception as e:
            previews[format_name] = f"Error: {str(e)}"
    
    return previews

def analyze_feature_coverage(feature_names: List[str]) -> Dict[str, Any]:
    """Analyze feature coverage to understand what measurement types are captured."""
    base_features = extract_feature_measurements(feature_names)
    
    # Count measurement types
    measurement_counts = {}
    for base_name, measurements in base_features.items():
        for measurement_type in measurements.keys():
            measurement_counts[measurement_type] = measurement_counts.get(measurement_type, 0) + 1
    
    # Analyze completeness
    expected_measurements = [
        'Mean Value',
        'Most Recent Value', 
        '24h Absolute Minimum',
        '24h Absolute Maximum',
        '24h Volatility (Std Dev)',
        '6h Trend (Slope)',
        '24h Trend (Slope)'
    ]
    
    coverage_analysis = {
        'total_features': len(feature_names),
        'base_features': len(base_features),
        'measurement_types_found': list(measurement_counts.keys()),
        'measurement_counts': measurement_counts,
        'expected_measurements': expected_measurements,
        'missing_measurements': [m for m in expected_measurements if m not in measurement_counts],
        'completeness_ratio': len([m for m in expected_measurements if m in measurement_counts]) / len(expected_measurements)
    }
    
    return coverage_analysis 