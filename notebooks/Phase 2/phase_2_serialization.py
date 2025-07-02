# phase_2_serialization.py

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
# FEATURE CATEGORIZATION
# =============================================================================

def categorize_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize engineered features by medical domain for better organization.
    
    Args:
        feature_names: List of feature column names
        
    Returns:
        Dictionary mapping categories to lists of feature names
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
        'Trends and Changes': [],
        'Other': []
    }
    
    # Define keywords for each category
    vital_keywords = ['heart rate', 'blood pressure', 'temperature', 'oxygen saturation', 'respiratory rate']
    cardio_keywords = ['cardiac', 'blood pressure', 'heart rate', 'central venous', 'pulmonary artery', 
                      'systemic vascular', 'cardiac output']
    resp_keywords = ['oxygen', 'respiratory', 'tidal volume', 'peep', 'pressure', 'inspired oxygen']
    neuro_keywords = ['glascow', 'coma', 'gcs']
    lab_keywords = ['glucose', 'creatinine', 'urea', 'sodium', 'potassium', 'chloride', 'calcium']
    blood_chem_keywords = ['ph', 'co2', 'bicarbonate', 'lactate', 'anion gap', 'pco2', 'po2']
    hemato_keywords = ['hematocrit', 'hemoglobin', 'platelets', 'red blood cell']
    coag_keywords = ['thromboplastin', 'prothrombin', 'inr', 'fibrinogen']
    trend_keywords = ['trend', 'slope', '_6h', '_24h']
    
    for feature in feature_names:
        feature_lower = feature.lower()
        categorized = False
        
        # Check trends first (most specific)
        if any(keyword in feature_lower for keyword in trend_keywords):
            categories['Trends and Changes'].append(feature)
            categorized = True
        # Then check other categories
        elif any(keyword in feature_lower for keyword in vital_keywords):
            categories['Vital Signs'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in cardio_keywords):
            categories['Cardiovascular'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in resp_keywords):
            categories['Respiratory'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in neuro_keywords):
            categories['Neurological'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in lab_keywords):
            categories['Laboratory Values'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in blood_chem_keywords):
            categories['Blood Chemistry'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in hemato_keywords):
            categories['Hematology'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in coag_keywords):
            categories['Coagulation'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in ['age', 'gender', 'ethnicity', 'weight', 'height']):
            categories['Static Demographics'].append(feature)
            categorized = True
        
        if not categorized:
            categories['Other'].append(feature)
    
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

def serialize_to_markdown_structured(patient_data: pd.Series, config: SerializationConfig) -> str:
    """
    Serialize patient data to structured Markdown format.
    
    Args:
        patient_data: Series with feature values for one patient
        config: Serialization configuration
        
    Returns:
        Markdown formatted string
    """
    feature_names = patient_data.index.tolist()
    categories = categorize_features(feature_names)
    
    markdown_parts = ["# ICU Patient Clinical Data Summary\n"]
    
    for category, features in categories.items():
        if not features:
            continue
            
        markdown_parts.append(f"## {category}\n")
        
        for feature in features:
            value = patient_data[feature]
            formatted_value = format_value(value, config)
            
            if config.include_feature_names:
                clean_name = clean_feature_name(feature)
                markdown_parts.append(f"- **{clean_name}**: {formatted_value}")
            else:
                markdown_parts.append(f"- {formatted_value}")
        
        markdown_parts.append("")  # Empty line between sections
    
    return "\n".join(markdown_parts)

def serialize_to_markdown_dense(patient_data: pd.Series, config: SerializationConfig) -> str:
    """
    Serialize patient data to dense Markdown format.
    
    Args:
        patient_data: Series with feature values for one patient
        config: Serialization configuration
        
    Returns:
        Dense Markdown formatted string
    """
    markdown_parts = ["**ICU Patient Data**: "]
    
    value_strings = []
    for feature, value in patient_data.items():
        formatted_value = format_value(value, config)
        
        if config.include_feature_names:
            clean_name = clean_feature_name(feature)
            value_strings.append(f"{clean_name}: {formatted_value}")
        else:
            value_strings.append(formatted_value)
    
    markdown_parts.append(", ".join(value_strings))
    
    return "".join(markdown_parts)

def serialize_to_json_structured(patient_data: pd.Series, config: SerializationConfig) -> str:
    """
    Serialize patient data to structured JSON format.
    
    Args:
        patient_data: Series with feature values for one patient
        config: Serialization configuration
        
    Returns:
        JSON formatted string
    """
    feature_names = patient_data.index.tolist()
    categories = categorize_features(feature_names)
    
    json_data = {
        "patient_type": "ICU_patient",
        "clinical_data": {}
    }
    
    for category, features in categories.items():
        if not features:
            continue
            
        category_data = {}
        for feature in features:
            value = patient_data[feature]
            
            if pd.isna(value):
                formatted_value = None
            else:
                formatted_value = round(float(value), config.max_decimal_places)
            
            if config.include_feature_names:
                clean_name = clean_feature_name(feature)
                category_data[clean_name] = formatted_value
            else:
                category_data[feature] = formatted_value
        
        json_data["clinical_data"][category.lower().replace(' ', '_')] = category_data
    
    return json.dumps(json_data, indent=2)

def serialize_to_json_flat(patient_data: pd.Series, config: SerializationConfig) -> str:
    """
    Serialize patient data to flat JSON format.
    
    Args:
        patient_data: Series with feature values for one patient
        config: Serialization configuration
        
    Returns:
        Flat JSON formatted string
    """
    json_data = {"patient_type": "ICU_patient"}
    
    for feature, value in patient_data.items():
        if pd.isna(value):
            formatted_value = None
        else:
            formatted_value = round(float(value), config.max_decimal_places)
        
        if config.include_feature_names:
            clean_name = clean_feature_name(feature)
            json_data[clean_name] = formatted_value
        else:
            json_data[feature] = formatted_value
    
    return json.dumps(json_data, indent=2)

def serialize_to_natural_language(patient_data: pd.Series, config: SerializationConfig) -> str:
    """
    Serialize patient data to natural language narrative format.
    
    Args:
        patient_data: Series with feature values for one patient
        config: Serialization configuration
        
    Returns:
        Natural language formatted string
    """
    feature_names = patient_data.index.tolist()
    categories = categorize_features(feature_names)
    
    narrative_parts = ["This ICU patient presents with the following clinical characteristics:\n"]
    
    for category, features in categories.items():
        if not features or category == 'Other':
            continue
            
        # Create narrative for each category
        values_text = []
        for feature in features:
            value = patient_data[feature]
            if pd.isna(value):
                continue
                
            formatted_value = format_value(value, config)
            clean_name = clean_feature_name(feature).lower()
            
            # Create more natural language
            if 'mean' in feature.lower():
                values_text.append(f"average {clean_name.replace('mean', '').strip()} of {formatted_value}")
            elif 'max' in feature.lower():
                values_text.append(f"peak {clean_name.replace('max', '').strip()} of {formatted_value}")
            elif 'min' in feature.lower():
                values_text.append(f"minimum {clean_name.replace('min', '').strip()} of {formatted_value}")
            elif 'std' in feature.lower():
                values_text.append(f"variability in {clean_name.replace('std', '').strip()} (std: {formatted_value})")
            elif 'trend' in feature.lower():
                trend_direction = "increasing" if float(value) > 0 else "decreasing" if float(value) < 0 else "stable"
                values_text.append(f"{trend_direction} trend in {clean_name.replace('trend', '').strip()}")
            else:
                values_text.append(f"{clean_name} of {formatted_value}")
        
        if values_text:
            if category == 'Vital Signs':
                narrative_parts.append(f"Vital signs show {', '.join(values_text[:3])}{'...' if len(values_text) > 3 else ''}.")
            elif category == 'Laboratory Values':
                narrative_parts.append(f"Laboratory results indicate {', '.join(values_text[:3])}{'...' if len(values_text) > 3 else ''}.")
            elif category == 'Trends and Changes':
                narrative_parts.append(f"Clinical trends reveal {', '.join(values_text[:2])}{'...' if len(values_text) > 2 else ''}.")
            else:
                narrative_parts.append(f"{category} measurements include {', '.join(values_text[:2])}{'...' if len(values_text) > 2 else ''}.")
    
    return " ".join(narrative_parts)

# =============================================================================
# MAIN SERIALIZATION FUNCTION
# =============================================================================

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
    
    # Dispatch to appropriate serialization function
    if format_name == 'markdown_structured':
        return serialize_to_markdown_structured(patient_data, config)
    elif format_name == 'markdown_dense':
        return serialize_to_markdown_dense(patient_data, config)
    elif format_name == 'json_structured':
        return serialize_to_json_structured(patient_data, config)
    elif format_name == 'json_flat':
        return serialize_to_json_flat(patient_data, config)
    elif format_name == 'natural_language':
        return serialize_to_natural_language(patient_data, config)
    else:
        raise ValueError(f"Serialization function not implemented for format: {format_name}")

def get_available_formats() -> List[str]:
    """Return list of available serialization formats."""
    return list(SERIALIZATION_FORMATS.keys())

def get_format_description(format_name: str) -> str:
    """Get description of a serialization format."""
    if format_name not in SERIALIZATION_FORMATS:
        return "Unknown format"
    return SERIALIZATION_FORMATS[format_name].description

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def preview_serializations(patient_data: pd.Series, max_length: int = 500) -> Dict[str, str]:
    """
    Generate previews of all serialization formats for a given patient.
    
    Args:
        patient_data: Series with feature values for one patient
        max_length: Maximum length of preview text
        
    Returns:
        Dictionary mapping format names to preview strings
    """
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

def save_serialization_examples(patient_data: pd.Series, output_dir: str, patient_id: str = "example"):
    """
    Save examples of all serialization formats to files.
    
    Args:
        patient_data: Series with feature values for one patient
        output_dir: Directory to save examples
        patient_id: Identifier for the patient (used in filenames)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for format_name in get_available_formats():
        try:
            serialized = serialize_patient_data(patient_data, format_name)
            
            # Determine file extension
            if 'json' in format_name:
                ext = 'json'
            elif 'markdown' in format_name:
                ext = 'md'
            else:
                ext = 'txt'
            
            filename = os.path.join(output_dir, f"{patient_id}_{format_name}.{ext}")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(serialized)
                
            logging.info(f"Saved {format_name} example to {filename}")
            
        except Exception as e:
            logging.error(f"Failed to save {format_name} example: {str(e)}") 