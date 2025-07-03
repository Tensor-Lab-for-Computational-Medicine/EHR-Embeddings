# text_generator.py
"""
Final core logic for generating representations. This version robustly filters
out all imputed values and correctly hides 6-hour data when the measurement count is zero.
"""
import pandas as pd
from collections import defaultdict
from typing import Tuple, Set, Any

from config import FEATURE_LABEL_MAP, _MEASUREMENT_SUFFIXES

def _parse_feature_name(feature_name: str) -> Tuple[str, str, str]:
    """Parses a feature name into its raw base, display base, and suffix."""
    if feature_name in ['age', 'gender', 'ethnicity', 'insurance']:
        return feature_name, feature_name.title(), 'value'

    for suffix in _MEASUREMENT_SUFFIXES:
        if feature_name.endswith(f'_{suffix}'):
            raw_base = feature_name[:-len(suffix)-1]
            display_base = raw_base.replace('_', ' ').title()
            
            if raw_base.endswith('_mean'):
                raw_base = raw_base[:-len('_mean')]
                display_base = raw_base.replace('_', ' ').title()
                suffix = f"mean_{suffix}"

            return raw_base, display_base, suffix
            
    return feature_name, feature_name.replace('_', ' ').title(), 'value'

def _get_qualitative_flag(raw_base_name: str, value: float, reference_ranges: pd.DataFrame, gender: str) -> str:
    """Generates an informative qualitative flag with rounded bounds."""
    lookup_key = f"{raw_base_name}_mean"
    if lookup_key not in reference_ranges.index or pd.isna(value):
        return ""

    bounds = reference_ranges.loc[lookup_key]
    l_bound_col, u_bound_col = ('Female Lower Bound', 'Female Upper Bound') if gender == 'F' else ('Male Lower Bound', 'Male Upper Bound')
    
    lower = pd.to_numeric(bounds.get(l_bound_col), errors='coerce')
    upper = pd.to_numeric(bounds.get(u_bound_col), errors='coerce')

    if pd.notna(upper) and value > upper: return f" (High > {upper:.2f})"
    if pd.notna(lower) and value < lower: return f" (Low < {lower:.2f})"
    if pd.notna(lower) and pd.notna(upper) and lower <= value <= upper: return " (Normal)"
    return ""

def _format_value(value: Any) -> str:
    """Formats numeric values to two decimal places and handles non-numeric types."""
    if not isinstance(value, (int, float)):
        return str(value)
    
    return f"{value:.2f}"

def _create_narrative_summary(patient_series: pd.Series, reference_ranges: pd.DataFrame, gender: str, zero_count_features: Set[str]) -> str:
    """Generates the concise, narrative summary for the F3 representation."""
    age_val = patient_series.get('age', 'N/A')
    gender_full = 'female' if gender == 'F' else 'male' if gender == 'M' else 'person'
    
    try:
        age_display = f"{float(age_val):.0f}"
    except (ValueError, TypeError):
        age_display = str(age_val)

    summary_lines = [f"This is a {age_display}-year-old {gender_full}."]
    summary_lines.append("\nKey abnormal findings from the first 24 hours include:")
    
    abnormal_findings = []
    interpretation_suffixes = ['mean_mean', 'mean_last', 'last', 'min_24h', 'max_24h', 'mean_min_24h', 'mean_max_24h']

    for feature, value in patient_series.items():
        if pd.isna(value): continue
        raw_base, display_base, suffix = _parse_feature_name(feature)
        
        if suffix in interpretation_suffixes and display_base not in zero_count_features:
            flag = _get_qualitative_flag(raw_base, value, reference_ranges, gender)
            if "High" in flag or "Low" in flag:
                label = FEATURE_LABEL_MAP.get(suffix, suffix.replace('_', ' ').title())
                formatted_val = _format_value(value)
                abnormal_findings.append(f"* {display_base}: The {label.lower()} was {formatted_val}{flag}.")

    if abnormal_findings:
        summary_lines.extend(abnormal_findings)
    else:
        summary_lines.append("* No significant abnormal findings noted in the non-imputed data.")
        
    return "\n".join(summary_lines)

def _generate_structured_representation(patient_series: pd.Series, representation_type: str, reference_ranges: pd.DataFrame, gender: str, zero_count_features: Set[str]) -> str:
    """Generates the full structured text for F1 or F2 representations."""
    feature_groups = defaultdict(list)
    
    # FIX: Pre-scan to find which groups have a zero 6-hour count
    zero_6h_count_features = set()
    for feature, value in patient_series.items():
        _, display_base, suffix = _parse_feature_name(feature)
        # Check for both simple and compound 6-hour count suffixes
        if suffix in ['count_6h', 'mean_count_6h'] and isinstance(value, (int, float)) and value < 1:
            zero_6h_count_features.add(display_base)

    for feature, value in patient_series.items():
        if pd.isna(value): continue
        raw_base, display_base, suffix = _parse_feature_name(feature)
        
        imputed_suffixes = ['mean_mean', 'mean_last', 'mean_stddev_24h', 'mean_slope_24h', 'mean_count']
        if suffix in imputed_suffixes and display_base in zero_count_features:
            continue

        label = FEATURE_LABEL_MAP.get(suffix) or suffix.replace('_', ' ').title()
        feature_groups[display_base].append((label, value, raw_base, suffix))

    full_text = []
    for display_base in sorted(feature_groups.keys()):
        unit = ""
        if not feature_groups[display_base]:
            continue
            
        first_feature_raw_base = feature_groups[display_base][0][2]
        lookup_key = f"{first_feature_raw_base}_mean"
        if lookup_key in reference_ranges.index:
            unit_val = reference_ranges.loc[lookup_key, 'Units']
            if pd.notna(unit_val) and str(unit_val).strip().upper() != 'N/A':
                unit = f" ({unit_val})"
        
        full_text.append(f"--- {display_base}{unit} ---")
        
        for label, value, raw_base, suffix in sorted(feature_groups[display_base], key=lambda x: x[0]):
            # FIX: Filter out 6h count and slope if the 6h count is zero for that group
            suffixes_to_filter = ['count_6h', 'slope_6h', 'mean_count_6h', 'mean_slope_6h']
            if suffix in suffixes_to_filter and display_base in zero_6h_count_features:
                continue
                
            flag = ""
            if representation_type == 'F2':
                interpretation_suffixes = ['mean_mean', 'mean_last', 'last', 'min_24h', 'max_24h', 'mean_min_24h', 'mean_max_24h']
                if suffix in interpretation_suffixes:
                    flag = _get_qualitative_flag(raw_base, value, reference_ranges, gender)
            
            full_text.append(f"{label}: {_format_value(value)}{flag}")
        full_text.append("")

    return "\n".join(full_text)

def generate_patient_representation(patient_series: pd.Series, representation_type: str, reference_ranges: pd.DataFrame, gender: str) -> str:
    """Main function to generate F1, F2, or the new F3 representation."""
    zero_count_features = {
        _parse_feature_name(feature)[1] for feature, value in patient_series.items() 
        if feature.endswith('_mean_count') and isinstance(value, (int, float)) and value < 1
    }

    if representation_type in ['F1', 'F2']:
        return _generate_structured_representation(patient_series, representation_type, reference_ranges, gender, zero_count_features)
    
    if representation_type == 'F3':
        return _create_narrative_summary(patient_series, reference_ranges, gender, zero_count_features)
    
    return ""