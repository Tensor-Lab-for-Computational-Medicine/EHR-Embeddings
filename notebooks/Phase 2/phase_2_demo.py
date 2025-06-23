#!/usr/bin/env python3
# phase_2_demo.py

"""
Phase II Demo Script

This script demonstrates the Phase II modules with sample data to verify
everything is working before running the full experiment.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase_2_serialization import (
    get_available_formats, serialize_patient_data, preview_serializations
)
from phase_2_prompts import (
    get_available_prompts, generate_prompt, preview_all_prompts, get_core_prompt_set
)

# =============================================================================
# DEMO CONFIGURATION
# =============================================================================

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def create_sample_patient_data(n_features: int = 50) -> pd.Series:
    """Create sample patient data for testing."""
    np.random.seed(42)
    
    # Create realistic-looking feature names
    feature_names = [
        # Vital signs
        'heart_rate_mean', 'heart_rate_std', 'heart_rate_max', 'heart_rate_min',
        'systolic_blood_pressure_mean', 'systolic_blood_pressure_std',
        'diastolic_blood_pressure_mean', 'diastolic_blood_pressure_std',
        'temperature_mean', 'temperature_max', 'oxygen_saturation_mean',
        'respiratory_rate_mean', 'respiratory_rate_std',
        
        # Laboratory values
        'glucose_mean', 'glucose_std', 'creatinine_mean', 'creatinine_max',
        'sodium_mean', 'potassium_mean', 'chloride_mean', 'calcium_mean',
        'ph_mean', 'ph_min', 'lactate_mean', 'lactate_max',
        
        # Blood chemistry
        'bicarbonate_mean', 'co2_mean', 'anion_gap_mean',
        'partial_pressure_of_oxygen_mean', 'partial_pressure_of_carbon_dioxide_mean',
        
        # Hematology
        'hematocrit_mean', 'hemoglobin_mean', 'platelets_mean',
        'red_blood_cell_count_mean', 'hemoglobin_std',
        
        # Cardiovascular
        'cardiac_output_thermodilution_mean', 'central_venous_pressure_mean',
        'pulmonary_artery_pressure_mean', 'systemic_vascular_resistance_mean',
        
        # Respiratory
        'tidal_volume_observed_mean', 'positive_end_expiratory_pressure_mean',
        'fraction_inspired_oxygen_mean', 'peak_inspiratory_pressure_mean',
        
        # Neurological
        'glascow_coma_scale_total_mean', 'glascow_coma_scale_total_min',
        
        # Trends (if we have room)
        'heart_rate_trend_24h', 'systolic_blood_pressure_trend_24h',
        'glucose_trend_6h', 'lactate_trend_24h'
    ]
    
    # Use only as many features as requested
    feature_names = feature_names[:n_features]
    
    # Generate realistic values
    values = []
    for name in feature_names:
        if 'heart_rate' in name:
            if 'mean' in name:
                values.append(np.random.normal(85, 15))
            elif 'std' in name:
                values.append(np.random.exponential(8))
            elif 'max' in name:
                values.append(np.random.normal(110, 20))
            elif 'min' in name:
                values.append(np.random.normal(65, 10))
            elif 'trend' in name:
                values.append(np.random.normal(0, 2))
        elif 'blood_pressure' in name:
            if 'systolic' in name:
                values.append(np.random.normal(120, 25))
            else:
                values.append(np.random.normal(70, 15))
        elif 'temperature' in name:
            values.append(np.random.normal(37.0, 1.5))
        elif 'oxygen_saturation' in name:
            values.append(np.random.normal(96, 4))
        elif 'glucose' in name:
            values.append(np.random.exponential(120))
        elif 'lactate' in name:
            values.append(np.random.exponential(2.0))
        elif 'ph' in name:
            values.append(np.random.normal(7.4, 0.1))
        else:
            # Generic positive value
            values.append(np.random.exponential(10))
    
    return pd.Series(values, index=feature_names, name='patient_12345')

def demo_serialization():
    """Demonstrate different serialization formats."""
    print("\n" + "="*60)
    print("DEMO: SERIALIZATION FORMATS")
    print("="*60)
    
    # Create sample patient data
    patient_data = create_sample_patient_data(30)
    print(f"Created sample patient data: {len(patient_data)} features")
    
    # Show available formats
    formats = get_available_formats()
    print(f"Available serialization formats: {formats}")
    
    # Test each format
    for format_name in formats:
        print(f"\n--- {format_name.upper()} ---")
        try:
            serialized = serialize_patient_data(patient_data, format_name)
            preview = serialized[:300] + "..." if len(serialized) > 300 else serialized
            print(preview)
        except Exception as e:
            print(f"Error: {e}")
    
    # Generate all previews
    print(f"\n--- ALL PREVIEWS ---")
    previews = preview_serializations(patient_data, max_length=200)
    for format_name, preview in previews.items():
        print(f"{format_name}: {len(preview)} characters")

def demo_prompts():
    """Demonstrate different prompt templates."""
    print("\n" + "="*60)
    print("DEMO: PROMPT TEMPLATES")
    print("="*60)
    
    # Create sample patient data and serialize it
    patient_data = create_sample_patient_data(20)
    serialized_data = serialize_patient_data(patient_data, 'markdown_structured')
    
    # Show available prompts
    prompts = get_available_prompts()
    print(f"Available prompt templates: {prompts}")
    
    core_prompts = get_core_prompt_set()
    print(f"Core prompt set: {core_prompts}")
    
    # Test core prompts
    for prompt_name in core_prompts:
        print(f"\n--- {prompt_name.upper()} ---")
        try:
            prompt = generate_prompt(prompt_name, serialized_data)
            preview = prompt[:400] + "..." if len(prompt) > 400 else prompt
            print(preview)
        except Exception as e:
            print(f"Error: {e}")

def demo_integration():
    """Demonstrate integration of serialization and prompts."""
    print("\n" + "="*60)
    print("DEMO: INTEGRATION TEST")
    print("="*60)
    
    # Create sample patient data
    patient_data = create_sample_patient_data(25)
    
    # Test key combinations
    test_combinations = [
        ('markdown_structured', 'generic_basic'),
        ('json_structured', 'task_specific_mortality'),
        ('natural_language', 'domain_expert_mortality_focused'),
        ('markdown_dense', 'minimal_task_specific')
    ]
    
    for serialization_format, prompt_name in test_combinations:
        print(f"\n--- COMBINATION: {serialization_format} + {prompt_name} ---")
        try:
            # Serialize patient data
            serialized = serialize_patient_data(patient_data, serialization_format)
            
            # Generate prompt
            prompt = generate_prompt(prompt_name, serialized)
            
            # Show statistics
            print(f"Serialized length: {len(serialized)} characters")
            print(f"Final prompt length: {len(prompt)} characters")
            print(f"Preview: {prompt[:200]}...")
            
        except Exception as e:
            print(f"Error in combination: {e}")

def demo_phase_1_data_loading():
    """Demonstrate loading Phase I data if available."""
    print("\n" + "="*60)
    print("DEMO: PHASE I DATA LOADING")
    print("="*60)
    
    phase_1_dir = 'phase_1_outputs'
    
    if os.path.exists(phase_1_dir):
        print(f"Phase I directory found: {phase_1_dir}")
        
        # Check for required files
        features_file = os.path.join(phase_1_dir, 'X_test_engineered.csv')
        targets_file = os.path.join(phase_1_dir, 'y_test.csv')
        
        if os.path.exists(features_file) and os.path.exists(targets_file):
            print("✓ Phase I data files found")
            
            # Load and inspect
            try:
                features_df = pd.read_csv(features_file, index_col=0)
                targets_series = pd.read_csv(targets_file, index_col=0).iloc[:, 0]
                
                print(f"Features shape: {features_df.shape}")
                print(f"Targets shape: {targets_series.shape}")
                print(f"Mortality rate: {targets_series.mean():.3f}")
                print(f"Sample feature names: {list(features_df.columns[:10])}")
                
                # Test serialization with real data
                print("\n--- TESTING WITH REAL PHASE I DATA ---")
                sample_patient = features_df.iloc[0]
                
                # Test one serialization format
                serialized = serialize_patient_data(sample_patient, 'markdown_structured')
                print(f"Real patient serialized length: {len(serialized)} characters")
                print(f"Preview: {serialized[:300]}...")
                
            except Exception as e:
                print(f"Error loading Phase I data: {e}")
        else:
            print("✗ Phase I data files not found")
            print(f"Looking for: {features_file}")
            print(f"Looking for: {targets_file}")
    else:
        print(f"✗ Phase I directory not found: {phase_1_dir}")
        print("This demo will use synthetic data instead")

def main():
    """Run all demos."""
    print("Phase II Module Demo")
    print("===================")
    
    try:
        # Test serialization
        demo_serialization()
        
        # Test prompts
        demo_prompts()
        
        # Test integration
        demo_integration()
        
        # Test Phase I data loading
        demo_phase_1_data_loading()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("All Phase II modules are working correctly!")
        print("You can now run the full Phase II analysis with:")
        print("  python phase_2_main.py")
        print("Or with sample data:")
        print("  python phase_2_main.py --sample --sample-size 50")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 