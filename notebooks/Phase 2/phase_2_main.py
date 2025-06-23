#!/usr/bin/env python3
# phase_2_main.py

"""
Phase II: Semantic Encoding & Classifier Analysis (Testing H3)

This script implements a comprehensive Phase II analysis that:
1. Loads Phase I engineered features 
2. Tests multiple serialization formats (Markdown, JSON, natural language)
3. Applies systematic prompt engineering (generic vs task-specific)
4. Generates embeddings using LLM APIs
5. Trains both Logistic Regression and XGBoost classifiers
6. Provides rigorous evaluation with uncertainty quantification

The goal is to systematically identify the best combination of serialization format,
prompt type, and classifier, while understanding WHY certain combinations perform better.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import time
import argparse
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase_2_serialization import get_available_formats, preview_serializations
from phase_2_prompts import get_available_prompts, get_core_prompt_set, preview_all_prompts
from phase_2_embeddings import run_embedding_experiment
from phase_2_classifiers import run_classifier_experiment, create_experiment_summary

# =============================================================================
# CONFIGURATION
# =============================================================================

# Phase II Experiment Configuration
PHASE_2_CONFIG = {
    # Data Configuration
    'phase_1_output_dir': 'phase_1_outputs',
    'phase_2_output_dir': 'phase_2_outputs',
    'use_sample': False,  # Set to True for quick testing
    'sample_size': 100,   # Number of patients for sampling
    
    # Serialization Configuration
    'serialization_formats': [
        'markdown_structured',
        'markdown_dense', 
        'json_structured',
        'json_flat',
        'natural_language'
    ],
    
    # Prompt Configuration  
    'prompt_names': [
        'generic_basic',
        'generic_detailed',
        'task_specific_mortality',
        'task_specific_mortality_detailed',
        'domain_expert_mortality_focused',
        'minimal_context',
        'minimal_task_specific'
    ],
    
    # Embedding Configuration
    'embedding_model': 'text-embedding-3-large',
    'use_parallel_embedding': False,
    'max_embedding_workers': 3,
    
    # Classifier Configuration
    'classifier_configs': [
        'logistic_regression',
        'logistic_regression_simple', 
        'xgboost',
        'xgboost_simple'
    ],
    
    # Evaluation Configuration
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    
    # Logging Configuration
    'log_level': logging.INFO,
    'save_intermediate_results': True
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str, log_level: int = logging.INFO):
    """Setup logging configuration for Phase II."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_filename = os.path.join(output_dir, 'phase_2_log.txt')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    return log_filename

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_phase_1_data(phase_1_dir: str) -> tuple:
    """
    Load Phase I engineered features and target data.
    
    Args:
        phase_1_dir: Directory containing Phase I outputs
        
    Returns:
        Tuple of (features_df, target_series)
    """
    logging.info(f"Loading Phase I data from: {phase_1_dir}")
    
    # Load engineered features
    features_path = os.path.join(phase_1_dir, 'X_test_engineered.csv')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Phase I features not found: {features_path}")
    
    features_df = pd.read_csv(features_path, index_col=0)
    logging.info(f"✓ Loaded features: {features_df.shape}")
    
    # Load target data
    target_path = os.path.join(phase_1_dir, 'y_test.csv')
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Phase I targets not found: {target_path}")
    
    target_series = pd.read_csv(target_path, index_col=0).iloc[:, 0]
    logging.info(f"✓ Loaded targets: {target_series.shape}")
    
    # Verify alignment
    if not features_df.index.equals(target_series.index):
        logging.warning("Misaligned indices between features and targets. Attempting to align...")
        common_indices = features_df.index.intersection(target_series.index)
        features_df = features_df.loc[common_indices]
        target_series = target_series.loc[common_indices]
        logging.info(f"✓ Aligned data: {len(common_indices)} samples")
    
    # Check class distribution
    class_dist = target_series.value_counts()
    mortality_rate = target_series.mean()
    logging.info(f"Target distribution: {dict(class_dist)} (mortality rate: {mortality_rate:.3f})")
    
    return features_df, target_series

def sample_data_if_requested(features_df: pd.DataFrame, target_series: pd.Series, 
                           use_sample: bool, sample_size: int, random_state: int = 42) -> tuple:
    """Sample data if requested for faster testing."""
    if not use_sample or sample_size >= len(features_df):
        return features_df, target_series
    
    logging.info(f"Sampling {sample_size} patients from {len(features_df)} total for testing")
    
    # Stratified sampling to maintain class balance
    from sklearn.model_selection import train_test_split
    
    _, sampled_features, _, sampled_targets = train_test_split(
        features_df, target_series, 
        test_size=sample_size,
        random_state=random_state,
        stratify=target_series
    )
    
    sampled_mortality_rate = sampled_targets.mean()
    logging.info(f"✓ Sampled data: {sampled_features.shape}, mortality rate: {sampled_mortality_rate:.3f}")
    
    return sampled_features, sampled_targets

# =============================================================================
# PHASE II PIPELINE FUNCTIONS
# =============================================================================

def run_serialization_preview(features_df: pd.DataFrame, output_dir: str):
    """Generate and save previews of different serialization formats."""
    logging.info("Generating serialization format previews...")
    
    # Select a representative patient for preview
    sample_patient = features_df.iloc[0]
    patient_id = sample_patient.name
    
    # Generate previews
    previews = preview_serializations(sample_patient, max_length=1000)
    
    # Save previews to files
    preview_dir = os.path.join(output_dir, 'serialization_previews')
    os.makedirs(preview_dir, exist_ok=True)
    
    for format_name, preview_text in previews.items():
        preview_file = os.path.join(preview_dir, f"preview_{format_name}.txt")
        with open(preview_file, 'w', encoding='utf-8') as f:
            f.write(f"# Serialization Preview: {format_name}\n")
            f.write(f"# Patient ID: {patient_id}\n\n")
            f.write(preview_text)
        
        logging.info(f"✓ Saved preview for {format_name}")

def run_prompt_preview(features_df: pd.DataFrame, output_dir: str):
    """Generate and save previews of different prompt templates."""
    logging.info("Generating prompt template previews...")
    
    # Select a representative patient and serialize with a standard format
    sample_patient = features_df.iloc[0]
    from phase_2_serialization import serialize_patient_data
    sample_serialized = serialize_patient_data(sample_patient, 'markdown_structured')
    
    # Generate prompt previews
    previews = preview_all_prompts(sample_serialized, max_length=800)
    
    # Save previews to files
    preview_dir = os.path.join(output_dir, 'prompt_previews')
    os.makedirs(preview_dir, exist_ok=True)
    
    for prompt_name, preview_text in previews.items():
        preview_file = os.path.join(preview_dir, f"prompt_{prompt_name}.txt")
        with open(preview_file, 'w', encoding='utf-8') as f:
            f.write(f"# Prompt Preview: {prompt_name}\n")
            f.write(f"# Patient ID: {sample_patient.name}\n\n")
            f.write(preview_text)
        
        logging.info(f"✓ Saved preview for {prompt_name}")

def validate_api_access():
    """Validate that OpenAI API access is available."""
    import openai
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.\n"
            "You can get an API key from: https://platform.openai.com/api-keys"
        )
    
    logging.info("✓ OpenAI API key found")
    
    # Test API access with a simple call
    try:
        openai.api_key = api_key
        # Make a minimal test call
        response = openai.Model.list()
        logging.info("✓ OpenAI API access validated")
    except Exception as e:
        raise ValueError(f"OpenAI API access failed: {str(e)}")

def run_phase_2_pipeline(config: dict):
    """Run the complete Phase II pipeline."""
    
    # Setup
    output_dir = config['phase_2_output_dir']
    log_filename = setup_logging(output_dir, config['log_level'])
    
    logging.info("="*60)
    logging.info("PHASE II: SEMANTIC ENCODING & CLASSIFIER ANALYSIS")
    logging.info("="*60)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Log file: {log_filename}")
    
    start_time = time.time()
    
    try:
        # Step 1: Load Phase I Data
        logging.info("\n" + "="*40)
        logging.info("STEP 1: LOADING PHASE I DATA")
        logging.info("="*40)
        
        features_df, target_series = load_phase_1_data(config['phase_1_output_dir'])
        
        # Sample data if requested
        features_df, target_series = sample_data_if_requested(
            features_df, target_series,
            config['use_sample'], config['sample_size'], config['random_state']
        )
        
        # Step 2: Generate Previews
        logging.info("\n" + "="*40)
        logging.info("STEP 2: GENERATING FORMAT PREVIEWS")
        logging.info("="*40)
        
        run_serialization_preview(features_df, output_dir)
        run_prompt_preview(features_df, output_dir)
        
        # Step 3: Validate API Access
        logging.info("\n" + "="*40)
        logging.info("STEP 3: VALIDATING API ACCESS")
        logging.info("="*40)
        
        validate_api_access()
        
        # Step 4: Generate Embeddings
        logging.info("\n" + "="*40)
        logging.info("STEP 4: GENERATING EMBEDDINGS")
        logging.info("="*40)
        
        embedding_output_dir = os.path.join(output_dir, 'embeddings')
        
        embedding_results = run_embedding_experiment(
            patient_data=features_df,
            target_data=target_series,
            serialization_formats=config['serialization_formats'],
            prompt_names=config['prompt_names'],
            output_dir=embedding_output_dir,
            model_name=config['embedding_model'],
            sample_size=None,  # Already sampled if requested
            use_parallel=config['use_parallel_embedding'],
            max_workers=config['max_embedding_workers']
        )
        
        # Step 5: Train and Evaluate Classifiers
        logging.info("\n" + "="*40)
        logging.info("STEP 5: TRAINING CLASSIFIERS")
        logging.info("="*40)
        
        classifier_output_dir = os.path.join(output_dir, 'classifiers')
        
        classifier_results = run_classifier_experiment(
            embeddings_results=embedding_results,
            target_data=target_series,
            classifier_configs=config['classifier_configs'],
            output_dir=classifier_output_dir,
            test_size=config['test_size'],
            val_size=config['val_size'],
            random_state=config['random_state']
        )
        
        # Step 6: Generate Final Analysis
        logging.info("\n" + "="*40)
        logging.info("STEP 6: GENERATING FINAL ANALYSIS")
        logging.info("="*40)
        
        analysis_results = create_comprehensive_analysis(
            embedding_results, classifier_results, output_dir
        )
        
        # Log completion
        total_time = time.time() - start_time
        logging.info("\n" + "="*40)
        logging.info("PHASE II ANALYSIS COMPLETE")
        logging.info("="*40)
        logging.info(f"Total execution time: {total_time/60:.2f} minutes")
        logging.info(f"Results saved to: {output_dir}")
        
        if analysis_results.get('best_combination'):
            best = analysis_results['best_combination']
            logging.info(f"Best performing combination:")
            logging.info(f"  Serialization: {best['serialization']}")
            logging.info(f"  Prompt: {best['prompt']}")
            logging.info(f"  Classifier: {best['classifier']}")
            logging.info(f"  ROC-AUC: {best['roc_auc']:.4f}")
            logging.info(f"  AUPRC: {best['auprc']:.4f}")
        
        return {
            'embedding_results': embedding_results,
            'classifier_results': classifier_results,
            'analysis_results': analysis_results,
            'execution_time': total_time,
            'output_dir': output_dir
        }
        
    except Exception as e:
        logging.error(f"Phase II pipeline failed: {str(e)}")
        raise

def create_comprehensive_analysis(embedding_results: dict, classifier_results: dict, 
                                output_dir: str) -> dict:
    """Create comprehensive analysis of Phase II results."""
    logging.info("Creating comprehensive analysis...")
    
    from phase_2_classifiers import compare_classifier_performance, plot_performance_comparison
    
    # Create performance comparison
    performance_df = compare_classifier_performance(classifier_results, 'roc_auc')
    
    if not performance_df.empty:
        # Save performance table
        perf_table_path = os.path.join(output_dir, 'performance_comparison.csv')
        performance_df.to_csv(perf_table_path, index=False)
        
        # Create visualization
        plot_path = os.path.join(output_dir, 'performance_heatmap_roc_auc.png')
        plot_performance_comparison(classifier_results, plot_path, 'roc_auc')
        
        plot_path_auprc = os.path.join(output_dir, 'performance_heatmap_auprc.png')
        plot_performance_comparison(classifier_results, plot_path_auprc, 'auprc')
        
        # Identify best combination
        best_idx = performance_df['roc_auc_point_estimate'].idxmax()
        best_row = performance_df.iloc[best_idx]
        
        # Parse combination name
        combo_parts = best_row['combination'].split('_')
        
        best_combination = {
            'combination': best_row['combination'],
            'serialization': '_'.join(combo_parts[:-2]),  # Assuming last 2 parts are prompt
            'prompt': '_'.join(combo_parts[-2:]),
            'classifier': best_row['classifier'],
            'roc_auc': best_row['roc_auc_point_estimate'],
            'roc_auc_ci_lower': best_row['roc_auc_ci_lower'],
            'roc_auc_ci_upper': best_row['roc_auc_ci_upper'],
            'auprc': performance_df[performance_df.index == best_idx]['auprc_point_estimate'].iloc[0] 
                    if 'auprc_point_estimate' in performance_df.columns else None
        }
        
        # Create analysis summary
        analysis_summary = {
            'best_combination': best_combination,
            'total_combinations_tested': len(performance_df),
            'serialization_formats_tested': performance_df['combination'].apply(
                lambda x: '_'.join(x.split('_')[:-2])
            ).nunique(),
            'prompt_types_tested': performance_df['combination'].apply(
                lambda x: '_'.join(x.split('_')[-2:])
            ).nunique(),
            'classifiers_tested': performance_df['classifier'].nunique(),
            'performance_statistics': {
                'roc_auc_mean': performance_df['roc_auc_point_estimate'].mean(),
                'roc_auc_std': performance_df['roc_auc_point_estimate'].std(),
                'roc_auc_range': [
                    performance_df['roc_auc_point_estimate'].min(),
                    performance_df['roc_auc_point_estimate'].max()
                ]
            }
        }
        
        # Save analysis summary
        summary_path = os.path.join(output_dir, 'comprehensive_analysis.json')
        with open(summary_path, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        
        logging.info(f"✓ Comprehensive analysis saved to {summary_path}")
        
        return analysis_summary
    
    else:
        logging.warning("No performance data available for analysis")
        return {'error': 'No performance data available'}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run Phase II analysis."""
    
    parser = argparse.ArgumentParser(description='Phase II: Semantic Encoding & Classifier Analysis')
    parser.add_argument('--config', type=str, help='Path to custom configuration JSON file')
    parser.add_argument('--sample', action='store_true', help='Use sample data for quick testing')
    parser.add_argument('--sample-size', type=int, default=100, help='Sample size for testing')
    parser.add_argument('--output-dir', type=str, default='phase_2_outputs', help='Output directory')
    parser.add_argument('--phase-1-dir', type=str, default='phase_1_outputs', help='Phase I output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = PHASE_2_CONFIG.copy()
    
    # Apply command line overrides
    if args.sample:
        config['use_sample'] = True
        config['sample_size'] = args.sample_size
    
    if args.output_dir:
        config['phase_2_output_dir'] = args.output_dir
    
    if args.phase_1_dir:
        config['phase_1_output_dir'] = args.phase_1_dir
    
    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        config.update(custom_config)
    
    # Run the pipeline
    try:
        results = run_phase_2_pipeline(config)
        print(f"\n✓ Phase II analysis completed successfully!")
        print(f"Results saved to: {results['output_dir']}")
        return 0
        
    except Exception as e:
        print(f"\n✗ Phase II analysis failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 