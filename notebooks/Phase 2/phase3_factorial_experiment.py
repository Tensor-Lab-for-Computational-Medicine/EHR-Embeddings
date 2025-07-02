# phase3_factorial_experiment.py
"""
Main execution script for Phase III factorial experiment.
Orchestrates the 2x8 factorial evaluation of semantic representation and prompting strategy.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

from phase3_components import (
    ReferenceRangeManager, 
    FormatGenerator, 
    get_all_format_ids,
    get_format_description
)
from phase3_prompts import (
    PromptManager,
    get_experimental_conditions,
    get_all_prompt_ids
)


class Phase3ExperimentManager:
    """Manages the complete Phase III factorial experiment"""
    
    def __init__(self, 
                 reference_ranges_path: str = "data/Lab_reference_ranges.csv",
                 output_dir: str = "notebooks/Phase 2/phase3_outputs",
                 log_level: str = "INFO"):
        """
        Initialize the experiment manager.
        
        Args:
            reference_ranges_path: Path to the reference ranges CSV file
            output_dir: Directory to save experiment outputs
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.reference_ranges_path = reference_ranges_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Initialize components
        self.logger.info("Initializing Phase III components...")
        self.reference_manager = ReferenceRangeManager(reference_ranges_path)
        self.format_generator = FormatGenerator(self.reference_manager)
        self.prompt_manager = PromptManager()
        
        # Get experimental conditions
        self.conditions = get_experimental_conditions()
        self.logger.info(f"Initialized {len(self.conditions)} experimental conditions")
        
    def _setup_logging(self, log_level: str):
        """Setup logging configuration"""
        log_file = self.output_dir / f"phase3_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def process_single_patient(self, 
                              patient_data: pd.Series, 
                              patient_id: str,
                              gender: str = 'unknown',
                              max_decimals: int = 3) -> Dict[str, str]:
        """
        Process a single patient through all 16 experimental conditions.
        
        Args:
            patient_data: Series with feature values for the patient
            patient_id: Unique identifier for the patient
            gender: Patient gender ('male', 'female', or 'unknown')
            max_decimals: Number of decimal places for numerical formatting
            
        Returns:
            Dictionary mapping condition names to complete prompts
        """
        results = {}
        
        self.logger.debug(f"Processing patient {patient_id} with {len(patient_data)} features")
        
        for condition in self.conditions:
            try:
                # Generate the serialized format
                serialized_data = self.format_generator.generate_format(
                    patient_data=patient_data,
                    format_id=condition['format_id'],
                    max_decimals=max_decimals,
                    gender=gender
                )
                
                # Combine with prompt
                full_prompt = self.prompt_manager.get_full_prompt(
                    prompt_id=condition['prompt_id'],
                    serialized_data=serialized_data
                )
                
                results[condition['condition_name']] = full_prompt
                
            except Exception as e:
                self.logger.error(f"Error processing patient {patient_id} in condition {condition['condition_name']}: {str(e)}")
                results[condition['condition_name']] = f"ERROR: {str(e)}"
        
        return results
    
    def process_dataset(self, 
                       data: pd.DataFrame,
                       patient_id_column: str = 'patient_id',
                       gender_column: Optional[str] = None,
                       output_format: str = 'individual_files',
                       max_decimals: int = 3) -> Dict[str, Any]:
        """
        Process entire dataset through all experimental conditions.
        
        Args:
            data: DataFrame with patient data (patients as rows, features as columns)
            patient_id_column: Name of column containing patient IDs
            gender_column: Name of column containing gender information (optional)
            output_format: 'individual_files', 'combined_file', or 'memory_only'
            max_decimals: Number of decimal places for numerical formatting
            
        Returns:
            Dictionary with experiment results and metadata
        """
        self.logger.info(f"Processing dataset with {len(data)} patients through {len(self.conditions)} conditions")
        
        # Prepare output structure
        experiment_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_patients': len(data),
                'num_conditions': len(self.conditions),
                'conditions': self.conditions,
                'reference_ranges_file': self.reference_ranges_path,
                'max_decimals': max_decimals
            },
            'results': {}
        }
        
        # Initialize results structure for each condition
        for condition in self.conditions:
            experiment_results['results'][condition['condition_name']] = {}
        
        # Process each patient
        for idx, (_, patient_row) in enumerate(data.iterrows()):
            if idx % 100 == 0:
                self.logger.info(f"Processing patient {idx + 1}/{len(data)}")
            
            # Extract patient info
            patient_id = str(patient_row[patient_id_column])
            gender = 'unknown'
            if gender_column and gender_column in patient_row:
                gender = str(patient_row[gender_column]).lower()
            
            # Extract feature data (exclude metadata columns)
            feature_columns = [col for col in data.columns if col not in [patient_id_column, gender_column]]
            patient_features = patient_row[feature_columns]
            
            # Process patient through all conditions
            patient_results = self.process_single_patient(
                patient_data=patient_features,
                patient_id=patient_id,
                gender=gender,
                max_decimals=max_decimals
            )
            
            # Store results
            for condition_name, prompt_text in patient_results.items():
                experiment_results['results'][condition_name][patient_id] = prompt_text
        
        # Save results based on output format
        if output_format == 'individual_files':
            self._save_individual_files(experiment_results)
        elif output_format == 'combined_file':
            self._save_combined_file(experiment_results)
        
        self.logger.info("Dataset processing completed")
        return experiment_results
    
    def _save_individual_files(self, experiment_results: Dict[str, Any]):
        """Save results as individual files for each condition"""
        self.logger.info("Saving results as individual files...")
        
        for condition_name, patient_prompts in experiment_results['results'].items():
            condition_dir = self.output_dir / condition_name
            condition_dir.mkdir(exist_ok=True)
            
            # Save each patient's prompt
            for patient_id, prompt_text in patient_prompts.items():
                prompt_file = condition_dir / f"{patient_id}.txt"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(prompt_text)
            
            # Save condition metadata
            condition_info = {
                'condition_name': condition_name,
                'prompt_id': next(c['prompt_id'] for c in self.conditions if c['condition_name'] == condition_name),
                'format_id': next(c['format_id'] for c in self.conditions if c['condition_name'] == condition_name),
                'num_patients': len(patient_prompts),
                'timestamp': experiment_results['metadata']['timestamp']
            }
            
            metadata_file = condition_dir / "condition_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(condition_info, f, indent=2)
        
        # Save overall metadata
        metadata_file = self.output_dir / "experiment_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results['metadata'], f, indent=2)
    
    def _save_combined_file(self, experiment_results: Dict[str, Any]):
        """Save all results in a single compressed file"""
        self.logger.info("Saving results as combined file...")
        
        output_file = self.output_dir / f"phase3_complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def generate_sample_outputs(self, sample_patient_data: pd.Series, 
                               patient_id: str = "sample_patient",
                               gender: str = "unknown") -> None:
        """Generate sample outputs for documentation and verification"""
        self.logger.info("Generating sample outputs for documentation...")
        
        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        # Process sample patient
        sample_results = self.process_single_patient(
            patient_data=sample_patient_data,
            patient_id=patient_id,
            gender=gender
        )
        
        # Save each condition as a separate sample file
        for condition_name, prompt_text in sample_results.items():
            sample_file = sample_dir / f"sample_{condition_name}.txt"
            
            # Add header with condition description
            condition = next(c for c in self.conditions if c['condition_name'] == condition_name)
            prompt_desc = self.prompt_manager.get_prompt_description(condition['prompt_id'])
            format_desc = get_format_description(condition['format_id'])
            
            header = f"""
# Phase III Sample Output
# Condition: {condition_name}
# Prompt Strategy: {condition['prompt_id']} - {prompt_desc}
# Format: {condition['format_id']} - {format_desc}
# Generated: {datetime.now().isoformat()}

{'=' * 80}

"""
            
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(header + prompt_text)
        
        self.logger.info(f"Sample outputs saved to {sample_dir}")
    
    def print_experiment_summary(self):
        """Print a comprehensive summary of the experimental design"""
        print("\n" + "=" * 80)
        print("PHASE III FACTORIAL EXPERIMENT SUMMARY")
        print("=" * 80)
        
        print(f"\nExperimental Design:")
        print(f"- Factors: 2 (Prompts) × 8 (Formats) = 16 total conditions")
        print(f"- Reference ranges file: {self.reference_ranges_path}")
        print(f"- Output directory: {self.output_dir}")
        
        print(f"\nPrompting Strategies ({len(get_all_prompt_ids())}):")
        for prompt_id in get_all_prompt_ids():
            strategy = self.prompt_manager.strategies[prompt_id]
            print(f"  {prompt_id}: {strategy.name}")
            print(f"     {strategy.description}")
        
        print(f"\nFormat Combinations ({len(get_all_format_ids())}):")
        for format_id in get_all_format_ids():
            print(f"  {format_id}: {get_format_description(format_id)}")
        
        print(f"\nExperimental Conditions ({len(self.conditions)}):")
        for condition in self.conditions:
            print(f"  {condition['condition_name']}: "
                  f"Prompt {condition['prompt_id']} × Format {condition['format_id']}")
        
        print("\n" + "=" * 80)


def main():
    """Main function for standalone execution"""
    print("Phase III Factorial Experiment Manager")
    print("=" * 50)
    
    # Initialize experiment manager
    manager = Phase3ExperimentManager()
    
    # Print experiment summary
    manager.print_experiment_summary()
    
    print("\nExperiment manager initialized successfully!")
    print("Use this manager to process your dataset through all experimental conditions.")
    
    return manager


if __name__ == "__main__":
    main() 