# phase3_prompts.py
"""
Prompting strategies for Phase III factorial evaluation.
Implements the two distinct prompting approaches: Task-Specific and Persona-Driven.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PromptStrategy:
    """Structure to hold prompt information"""
    prompt_id: str
    name: str
    text: str
    description: str


class PromptManager:
    """Manages the two prompting strategies for factorial evaluation"""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[str, PromptStrategy]:
        """Initialize the two prompting strategies"""
        strategies = {}
        
        # Prompt A: Task-Specific Predictive
        strategies['A'] = PromptStrategy(
            prompt_id='A',
            name='Task-Specific Predictive',
            text='Generate an embedding specifically for predicting in-hospital mortality from the following patient data.',
            description='Direct, concise, and task-aligned prompt establishing clear predictive objective without additional cognitive scaffolding.'
        )
        
        # Prompt B: Persona-Driven Diagnostic
        strategies['B'] = PromptStrategy(
            prompt_id='B',
            name='Persona-Driven Diagnostic',
            text='You are an experienced ICU physician. Analyze the following data to identify the primary drivers of clinical risk. Generate an embedding that captures the patient\'s overall severity. First, identify the most abnormal values relative to normal ranges. Second, consider the trends and volatility of these values. Finally, synthesize these factors into the embedding.',
            description='Advanced prompt with expert persona and Chain-of-Thought reasoning structure to enable more nuanced embedding generation.'
        )
        
        return strategies
    
    def get_prompt(self, prompt_id: str) -> str:
        """Get the prompt text for a specific strategy"""
        if prompt_id not in self.strategies:
            raise ValueError(f"Unknown prompt ID: {prompt_id}. Use 'A' or 'B'")
        return self.strategies[prompt_id].text
    
    def get_prompt_description(self, prompt_id: str) -> str:
        """Get the description for a specific strategy"""
        if prompt_id not in self.strategies:
            raise ValueError(f"Unknown prompt ID: {prompt_id}. Use 'A' or 'B'")
        return self.strategies[prompt_id].description
    
    def get_full_prompt(self, prompt_id: str, serialized_data: str) -> str:
        """Combine prompt with serialized patient data"""
        prompt_text = self.get_prompt(prompt_id)
        return f"{prompt_text}\n\n{serialized_data}"
    
    def get_all_prompt_ids(self) -> List[str]:
        """Return list of all available prompt IDs"""
        return list(self.strategies.keys())
    
    def get_strategy_summary(self) -> Dict[str, Dict[str, str]]:
        """Get summary of all strategies for reporting"""
        summary = {}
        for prompt_id, strategy in self.strategies.items():
            summary[prompt_id] = {
                'name': strategy.name,
                'description': strategy.description,
                'text': strategy.text
            }
        return summary


def get_all_prompt_ids() -> List[str]:
    """Convenience function to get all prompt IDs"""
    return ['A', 'B']


def get_experimental_conditions() -> List[Dict[str, str]]:
    """
    Generate all 16 experimental conditions (2 prompts × 8 formats).
    
    Returns:
        List of dictionaries with 'prompt_id', 'format_id', and 'condition_name'
    """
    from phase3_components import get_all_format_ids
    
    conditions = []
    prompt_ids = get_all_prompt_ids()
    format_ids = get_all_format_ids()
    
    for prompt_id in prompt_ids:
        for format_id in format_ids:
            condition_name = f"P{prompt_id}_F{format_id}"
            conditions.append({
                'prompt_id': prompt_id,
                'format_id': format_id,
                'condition_name': condition_name
            })
    
    return conditions


def print_experimental_design():
    """Print a summary of the factorial experimental design"""
    from phase3_components import get_format_description
    
    prompt_manager = PromptManager()
    conditions = get_experimental_conditions()
    
    print("Phase III Factorial Experimental Design")
    print("=" * 50)
    print(f"Total Conditions: {len(conditions)}")
    print()
    
    print("Prompting Strategies:")
    print("-" * 20)
    for prompt_id in ['A', 'B']:
        strategy = prompt_manager.strategies[prompt_id]
        print(f"Prompt {prompt_id}: {strategy.name}")
        print(f"  Description: {strategy.description}")
        print(f"  Text: {strategy.text}")
        print()
    
    print("Format Combinations:")
    print("-" * 20)
    from phase3_components import get_all_format_ids
    for format_id in get_all_format_ids():
        print(f"Format {format_id}: {get_format_description(format_id)}")
    print()
    
    print("Experimental Conditions Matrix:")
    print("-" * 30)
    print("Condition\tPrompt\tFormat\tDescription")
    for condition in conditions:
        prompt_desc = prompt_manager.strategies[condition['prompt_id']].name
        format_desc = get_format_description(condition['format_id'])
        print(f"{condition['condition_name']}\t{prompt_desc}\t{format_desc}")


if __name__ == "__main__":
    print_experimental_design() 