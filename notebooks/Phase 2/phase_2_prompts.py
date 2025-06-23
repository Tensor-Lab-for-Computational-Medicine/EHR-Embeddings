# phase_2_prompts.py

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================

class PromptType(Enum):
    """Enumeration of different prompt types for systematic testing."""
    GENERIC = "generic"
    TASK_SPECIFIC = "task_specific"
    DOMAIN_EXPERT = "domain_expert"
    MINIMAL = "minimal"

@dataclass
class PromptConfig:
    """Configuration for a specific prompt template."""
    name: str
    prompt_type: PromptType
    template: str
    description: str
    includes_task_context: bool
    includes_domain_knowledge: bool
    instruction_style: str  # 'direct', 'conversational', 'clinical'

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATES = {
    'generic_basic': PromptConfig(
        name='generic_basic',
        prompt_type=PromptType.GENERIC,
        template="""Generate a medical summary embedding for the following patient data:

{patient_data}

Please create a comprehensive embedding that captures the key clinical information.""",
        description='Basic generic prompt without task-specific context',
        includes_task_context=False,
        includes_domain_knowledge=False,
        instruction_style='direct'
    ),
    
    'generic_detailed': PromptConfig(
        name='generic_detailed',
        prompt_type=PromptType.GENERIC,
        template="""You are tasked with creating a semantic embedding that captures the essential clinical characteristics of an ICU patient. 

Patient Data:
{patient_data}

Please generate an embedding that:
1. Captures the patient's overall clinical state
2. Preserves important physiological patterns
3. Maintains relevant temporal information
4. Represents the complexity of the clinical picture

Focus on creating a rich, informative representation suitable for medical analysis.""",
        description='Detailed generic prompt with structured instructions',
        includes_task_context=False,
        includes_domain_knowledge=True,
        instruction_style='conversational'
    ),
    
    'task_specific_mortality': PromptConfig(
        name='task_specific_mortality',
        prompt_type=PromptType.TASK_SPECIFIC,
        template="""Generate an embedding to predict in-hospital mortality based on the following patient data:

{patient_data}

Create an embedding that specifically captures clinical features and patterns most relevant for predicting mortality risk in ICU patients. Focus on critical physiological indicators, organ dysfunction markers, and clinical instability patterns.""",
        description='Task-specific prompt for mortality prediction',
        includes_task_context=True,
        includes_domain_knowledge=True,
        instruction_style='direct'
    ),
    
    'task_specific_mortality_detailed': PromptConfig(
        name='task_specific_mortality_detailed',
        prompt_type=PromptType.TASK_SPECIFIC,
        template="""Your task is to create an embedding optimized for predicting in-hospital mortality in ICU patients.

Patient Clinical Data:
{patient_data}

Generate an embedding that prioritizes:
- Hemodynamic instability indicators (blood pressure variability, cardiac output)
- Respiratory failure markers (oxygenation, ventilatory support)
- Organ dysfunction signs (renal, hepatic, neurological)
- Metabolic derangements (lactate, pH, glucose control)
- Clinical trajectory and trends over time

The embedding should be specifically tuned to distinguish between patients at high vs. low mortality risk.""",
        description='Detailed task-specific prompt with clinical priorities',
        includes_task_context=True,
        includes_domain_knowledge=True,
        instruction_style='clinical'
    ),
    
    'domain_expert_intensivist': PromptConfig(
        name='domain_expert_intensivist',
        prompt_type=PromptType.DOMAIN_EXPERT,
        template="""As an experienced ICU physician, analyze the following patient data and create an embedding that represents your clinical assessment:

{patient_data}

Drawing on your expertise in critical care medicine, create an embedding that captures:
- The patient's severity of illness and physiological reserve
- Signs of single or multi-organ dysfunction
- Response to ongoing interventions
- Clinical patterns that inform prognosis

Your embedding should reflect the nuanced clinical reasoning that guides ICU decision-making.""",
        description='Domain expert prompt simulating intensivist perspective',
        includes_task_context=False,
        includes_domain_knowledge=True,
        instruction_style='clinical'
    ),
    
    'domain_expert_mortality_focused': PromptConfig(
        name='domain_expert_mortality_focused',
        prompt_type=PromptType.DOMAIN_EXPERT,
        template="""As a critical care specialist with expertise in mortality prediction, analyze this ICU patient data:

{patient_data}

Create an embedding that captures your clinical assessment for mortality risk prediction. Consider:
- Severity scoring implications (APACHE, SOFA conceptual frameworks)
- Physiological reserve and failure patterns
- Trajectory of clinical parameters
- Known mortality risk factors in critical care

Your embedding should embody the clinical expertise used in mortality prognostication.""",
        description='Domain expert prompt focused on mortality prediction expertise',
        includes_task_context=True,
        includes_domain_knowledge=True,
        instruction_style='clinical'
    ),
    
    'minimal_context': PromptConfig(
        name='minimal_context',
        prompt_type=PromptType.MINIMAL,
        template="""Patient data:
{patient_data}

Generate embedding.""",
        description='Minimal prompt to test baseline performance',
        includes_task_context=False,
        includes_domain_knowledge=False,
        instruction_style='direct'
    ),
    
    'minimal_task_specific': PromptConfig(
        name='minimal_task_specific',
        prompt_type=PromptType.MINIMAL,
        template="""Predict mortality from:
{patient_data}

Generate embedding.""",
        description='Minimal task-specific prompt',
        includes_task_context=True,
        includes_domain_knowledge=False,
        instruction_style='direct'
    )
}

# =============================================================================
# PROMPT GENERATION FUNCTIONS
# =============================================================================

def generate_prompt(prompt_name: str, patient_data: str) -> str:
    """
    Generate a complete prompt using the specified template and patient data.
    
    Args:
        prompt_name: Name of the prompt template to use
        patient_data: Serialized patient data string
        
    Returns:
        Complete prompt string ready for LLM input
        
    Raises:
        ValueError: If prompt_name is not recognized
    """
    if prompt_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {prompt_name}. "
                        f"Available templates: {list(PROMPT_TEMPLATES.keys())}")
    
    config = PROMPT_TEMPLATES[prompt_name]
    return config.template.format(patient_data=patient_data)

def get_available_prompts() -> List[str]:
    """Return list of available prompt template names."""
    return list(PROMPT_TEMPLATES.keys())

def get_prompts_by_type(prompt_type: PromptType) -> List[str]:
    """
    Get all prompt names of a specific type.
    
    Args:
        prompt_type: Type of prompts to retrieve
        
    Returns:
        List of prompt names matching the specified type
    """
    return [name for name, config in PROMPT_TEMPLATES.items() 
            if config.prompt_type == prompt_type]

def get_prompt_config(prompt_name: str) -> PromptConfig:
    """
    Get the configuration for a specific prompt.
    
    Args:
        prompt_name: Name of the prompt template
        
    Returns:
        PromptConfig object
        
    Raises:
        ValueError: If prompt_name is not recognized
    """
    if prompt_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {prompt_name}")
    
    return PROMPT_TEMPLATES[prompt_name]

# =============================================================================
# PROMPT ANALYSIS FUNCTIONS
# =============================================================================

def analyze_prompt_characteristics() -> Dict[str, Any]:
    """
    Analyze characteristics of all available prompts.
    
    Returns:
        Dictionary with analysis of prompt characteristics
    """
    analysis = {
        'total_prompts': len(PROMPT_TEMPLATES),
        'by_type': {},
        'by_task_context': {'with_context': 0, 'without_context': 0},
        'by_domain_knowledge': {'with_domain': 0, 'without_domain': 0},
        'by_instruction_style': {},
        'prompt_details': []
    }
    
    # Count by type
    for prompt_type in PromptType:
        analysis['by_type'][prompt_type.value] = len(get_prompts_by_type(prompt_type))
    
    # Analyze each prompt
    for name, config in PROMPT_TEMPLATES.items():
        # Task context
        if config.includes_task_context:
            analysis['by_task_context']['with_context'] += 1
        else:
            analysis['by_task_context']['without_context'] += 1
        
        # Domain knowledge
        if config.includes_domain_knowledge:
            analysis['by_domain_knowledge']['with_domain'] += 1
        else:
            analysis['by_domain_knowledge']['without_domain'] += 1
        
        # Instruction style
        style = config.instruction_style
        if style not in analysis['by_instruction_style']:
            analysis['by_instruction_style'][style] = 0
        analysis['by_instruction_style'][style] += 1
        
        # Detailed info
        analysis['prompt_details'].append({
            'name': name,
            'type': config.prompt_type.value,
            'has_task_context': config.includes_task_context,
            'has_domain_knowledge': config.includes_domain_knowledge,
            'instruction_style': config.instruction_style,
            'description': config.description
        })
    
    return analysis

def create_prompt_comparison_matrix() -> Dict[str, Dict[str, bool]]:
    """
    Create a comparison matrix of prompt characteristics.
    
    Returns:
        Dictionary mapping prompt names to their characteristics
    """
    matrix = {}
    
    for name, config in PROMPT_TEMPLATES.items():
        matrix[name] = {
            'is_generic': config.prompt_type == PromptType.GENERIC,
            'is_task_specific': config.prompt_type == PromptType.TASK_SPECIFIC,
            'is_domain_expert': config.prompt_type == PromptType.DOMAIN_EXPERT,
            'is_minimal': config.prompt_type == PromptType.MINIMAL,
            'includes_task_context': config.includes_task_context,
            'includes_domain_knowledge': config.includes_domain_knowledge,
            'is_direct_style': config.instruction_style == 'direct',
            'is_conversational_style': config.instruction_style == 'conversational',
            'is_clinical_style': config.instruction_style == 'clinical'
        }
    
    return matrix

# =============================================================================
# SYSTEMATIC PROMPT TESTING UTILITIES
# =============================================================================

def get_systematic_prompt_pairs() -> List[tuple]:
    """
    Get pairs of prompts for systematic comparison.
    
    Returns:
        List of tuples (prompt1_name, prompt2_name, comparison_type)
    """
    pairs = []
    
    # Generic vs Task-Specific pairs
    generic_prompts = get_prompts_by_type(PromptType.GENERIC)
    task_specific_prompts = get_prompts_by_type(PromptType.TASK_SPECIFIC)
    
    for generic in generic_prompts:
        for task_specific in task_specific_prompts:
            pairs.append((generic, task_specific, "generic_vs_task_specific"))
    
    # Minimal vs Detailed pairs
    minimal_prompts = get_prompts_by_type(PromptType.MINIMAL)
    detailed_prompts = [name for name, config in PROMPT_TEMPLATES.items() 
                       if config.prompt_type in [PromptType.GENERIC, PromptType.TASK_SPECIFIC, PromptType.DOMAIN_EXPERT]]
    
    for minimal in minimal_prompts:
        for detailed in detailed_prompts[:2]:  # Limit to avoid too many combinations
            pairs.append((minimal, detailed, "minimal_vs_detailed"))
    
    # Domain Expert vs Regular pairs
    expert_prompts = get_prompts_by_type(PromptType.DOMAIN_EXPERT)
    regular_prompts = generic_prompts + task_specific_prompts
    
    for expert in expert_prompts:
        for regular in regular_prompts[:2]:  # Limit to avoid too many combinations
            pairs.append((expert, regular, "expert_vs_regular"))
    
    return pairs

def get_core_prompt_set() -> List[str]:
    """
    Get a core set of prompts for efficient systematic testing.
    
    Returns:
        List of prompt names representing key variations
    """
    return [
        'generic_basic',              # Basic generic
        'generic_detailed',           # Detailed generic  
        'task_specific_mortality',    # Basic task-specific
        'task_specific_mortality_detailed',  # Detailed task-specific
        'domain_expert_mortality_focused',   # Expert with task focus
        'minimal_context',            # Minimal baseline
        'minimal_task_specific'       # Minimal with task
    ]

# =============================================================================
# PROMPT VALIDATION
# =============================================================================

def validate_prompt_template(prompt_name: str, sample_data: str = "Sample patient data") -> Dict[str, Any]:
    """
    Validate a prompt template by testing it with sample data.
    
    Args:
        prompt_name: Name of the prompt template to validate
        sample_data: Sample patient data for testing
        
    Returns:
        Dictionary with validation results
    """
    try:
        prompt = generate_prompt(prompt_name, sample_data)
        config = get_prompt_config(prompt_name)
        
        validation = {
            'is_valid': True,
            'prompt_length': len(prompt),
            'has_patient_data': sample_data in prompt,
            'config': {
                'type': config.prompt_type.value,
                'has_task_context': config.includes_task_context,
                'has_domain_knowledge': config.includes_domain_knowledge,
                'instruction_style': config.instruction_style
            },
            'generated_prompt': prompt,
            'error': None
        }
        
    except Exception as e:
        validation = {
            'is_valid': False,
            'error': str(e),
            'prompt_length': 0,
            'has_patient_data': False
        }
    
    return validation

def validate_all_prompts(sample_data: str = "Sample patient data") -> Dict[str, Dict[str, Any]]:
    """
    Validate all prompt templates.
    
    Args:
        sample_data: Sample patient data for testing
        
    Returns:
        Dictionary mapping prompt names to validation results
    """
    results = {}
    
    for prompt_name in get_available_prompts():
        results[prompt_name] = validate_prompt_template(prompt_name, sample_data)
        
    return results

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def preview_all_prompts(sample_data: str, max_length: int = 300) -> Dict[str, str]:
    """
    Generate previews of all prompts with sample data.
    
    Args:
        sample_data: Sample patient data for previews
        max_length: Maximum length of preview text
        
    Returns:
        Dictionary mapping prompt names to preview strings
    """
    previews = {}
    
    for prompt_name in get_available_prompts():
        try:
            prompt = generate_prompt(prompt_name, sample_data)
            preview = prompt[:max_length]
            if len(prompt) > max_length:
                preview += "..."
            previews[prompt_name] = preview
        except Exception as e:
            previews[prompt_name] = f"Error: {str(e)}"
    
    return previews

def save_prompt_examples(sample_data: str, output_dir: str):
    """
    Save examples of all prompts to files.
    
    Args:
        sample_data: Sample patient data for examples
        output_dir: Directory to save examples
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for prompt_name in get_available_prompts():
        try:
            prompt = generate_prompt(prompt_name, sample_data)
            filename = os.path.join(output_dir, f"prompt_{prompt_name}.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Prompt: {prompt_name}\n")
                f.write(f"# Description: {get_prompt_config(prompt_name).description}\n\n")
                f.write(prompt)
                
            logging.info(f"Saved {prompt_name} example to {filename}")
            
        except Exception as e:
            logging.error(f"Failed to save {prompt_name} example: {str(e)}") 