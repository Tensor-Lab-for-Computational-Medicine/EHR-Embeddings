# phase_2_embeddings.py

import pandas as pd
import numpy as np
import time
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import json

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str
    max_tokens: int = 8192
    batch_size: int = 50
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    timeout: float = 30.0

# Supported embedding models
EMBEDDING_MODELS = {
    'gemini-embedding-exp-03-07': EmbeddingConfig(
        model_name='gemini-embedding-exp-03-07',
        max_tokens=8192,
        batch_size=1  # Gemini embedding models support one instance per request
    ),
    'text-embedding-004': EmbeddingConfig(
        model_name='text-embedding-004',
        max_tokens=2048,
        batch_size=1
    ),
    'gemini-embedding-001': EmbeddingConfig(
        model_name='gemini-embedding-001',
        max_tokens=2048,
        batch_size=1
    )
}

# =============================================================================
# EMBEDDING GENERATION FUNCTIONS
# =============================================================================

class EmbeddingGenerator:
    """Main class for generating embeddings using Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-embedding-exp-03-07'):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: Google Gemini API key (if None, will look for GOOGLE_API_KEY or GEMINI_API_KEY env var)
            model_name: Name of the embedding model to use
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
                "or pass api_key parameter. Get your key at: https://aistudio.google.com/app/apikey"
            )
        
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(EMBEDDING_MODELS.keys())}")
        
        self.config = EMBEDDING_MODELS[model_name]
        self.model_name = model_name
        
        # Initialize the Gemini client
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Google Generative AI library not installed. Install with: pip install google-genai"
            )
        
        logging.info(f"Initialized EmbeddingGenerator with model: {model_name}")
    
    def _generate_single_embedding(self, text: str, patient_id: str = None) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text using Google Gemini API.
        
        Args:
            text: Input text to embed
            patient_id: Optional patient identifier for logging
            
        Returns:
            Embedding vector as numpy array, or None if failed
        """
        for attempt in range(self.config.max_retries):
            try:
                # Truncate text if too long (rough estimate: 4 chars per token)
                if len(text) > self.config.max_tokens * 4:
                    text = text[:self.config.max_tokens * 4]
                    logging.warning(f"Truncated text for patient {patient_id} (attempt {attempt + 1})")
                
                # Generate embedding using Gemini API
                result = self.client.models.embed_content(
                    model=self.config.model_name,
                    contents=text
                )
                
                # Extract embedding from response
                if result.embeddings and len(result.embeddings) > 0:
                    embedding = np.array(result.embeddings[0].values)
                    return embedding
                else:
                    logging.error(f"Empty embedding response for patient {patient_id} (attempt {attempt + 1})")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                if "rate limit" in error_msg or "quota" in error_msg:
                    wait_time = (2 ** attempt) * self.config.retry_delay
                    logging.warning(f"Rate limit hit for patient {patient_id}. Waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    logging.error(f"API error for patient {patient_id}: {str(e)} (attempt {attempt + 1})")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
        
        logging.error(f"Failed to generate embedding for patient {patient_id} after {self.config.max_retries} attempts")
        return None
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        patient_ids: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            patient_ids: Optional list of patient IDs corresponding to texts
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping patient IDs to embedding vectors
        """
        if patient_ids is None:
            patient_ids = [f"patient_{i}" for i in range(len(texts))]
        
        if len(texts) != len(patient_ids):
            raise ValueError("Length of texts and patient_ids must match")
        
        embeddings = {}
        
        # Process one by one (Gemini embedding models support one instance per request)
        for text, patient_id in tqdm(zip(texts, patient_ids), 
                                   total=len(texts),
                                   desc="Generating embeddings", 
                                   disable=not show_progress):
            
            embedding = self._generate_single_embedding(text, patient_id)
            if embedding is not None:
                embeddings[patient_id] = embedding
            
            # Rate limiting
            time.sleep(self.config.rate_limit_delay)
        
        logging.info(f"Successfully generated {len(embeddings)} embeddings out of {len(texts)} requests")
        return embeddings
    
    def generate_embeddings_parallel(
        self, 
        texts: List[str], 
        patient_ids: Optional[List[str]] = None,
        max_workers: int = 5,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings using parallel processing (use with caution for rate limits).
        
        Args:
            texts: List of texts to embed
            patient_ids: Optional list of patient IDs corresponding to texts
            max_workers: Maximum number of parallel workers
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping patient IDs to embedding vectors
        """
        if patient_ids is None:
            patient_ids = [f"patient_{i}" for i in range(len(texts))]
        
        if len(texts) != len(patient_ids):
            raise ValueError("Length of texts and patient_ids must match")
        
        embeddings = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self._generate_single_embedding, text, patient_id): patient_id
                for text, patient_id in zip(texts, patient_ids)
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_id), 
                             total=len(future_to_id),
                             desc="Generating embeddings", 
                             disable=not show_progress):
                
                patient_id = future_to_id[future]
                try:
                    embedding = future.result()
                    if embedding is not None:
                        embeddings[patient_id] = embedding
                except Exception as e:
                    logging.error(f"Exception in parallel processing for patient {patient_id}: {str(e)}")
        
        logging.info(f"Successfully generated {len(embeddings)} embeddings out of {len(texts)} requests")
        return embeddings

# =============================================================================
# EXPERIMENT RUNNING FUNCTIONS
# =============================================================================

def run_embedding_experiment(
    patient_data: pd.DataFrame,
    target_data: pd.Series,
    serialization_formats: List[str],
    prompt_names: List[str],
    output_dir: str,
    model_name: str = 'gemini-embedding-exp-03-07',
    sample_size: Optional[int] = None,
    use_parallel: bool = False,
    max_workers: int = 5
) -> Dict[str, Any]:
    """
    Run comprehensive embedding experiment with different serialization and prompt combinations.
    
    Args:
        patient_data: DataFrame with patient features
        target_data: Series with target outcomes
        serialization_formats: List of serialization format names to test
        prompt_names: List of prompt template names to test
        output_dir: Directory to save results
        model_name: Embedding model to use
        sample_size: Optional number of patients to sample for testing
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary containing experiment results
    """
    logging.info(f"Starting embedding experiment with {len(serialization_formats)} serialization formats "
                f"and {len(prompt_names)} prompt types")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data if requested
    if sample_size and sample_size < len(patient_data):
        logging.info(f"Sampling {sample_size} patients from {len(patient_data)} total")
        sampled_indices = patient_data.sample(n=sample_size, random_state=42).index
        patient_data = patient_data.loc[sampled_indices]
        target_data = target_data.loc[sampled_indices]
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(model_name=model_name)
    
    # Import serialization and prompt modules
    from phase_2_serialization import serialize_patient_data
    from phase_2_prompts import apply_prompt_template
    
    # Store all results
    experiment_results = {
        'embeddings': {},
        'metadata': {},
        'experiment_config': {
            'model_name': model_name,
            'serialization_formats': serialization_formats,
            'prompt_names': prompt_names,
            'sample_size': len(patient_data),
            'use_parallel': use_parallel
        }
    }
    
    # Run experiments for each combination
    total_combinations = len(serialization_formats) * len(prompt_names)
    combination_idx = 0
    
    for serialization_format in serialization_formats:
        for prompt_name in prompt_names:
            combination_idx += 1
            combo_key = f"{serialization_format}_{prompt_name}"
            
            logging.info(f"Processing combination {combination_idx}/{total_combinations}: {combo_key}")
            
            try:
                # Generate serialized texts for all patients
                logging.info(f"  Serializing {len(patient_data)} patients...")
                serialized_texts = []
                patient_ids = []
                
                for patient_id, patient_row in patient_data.iterrows():
                    # Serialize patient data
                    serialized_text = serialize_patient_data(patient_row, serialization_format)
                    
                    # Apply prompt template
                    prompted_text = apply_prompt_template(serialized_text, prompt_name)
                    
                    serialized_texts.append(prompted_text)
                    patient_ids.append(str(patient_id))
                
                # Generate embeddings
                logging.info(f"  Generating embeddings...")
                if use_parallel:
                    embeddings = embedding_generator.generate_embeddings_parallel(
                        serialized_texts, patient_ids, max_workers=max_workers
                    )
                else:
                    embeddings = embedding_generator.generate_embeddings_batch(
                        serialized_texts, patient_ids
                    )
                
                # Store results
                experiment_results['embeddings'][combo_key] = embeddings
                experiment_results['metadata'][combo_key] = {
                    'serialization_format': serialization_format,
                    'prompt_name': prompt_name,
                    'num_embeddings': len(embeddings),
                    'embedding_dimension': len(next(iter(embeddings.values()))) if embeddings else 0,
                    'success_rate': len(embeddings) / len(patient_data)
                }
                
                # Save intermediate results
                combo_output_file = os.path.join(output_dir, f"embeddings_{combo_key}.pkl")
                with open(combo_output_file, 'wb') as f:
                    pickle.dump(embeddings, f)
                
                logging.info(f"  ✓ Generated {len(embeddings)} embeddings "
                           f"(success rate: {len(embeddings)/len(patient_data):.2%})")
                
            except Exception as e:
                logging.error(f"  ✗ Failed to process combination {combo_key}: {str(e)}")
                experiment_results['embeddings'][combo_key] = {}
                experiment_results['metadata'][combo_key] = {
                    'serialization_format': serialization_format,
                    'prompt_name': prompt_name,
                    'error': str(e),
                    'num_embeddings': 0,
                    'success_rate': 0.0
                }
    
    # Save complete experiment results
    results_file = os.path.join(output_dir, 'embedding_experiment_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(experiment_results, f)
    
    # Save metadata as JSON for easy inspection
    metadata_file = os.path.join(output_dir, 'embedding_experiment_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(experiment_results['metadata'], f, indent=2)
    
    logging.info(f"Embedding experiment completed. Results saved to {output_dir}")
    
    return experiment_results

def save_embedding_results(embeddings_dict: Dict[str, np.ndarray], output_path: str):
    """Save embedding results to file."""
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    logging.info(f"Saved embeddings to {output_path}")

def load_embedding_results(results_path: str) -> Dict[str, Any]:
    """Load embedding experiment results from file."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)

def get_embedding_matrix(embeddings_dict: Dict[str, np.ndarray], patient_ids: List[str]) -> np.ndarray:
    """
    Convert embeddings dictionary to matrix format.
    
    Args:
        embeddings_dict: Dictionary mapping patient IDs to embeddings
        patient_ids: List of patient IDs in desired order
        
    Returns:
        Matrix where each row is an embedding vector
    """
    embeddings_list = []
    for patient_id in patient_ids:
        if patient_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[patient_id])
        else:
            # Handle missing embeddings - you might want to skip or use zeros
            logging.warning(f"Missing embedding for patient {patient_id}")
            continue
    
    if not embeddings_list:
        raise ValueError("No valid embeddings found")
    
    return np.array(embeddings_list)

def compare_embedding_quality(
    embeddings_dict_1: Dict[str, np.ndarray],
    embeddings_dict_2: Dict[str, np.ndarray],
    patient_ids: List[str]
) -> Dict[str, float]:
    """
    Compare quality between two sets of embeddings.
    
    Args:
        embeddings_dict_1: First set of embeddings
        embeddings_dict_2: Second set of embeddings
        patient_ids: List of patient IDs to compare
        
    Returns:
        Dictionary with quality comparison metrics
    """
    # Get common patient IDs
    common_ids = [pid for pid in patient_ids 
                  if pid in embeddings_dict_1 and pid in embeddings_dict_2]
    
    if len(common_ids) < 2:
        return {'error': 'Not enough common embeddings for comparison'}
    
    # Extract embeddings for common patients
    emb1_matrix = np.array([embeddings_dict_1[pid] for pid in common_ids])
    emb2_matrix = np.array([embeddings_dict_2[pid] for pid in common_ids])
    
    # Calculate similarity metrics
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Average pairwise cosine similarity within each set
    sim1 = cosine_similarity(emb1_matrix).mean()
    sim2 = cosine_similarity(emb2_matrix).mean()
    
    # Correlation between corresponding embeddings
    correlations = [np.corrcoef(emb1_matrix[i], emb2_matrix[i])[0,1] 
                   for i in range(len(common_ids))]
    avg_correlation = np.nanmean(correlations)
    
    return {
        'common_patients': len(common_ids),
        'avg_similarity_set1': sim1,
        'avg_similarity_set2': sim2,
        'avg_cross_correlation': avg_correlation,
        'similarity_difference': abs(sim1 - sim2)
    } 