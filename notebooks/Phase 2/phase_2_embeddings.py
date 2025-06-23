# phase_2_embeddings.py

import pandas as pd
import numpy as np
import openai
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
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 50
    rate_limit_delay: float = 0.1
    timeout: float = 30.0

# Supported embedding models
EMBEDDING_MODELS = {
    'text-embedding-3-large': EmbeddingConfig(
        model_name='text-embedding-3-large',
        max_tokens=8192,
        batch_size=50
    ),
    'text-embedding-3-small': EmbeddingConfig(
        model_name='text-embedding-3-small',
        max_tokens=8192,
        batch_size=100
    ),
    'text-embedding-ada-002': EmbeddingConfig(
        model_name='text-embedding-ada-002',
        max_tokens=8192,
        batch_size=100
    )
}

# =============================================================================
# EMBEDDING GENERATION FUNCTIONS
# =============================================================================

class EmbeddingGenerator:
    """Main class for generating embeddings using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'text-embedding-3-large'):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            model_name: Name of the embedding model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Set the API key
        openai.api_key = self.api_key
        
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(EMBEDDING_MODELS.keys())}")
        
        self.config = EMBEDDING_MODELS[model_name]
        self.model_name = model_name
        
        logging.info(f"Initialized EmbeddingGenerator with model: {model_name}")
    
    def _generate_single_embedding(self, text: str, patient_id: str = None) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text using OpenAI API.
        
        Args:
            text: Input text to embed
            patient_id: Optional patient identifier for logging
            
        Returns:
            Embedding vector as numpy array, or None if failed
        """
        for attempt in range(self.config.max_retries):
            try:
                # Truncate text if too long
                if len(text) > self.config.max_tokens * 4:  # Rough estimate: 4 chars per token
                    text = text[:self.config.max_tokens * 4]
                    logging.warning(f"Truncated text for patient {patient_id} (attempt {attempt + 1})")
                
                response = openai.Embedding.create(
                    model=self.config.model_name,
                    input=text,
                    timeout=self.config.timeout
                )
                
                embedding = np.array(response['data'][0]['embedding'])
                return embedding
                
            except openai.error.RateLimitError as e:
                wait_time = (2 ** attempt) * self.config.retry_delay
                logging.warning(f"Rate limit hit for patient {patient_id}. Waiting {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                
            except openai.error.APIError as e:
                logging.error(f"API error for patient {patient_id}: {str(e)} (attempt {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                
            except Exception as e:
                logging.error(f"Unexpected error for patient {patient_id}: {str(e)} (attempt {attempt + 1})")
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
        
        # Process in batches with rate limiting
        for i in tqdm(range(0, len(texts), self.config.batch_size), 
                     desc="Generating embeddings", disable=not show_progress):
            
            batch_texts = texts[i:i + self.config.batch_size]
            batch_ids = patient_ids[i:i + self.config.batch_size]
            
            for text, patient_id in zip(batch_texts, batch_ids):
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
                    logging.error(f"Error processing patient {patient_id}: {str(e)}")
        
        logging.info(f"Successfully generated {len(embeddings)} embeddings out of {len(texts)} requests")
        return embeddings

# =============================================================================
# EMBEDDING EXPERIMENT FUNCTIONS
# =============================================================================

def run_embedding_experiment(
    patient_data: pd.DataFrame,
    target_data: pd.Series,
    serialization_formats: List[str],
    prompt_names: List[str],
    output_dir: str,
    model_name: str = 'text-embedding-3-large',
    sample_size: Optional[int] = None,
    use_parallel: bool = False,
    max_workers: int = 5
) -> Dict[str, Any]:
    """
    Run a complete embedding experiment with multiple serialization and prompt combinations.
    
    Args:
        patient_data: DataFrame with patient features (one row per patient)
        target_data: Series with target outcomes (mortality labels)
        serialization_formats: List of serialization format names to test
        prompt_names: List of prompt template names to test
        output_dir: Directory to save results
        model_name: OpenAI embedding model to use
        sample_size: Optional sample size for testing (None = use all data)
        use_parallel: Whether to use parallel processing
        max_workers: Number of parallel workers if using parallel processing
        
    Returns:
        Dictionary with experiment results and metadata
    """
    from .phase_2_serialization import serialize_patient_data, get_available_formats
    from .phase_2_prompts import generate_prompt, get_available_prompts
    
    # Validate inputs
    if not all(fmt in get_available_formats() for fmt in serialization_formats):
        invalid_formats = [fmt for fmt in serialization_formats if fmt not in get_available_formats()]
        raise ValueError(f"Invalid serialization formats: {invalid_formats}")
    
    if not all(prompt in get_available_prompts() for prompt in prompt_names):
        invalid_prompts = [prompt for prompt in prompt_names if prompt not in get_available_prompts()]
        raise ValueError(f"Invalid prompt names: {invalid_prompts}")
    
    # Sample data if requested
    if sample_size is not None and sample_size < len(patient_data):
        logging.info(f"Sampling {sample_size} patients from {len(patient_data)} total")
        sample_indices = np.random.choice(patient_data.index, size=sample_size, replace=False)
        patient_data = patient_data.loc[sample_indices]
        target_data = target_data.loc[sample_indices]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(model_name=model_name)
    
    # Store all results
    experiment_results = {
        'metadata': {
            'model_name': model_name,
            'n_patients': len(patient_data),
            'serialization_formats': serialization_formats,
            'prompt_names': prompt_names,
            'use_parallel': use_parallel,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'embeddings': {},
        'serialized_texts': {},
        'prompts': {},
        'patient_ids': patient_data.index.tolist(),
        'target_data': target_data.to_dict()
    }
    
    total_combinations = len(serialization_formats) * len(prompt_names)
    logging.info(f"Running embedding experiment with {total_combinations} combinations")
    
    # Generate embeddings for each combination
    combination_count = 0
    for serialization_format in serialization_formats:
        for prompt_name in prompt_names:
            combination_count += 1
            combo_key = f"{serialization_format}_{prompt_name}"
            
            logging.info(f"Processing combination {combination_count}/{total_combinations}: {combo_key}")
            
            # Serialize patient data
            logging.info(f"Serializing patient data using format: {serialization_format}")
            serialized_texts = []
            patient_ids = []
            
            for patient_id, patient_row in patient_data.iterrows():
                try:
                    serialized_text = serialize_patient_data(patient_row, serialization_format)
                    serialized_texts.append(serialized_text)
                    patient_ids.append(str(patient_id))
                except Exception as e:
                    logging.error(f"Failed to serialize patient {patient_id}: {str(e)}")
                    continue
            
            # Generate prompts
            logging.info(f"Generating prompts using template: {prompt_name}")
            final_prompts = []
            valid_patient_ids = []
            
            for i, (patient_id, serialized_text) in enumerate(zip(patient_ids, serialized_texts)):
                try:
                    prompt = generate_prompt(prompt_name, serialized_text)
                    final_prompts.append(prompt)
                    valid_patient_ids.append(patient_id)
                except Exception as e:
                    logging.error(f"Failed to generate prompt for patient {patient_id}: {str(e)}")
                    continue
            
            # Generate embeddings
            logging.info(f"Generating embeddings for {len(final_prompts)} patients")
            if use_parallel:
                embeddings = generator.generate_embeddings_parallel(
                    final_prompts, valid_patient_ids, max_workers=max_workers
                )
            else:
                embeddings = generator.generate_embeddings_batch(
                    final_prompts, valid_patient_ids
                )
            
            # Store results
            experiment_results['embeddings'][combo_key] = embeddings
            experiment_results['serialized_texts'][combo_key] = {
                pid: text for pid, text in zip(valid_patient_ids, serialized_texts[:len(valid_patient_ids)])
            }
            experiment_results['prompts'][combo_key] = {
                pid: prompt for pid, prompt in zip(valid_patient_ids, final_prompts)
            }
            
            # Save intermediate results
            combo_filename = os.path.join(output_dir, f"embeddings_{combo_key}.pkl")
            with open(combo_filename, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'patient_ids': valid_patient_ids,
                    'metadata': {
                        'serialization_format': serialization_format,
                        'prompt_name': prompt_name,
                        'model_name': model_name,
                        'n_embeddings': len(embeddings)
                    }
                }, f)
            
            logging.info(f"Saved {len(embeddings)} embeddings to {combo_filename}")
    
    # Save complete experiment results
    results_filename = os.path.join(output_dir, "embedding_experiment_results.pkl")
    with open(results_filename, 'wb') as f:
        pickle.dump(experiment_results, f)
    
    # Save summary statistics
    summary = {
        'total_combinations': total_combinations,
        'successful_combinations': len(experiment_results['embeddings']),
        'embedding_counts': {
            combo: len(embeddings) 
            for combo, embeddings in experiment_results['embeddings'].items()
        },
        'metadata': experiment_results['metadata']
    }
    
    summary_filename = os.path.join(output_dir, "embedding_experiment_summary.json")
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Embedding experiment complete. Results saved to {output_dir}")
    logging.info(f"Summary: {summary['successful_combinations']}/{summary['total_combinations']} combinations successful")
    
    return experiment_results

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_embedding_results(results_path: str) -> Dict[str, Any]:
    """Load embedding experiment results from pickle file."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)

def get_embedding_matrix(embeddings_dict: Dict[str, np.ndarray], patient_ids: List[str]) -> np.ndarray:
    """
    Convert embeddings dictionary to matrix format.
    
    Args:
        embeddings_dict: Dictionary mapping patient IDs to embedding vectors
        patient_ids: Ordered list of patient IDs
        
    Returns:
        Matrix with shape (n_patients, embedding_dim)
    """
    matrices = []
    valid_ids = []
    
    for patient_id in patient_ids:
        if str(patient_id) in embeddings_dict:
            matrices.append(embeddings_dict[str(patient_id)])
            valid_ids.append(patient_id)
    
    if not matrices:
        raise ValueError("No valid embeddings found")
    
    return np.vstack(matrices), valid_ids

def compare_embedding_quality(
    embeddings_dict_1: Dict[str, np.ndarray],
    embeddings_dict_2: Dict[str, np.ndarray],
    patient_ids: List[str]
) -> Dict[str, float]:
    """
    Compare the quality of two sets of embeddings using various metrics.
    
    Args:
        embeddings_dict_1: First set of embeddings
        embeddings_dict_2: Second set of embeddings
        patient_ids: Patient IDs to compare
        
    Returns:
        Dictionary with comparison metrics
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get common patient IDs
    common_ids = [pid for pid in patient_ids 
                  if str(pid) in embeddings_dict_1 and str(pid) in embeddings_dict_2]
    
    if len(common_ids) < 2:
        return {'error': 'Insufficient common patients for comparison'}
    
    # Get embedding matrices
    matrix_1, _ = get_embedding_matrix(embeddings_dict_1, common_ids)
    matrix_2, _ = get_embedding_matrix(embeddings_dict_2, common_ids)
    
    # Calculate similarity metrics
    cosine_sim_1 = cosine_similarity(matrix_1)
    cosine_sim_2 = cosine_similarity(matrix_2)
    
    # Compare similarity structures
    correlation = np.corrcoef(cosine_sim_1.flatten(), cosine_sim_2.flatten())[0, 1]
    
    # Calculate other metrics
    embedding_dim_1 = matrix_1.shape[1]
    embedding_dim_2 = matrix_2.shape[1]
    
    mean_norm_1 = np.mean(np.linalg.norm(matrix_1, axis=1))
    mean_norm_2 = np.mean(np.linalg.norm(matrix_2, axis=1))
    
    return {
        'n_common_patients': len(common_ids),
        'similarity_correlation': correlation,
        'embedding_dim_1': embedding_dim_1,
        'embedding_dim_2': embedding_dim_2,
        'mean_norm_1': mean_norm_1,
        'mean_norm_2': mean_norm_2,
        'norm_ratio': mean_norm_2 / mean_norm_1 if mean_norm_1 > 0 else float('inf')
    } 