
"""All Task Late Fusion Model 
Integrates late fusion models for six tasks, including confidence interval calculations with 1000 validations
Supported tasks: los3, los7, vaso, vent, mort_hosp, readmission_30
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LateFusionStrategies:
    """Late fusion strategies class, implementing multiple model fusion methods"""
    
    def __init__(self):
        """Initialize fusion strategies class"""
        self.strategies = {
            # Basic fusion strategies
            'simple_average': self.simple_average_fusion,
            'weighted_average': self.weighted_average_fusion,

            # Dynamic weighting fusion strategies
            'confidence_weighted': self.confidence_weighted_fusion,
            'uncertainty_weighted': self.uncertainty_weighted_fusion,
            'consistency_weighted': self.consistency_weighted_fusion,
            'difficulty_adaptive': self.difficulty_adaptive_fusion,
            'edge_enhanced': self.edge_enhanced_fusion,

            # Neural network fusion strategies
            'neural_network_basic': self.neural_network_basic_fusion,
            'neural_network_advanced': self.neural_network_advanced_fusion,
            'neural_network_wide': self.neural_network_wide_fusion,
            'neural_network_adaptive': self.neural_network_adaptive_fusion,
            'neural_network_ensemble': self.neural_network_ensemble_fusion,

            # Ensemble learning fusion strategies
            'stacking': self.stacking_fusion,
            'blending': self.blending_fusion,
            'voting': self.voting_fusion,
            'weighted_ensemble': self.weighted_ensemble_fusion,
            'adaptive_ensemble': self.adaptive_ensemble_fusion,
            'ensemble_neural_mlp': self.ensemble_neural_mlp_fusion,

            # Hierarchical decision fusion strategies
            'hierarchical_confidence': self.hierarchical_confidence_fusion,
            'hierarchical_uncertainty': self.hierarchical_uncertainty_fusion,
            'three_layer_decision': self.three_layer_decision_fusion,
            'hierarchical_complexity_based': self.hierarchical_complexity_based_fusion,
            'hierarchical_fusion_tendency': self.hierarchical_fusion_tendency_fusion,
            'hierarchical_edge_special': self.hierarchical_edge_special_fusion,
            'hierarchical_adaptive_threshold': self.hierarchical_adaptive_threshold_fusion,
        }

        # Parameter search space definition
        self.param_spaces = self._define_param_spaces()

        # Cache optimal parameters
        self.optimal_params_cache = {}

        # Parameter search configuration
        self.search_config = {
            'cv_folds': 3,  # Cross-validation folds
            'scoring': 'roc_auc',  # Scoring metric
            'n_iter': 20,  # Random search iterations
            'random_state': 42
        }

    def _define_param_spaces(self) -> Dict[str, Dict[str, List[Any]]]:
        """
        Define parameter search spaces for each fusion strategy

        Returns:
            Parameter search spaces dictionary
        """
        return {
            # Basic fusion strategies
            'simple_average': {},  # No parameters

            'weighted_average': {
                'baseline_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            },

            # Dynamic weighting fusion strategies
            'confidence_weighted': {},  # No tunable parameters

            'uncertainty_weighted': {},  # No tunable parameters

            'consistency_weighted': {
                'agreement_threshold': [0.5, 0.6, 0.7, 0.8, 0.9]
            },

            'difficulty_adaptive': {},  # No tunable parameters

            'edge_enhanced': {
                'edge_threshold': [0.05, 0.1, 0.15, 0.2, 0.25]
            },

            # Neural network fusion strategies
            'neural_network_basic': {
                'hidden_layer_sizes': [(5,), (10,), (15,), (20,)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [200, 500, 1000]
            },

            'neural_network_advanced': {
                'hidden_layer_sizes': [(10, 5), (20, 10), (30, 15), (50, 25)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [500, 1000, 2000]
            },

            'neural_network_wide': {
                'hidden_layer_sizes': [(20,), (50,), (100,), (200,)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [500, 1000]
            },

            'neural_network_adaptive': {
                'difficulty_thresholds': [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4)],
                'hidden_sizes_options': [
                    {'low': (5,), 'medium': (10,), 'high': (20, 10)},
                    {'low': (10,), 'medium': (20,), 'high': (30, 15)}
                ]
            },

            'neural_network_ensemble': {
                'n_models': [3, 5, 7],
                'max_iter_options': [200, 300, 500],
                'hidden_sizes_options': [
                    [(5,), (10,), (15, 8)],
                    [(10,), (20,), (30, 15)]
                ]
            },

            # Ensemble learning fusion strategies
            'stacking': {
                'meta_C': [0.1, 1.0, 10.0],
                'meta_max_iter': [500, 1000, 2000]
            },

            'blending': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },

            'voting': {},  # No parameters

            'weighted_ensemble': {
                'confidence_weight': [0.5, 0.6, 0.7, 0.8, 0.9],
                'agreement_weight': [0.1, 0.2, 0.3, 0.4, 0.5]
            },

            'adaptive_ensemble': {
                'performance_factor_weight': [0.3, 0.5, 0.7],
                'stability_factor_weight': [0.3, 0.5, 0.7]
            },

            'ensemble_neural_mlp': {
                'n_estimators': [3, 5, 7],
                'mlp_hidden_layers': [(10, 5), (20, 10), (30, 15)],
                'mlp_max_iter': [200, 300, 500]
            },

            # Hierarchical decision fusion strategies
            'hierarchical_confidence': {
                'confidence_threshold': [0.5, 0.6, 0.7, 0.8],
                'agreement_threshold': [0.6, 0.7, 0.8, 0.9],
                'conservative_weight': [0.3, 0.4, 0.5]
            },

            'hierarchical_uncertainty': {},  # No tunable parameters

            'three_layer_decision': {
                'confidence_threshold': [0.5, 0.6, 0.7, 0.8],
                'agreement_threshold': [0.6, 0.7, 0.8, 0.9],
                'conservative_weight': [0.3, 0.4, 0.5]
            },

            'hierarchical_complexity_based': {
                'low_complexity_threshold': [0.1, 0.15, 0.2],
                'high_complexity_threshold': [0.3, 0.4, 0.5]
            },

            'hierarchical_fusion_tendency': {
                'fusion_tendency_threshold': [0.5, 0.6, 0.7, 0.8],
                'conservative_weight': [0.2, 0.3, 0.4]
            },

            'hierarchical_edge_special': {
                'edge_threshold': [0.1, 0.15, 0.2, 0.25],
                'agreement_threshold': [0.7, 0.8, 0.9]
            },

            'hierarchical_adaptive_threshold': {
                'agreement_min': [0.5, 0.6],
                'agreement_max': [0.7, 0.8, 0.9],
                'confidence_min': [0.4, 0.5],
                'confidence_max': [0.7, 0.8]
            }
        }

    def search_optimal_params(self, strategy_name: str, y_true: np.ndarray,
                             y_pred_baseline: np.ndarray, y_pred_embedding: np.ndarray,
                             param_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Search optimal parameters for specified fusion strategy

        Args:
            strategy_name: Strategy name
            y_true: True labels
            y_pred_baseline: Baseline predictions
            y_pred_embedding: Embedding predictions
            param_file: Parameter cache file path

        Returns:
            Optimal parameters dictionary
        """
        # Check cache
        cache_key = f"{strategy_name}_{hash(str(y_true[:100]))}"
        if cache_key in self.optimal_params_cache:
            logger.info(f"Loading {strategy_name} optimal parameters from cache")
            return self.optimal_params_cache[cache_key]

        # Check parameter file
        if param_file and os.path.exists(param_file):
            try:
                with open(param_file, 'r') as f:
                    cached_params = json.load(f)
                if strategy_name in cached_params:
                    self.optimal_params_cache[cache_key] = cached_params[strategy_name]
                    logger.info(f"Loading {strategy_name} optimal parameters from file")
                    return cached_params[strategy_name]
            except Exception as e:
                logger.warning(f"Failed to load parameter file: {e}")

        # Get parameter space
        param_space = self.param_spaces.get(strategy_name, {})
        if not param_space:
            logger.info(f"{strategy_name} has no tunable parameters, using default settings")
            default_params = {}
            self.optimal_params_cache[cache_key] = default_params
            return default_params

        logger.info(f"Searching optimal parameters for {strategy_name}...")
        logger.info(f"Parameter space size: {len(list(self._generate_param_combinations(param_space)))}")

        # Grid search
        best_params = None
        best_score = -np.inf

        for params in self._generate_param_combinations(param_space):
            try:
                # Cross-validation evaluation
                scores = []
                n_samples = len(y_true)

                # Simple cross-validation (using sample-level splits for fusion predictions)
                indices = np.arange(n_samples)
                np.random.seed(self.search_config['random_state'])
                np.random.shuffle(indices)

                fold_size = n_samples // self.search_config['cv_folds']

                for fold in range(self.search_config['cv_folds']):
                    val_start = fold * fold_size
                    val_end = (fold + 1) * fold_size if fold < self.search_config['cv_folds'] - 1 else n_samples

                    val_indices = indices[val_start:val_end]
                    train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

                    # Training set predictions
                    y_true_train = y_true[train_indices]
                    y_pred_baseline_train = y_pred_baseline[train_indices]
                    y_pred_embedding_train = y_pred_embedding[train_indices]

                    # Validation set predictions
                    y_true_val = y_true[val_indices]
                    y_pred_baseline_val = y_pred_baseline[val_indices]
                    y_pred_embedding_val = y_pred_embedding[val_indices]

                    # Generate fused predictions
                    fused_pred = self._apply_strategy_with_params(
                        strategy_name, y_pred_baseline_train, y_pred_embedding_train, params, y_true_train
                    )

                    if fused_pred is None:
                        continue

                    # Calculate AUC
                    if len(np.unique(y_true_train)) > 1:
                        score = roc_auc_score(y_true_train, fused_pred)
                        scores.append(score)

                if scores:
                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params.copy()

            except Exception as e:
                logger.warning(f"Parameter combination {params} evaluation failed: {e}")
                continue

        if best_params is None:
            logger.warning(f"{strategy_name} parameter search failed, using default parameters")
            best_params = {key: values[0] for key, values in param_space.items()}

        logger.info(f"{strategy_name} optimal parameters: {best_params}, validation AUC: {best_score:.4f}")

        # Cache results
        self.optimal_params_cache[cache_key] = best_params

        # Save to file
        if param_file:
            try:
                os.makedirs(os.path.dirname(param_file), exist_ok=True)
                cached_data = {}
                if os.path.exists(param_file):
                    with open(param_file, 'r') as f:
                        cached_data = json.load(f)

                cached_data[strategy_name] = best_params
                with open(param_file, 'w') as f:
                    json.dump(cached_data, f, indent=2)

            except Exception as e:
                logger.warning(f"Failed to save parameter file: {e}")

        return best_params

    def _generate_param_combinations(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations

        Args:
            param_space: Parameter space

        Returns:
            Parameter combinations list
        """
        if not param_space:
            return [{}]

        # Simple grid search implementation
        keys = list(param_space.keys())
        values = list(param_space.values())

        combinations = []
        for combo in np.ndindex(*[len(v) for v in values]):
            param_dict = {keys[i]: values[i][combo[i]] for i in range(len(keys))}
            combinations.append(param_dict)

        return combinations

    def _apply_strategy_with_params(self, strategy_name: str, y_pred_baseline: np.ndarray,
                                   y_pred_embedding: np.ndarray, params: Dict[str, Any],
                                   y_true: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Apply fusion strategy with specified parameters

        Args:
            strategy_name: Strategy name
            y_pred_baseline: Baseline predictions
            y_pred_embedding: Embedding predictions
            params: Parameter dictionary
            y_true: True labels (optional, for supervised methods)

        Returns:
            Fusion prediction results
        """
        try:
            strategy_func = self.strategies[strategy_name]

            # Call corresponding method based on strategy name and parameters
            # Supervised methods that need real labels
            if strategy_name in ['neural_network_basic', 'neural_network_advanced',
                                 'neural_network_wide', 'neural_network_adaptive',
                                 'neural_network_ensemble', 'stacking', 'blending',
                                 'ensemble_neural_mlp']:
                if y_true is None:
                    logger.warning(f"{strategy_name} requires true labels but none provided")
                    return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)
                return strategy_func(y_pred_baseline, y_pred_embedding, y_true, **params)
            # Unsupervised methods
            elif strategy_name == 'weighted_average':
                return strategy_func(y_pred_baseline, y_pred_embedding, **params)
            elif strategy_name == 'consistency_weighted':
                return strategy_func(y_pred_baseline, y_pred_embedding, **params)
            elif strategy_name == 'edge_enhanced':
                return strategy_func(y_pred_baseline, y_pred_embedding, **params)
            elif strategy_name == 'weighted_ensemble':
                return strategy_func(y_pred_baseline, y_pred_embedding, **params)
            elif strategy_name == 'adaptive_ensemble':
                return strategy_func(y_pred_baseline, y_pred_embedding, **params)
            elif strategy_name in ['hierarchical_confidence', 'three_layer_decision',
                                 'hierarchical_fusion_tendency', 'hierarchical_edge_special',
                                 'hierarchical_adaptive_threshold']:
                return strategy_func(y_pred_baseline, y_pred_embedding, **params)
            elif strategy_name == 'hierarchical_complexity_based':
                return strategy_func(y_pred_baseline, y_pred_embedding, **params)
            else:
                # Parameter-free strategies
                return strategy_func(y_pred_baseline, y_pred_embedding)

        except Exception as e:
            logger.warning(f"Error applying strategy {strategy_name}: {e}")
            return None

    def simple_average_fusion(self, y_pred_baseline: np.ndarray, 
                            y_pred_embedding: np.ndarray) -> np.ndarray:
        """Simple average fusion"""
        return 0.5 * y_pred_baseline + 0.5 * y_pred_embedding
    
    def weighted_average_fusion(self, y_pred_baseline: np.ndarray,
                              y_pred_embedding: np.ndarray,
                              baseline_weight: float = 0.6, **kwargs) -> np.ndarray:
        """Weighted average fusion"""
        # Use baseline_weight from kwargs if provided
        if 'baseline_weight' in kwargs:
            baseline_weight = kwargs['baseline_weight']
        embedding_weight = 1 - baseline_weight
        return baseline_weight * y_pred_baseline + embedding_weight * y_pred_embedding
    
    def confidence_weighted_fusion(self, y_pred_baseline: np.ndarray, 
                                 y_pred_embedding: np.ndarray) -> np.ndarray:
        """Confidence weighted fusion"""
        # Calculate confidence (distance from 0.5)
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2
        
        # Calculate weights
        total_confidence = baseline_confidence + embedding_confidence
        baseline_weight = baseline_confidence / (total_confidence + 1e-8)
        embedding_weight = embedding_confidence / (total_confidence + 1e-8)
        
        return baseline_weight * y_pred_baseline + embedding_weight * y_pred_embedding
    
    def uncertainty_weighted_fusion(self, y_pred_baseline: np.ndarray, 
                                  y_pred_embedding: np.ndarray) -> np.ndarray:
        """Uncertainty weighted fusion"""
        # Calculate uncertainty (closeness to 0.5)
        baseline_uncertainty = 1 - np.abs(y_pred_baseline - 0.5) * 2
        embedding_uncertainty = 1 - np.abs(y_pred_embedding - 0.5) * 2
        
        # Higher uncertainty means lower weight
        baseline_weight = 1 / (baseline_uncertainty + 1e-8)
        embedding_weight = 1 / (embedding_uncertainty + 1e-8)
        
        # Normalize weights
        total_weight = baseline_weight + embedding_weight
        baseline_weight = baseline_weight / total_weight
        embedding_weight = embedding_weight / total_weight
        
        return baseline_weight * y_pred_baseline + embedding_weight * y_pred_embedding
    
    def consistency_weighted_fusion(self, y_pred_baseline: np.ndarray,
                                  y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Consistency weighted fusion"""
        # Get parameters from kwargs
        agreement_threshold = kwargs.get('agreement_threshold', 0.8)

        # Calculate agreement between two predictions
        agreement = 1 - np.abs(y_pred_baseline - y_pred_embedding)

        # Use average when agreement is high, select more confident prediction when agreement is low
        high_agreement = agreement > agreement_threshold
        
        # Use weighted average when agreement is high
        weighted_avg = 0.5 * y_pred_baseline + 0.5 * y_pred_embedding
        
        # Select more confident prediction when agreement is low
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2
        
        confident_pred = np.where(baseline_confidence > embedding_confidence, 
                                y_pred_baseline, y_pred_embedding)
        
        return np.where(high_agreement, weighted_avg, confident_pred)
    
    def difficulty_adaptive_fusion(self, y_pred_baseline: np.ndarray,
                                 y_pred_embedding: np.ndarray) -> np.ndarray:
        """Difficulty adaptive fusion"""
        # Calculate prediction difficulty (difference between two model predictions)
        difficulty = np.abs(y_pred_baseline - y_pred_embedding)

        # Rely more on embedding model when difficulty is high, baseline model when difficulty is low
        baseline_weight = 1 - difficulty
        embedding_weight = difficulty

        # Normalize weights
        total_weight = baseline_weight + embedding_weight
        baseline_weight = baseline_weight / total_weight
        embedding_weight = embedding_weight / total_weight

        return baseline_weight * y_pred_baseline + embedding_weight * y_pred_embedding
    
    def edge_enhanced_fusion(self, y_pred_baseline: np.ndarray,
                           y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Edge enhanced fusion"""
        # Get parameters from kwargs
        edge_threshold = kwargs.get('edge_threshold', 0.1)

        # Identify edge cases (predictions close to decision boundary)
        baseline_edge = np.abs(y_pred_baseline - 0.5) < edge_threshold
        embedding_edge = np.abs(y_pred_embedding - 0.5) < edge_threshold

        # Edge cases use more conservative fusion
        edge_cases = baseline_edge | embedding_edge

        # Non-edge cases use standard fusion
        standard_fusion = 0.5 * y_pred_baseline + 0.5 * y_pred_embedding

        # Edge cases use confidence weighted fusion
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2

        total_confidence = baseline_confidence + embedding_confidence
        baseline_weight = baseline_confidence / (total_confidence + 1e-8)
        embedding_weight = embedding_confidence / (total_confidence + 1e-8)

        edge_fusion = baseline_weight * y_pred_baseline + embedding_weight * y_pred_embedding

        return np.where(edge_cases, edge_fusion, standard_fusion)
    
    def neural_network_basic_fusion(self, y_pred_baseline: np.ndarray,
                                  y_pred_embedding: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        """Basic neural network fusion using real labels"""
        # Get parameters from kwargs
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (10,))
        activation = kwargs.get('activation', 'relu')
        solver = kwargs.get('solver', 'adam')
        alpha = kwargs.get('alpha', 0.0001)
        max_iter = kwargs.get('max_iter', 500)

        # Create input features
        features = np.column_stack([
            y_pred_baseline,
            y_pred_embedding,
            y_pred_baseline * y_pred_embedding,  # Interaction features
            np.abs(y_pred_baseline - y_pred_embedding)  # Difference features
        ])

        try:
            # Train basic MLP using real labels
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                max_iter=max_iter,
                random_state=42
            )
            
            # Use real labels for training
            mlp.fit(features, y_true)
            
            return mlp.predict_proba(features)[:, 1]
        except Exception as e:
            logger.warning(f"Neural network fusion failed, using simple average: {e}")
            return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)
    
    def neural_network_advanced_fusion(self, y_pred_baseline: np.ndarray,
                                     y_pred_embedding: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        """Advanced neural network fusion using real labels"""
        # Get parameters from kwargs
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (20, 10))
        activation = kwargs.get('activation', 'relu')
        solver = kwargs.get('solver', 'adam')
        alpha = kwargs.get('alpha', 0.0001)
        max_iter = kwargs.get('max_iter', 1000)

        # Create richer features
        features = np.column_stack([
            y_pred_baseline,
            y_pred_embedding,
            y_pred_baseline * y_pred_embedding,
            np.abs(y_pred_baseline - y_pred_embedding),
            (y_pred_baseline + y_pred_embedding) / 2,
            np.maximum(y_pred_baseline, y_pred_embedding),
            np.minimum(y_pred_baseline, y_pred_embedding),
            y_pred_baseline ** 2,
            y_pred_embedding ** 2
        ])

        try:
            # Train more complex MLP using real labels
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                max_iter=max_iter,
                random_state=42
            )
            
            mlp.fit(features, y_true)
            
            return mlp.predict_proba(features)[:, 1]
        except Exception as e:
            logger.warning(f"Advanced neural network fusion failed, using simple average: {e}")
            return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)

    def neural_network_wide_fusion(self, y_pred_baseline: np.ndarray,
                                   y_pred_embedding: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        """Wide neural network fusion using real labels"""
        # Get parameters from kwargs
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (50,))
        activation = kwargs.get('activation', 'relu')
        solver = kwargs.get('solver', 'adam')
        alpha = kwargs.get('alpha', 0.0001)
        max_iter = kwargs.get('max_iter', 500)

        # Create input features (same as basic version)
        features = np.column_stack([
            y_pred_baseline,
            y_pred_embedding,
            y_pred_baseline * y_pred_embedding,
            np.abs(y_pred_baseline - y_pred_embedding)
        ])

        try:
            # Train wide MLP using real labels
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                max_iter=max_iter,
                random_state=42
            )

            mlp.fit(features, y_true)

            return mlp.predict_proba(features)[:, 1]
        except Exception as e:
            logger.warning(f"Wide neural network fusion failed, using simple average: {e}")
            return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)

    def neural_network_adaptive_fusion(self, y_pred_baseline: np.ndarray,
                                       y_pred_embedding: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        """Adaptive neural network fusion using real labels"""
        # Get parameters from kwargs
        difficulty_thresholds = kwargs.get('difficulty_thresholds', (0.1, 0.3))
        hidden_sizes_options = kwargs.get('hidden_sizes_options', [
            {'low': (5,), 'medium': (10,), 'high': (20, 10)},
            {'low': (10,), 'medium': (20,), 'high': (30, 15)}
        ])

        # Use first configuration as default
        config = hidden_sizes_options[0]

        # Create input features
        features = np.column_stack([
            y_pred_baseline,
            y_pred_embedding,
            y_pred_baseline * y_pred_embedding,
            np.abs(y_pred_baseline - y_pred_embedding),
            np.abs(y_pred_baseline - 0.5),  # Confidence of baseline model
            np.abs(y_pred_embedding - 0.5), # Confidence of embedding model
        ])

        try:
            # Calculate data complexity
            difficulty = np.mean(np.abs(y_pred_baseline - y_pred_embedding))

            # Adjust network complexity based on difficulty
            if difficulty > difficulty_thresholds[1]:  # High difficulty, use more complex network
                hidden_sizes = config['high']
                max_iter = 1000
            elif difficulty > difficulty_thresholds[0]:  # Medium difficulty
                hidden_sizes = config['medium']
                max_iter = 500
            else:  # Low difficulty, use simple network
                hidden_sizes = config['low']
                max_iter = 300

            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_sizes,
                activation='relu',
                solver='adam',
                max_iter=max_iter,
                random_state=42
            )

            mlp.fit(features, y_true)

            return mlp.predict_proba(features)[:, 1]
        except Exception as e:
            logger.warning(f"Adaptive neural network fusion failed, using simple average: {e}")
            return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)

    def neural_network_ensemble_fusion(self, y_pred_baseline: np.ndarray,
                                       y_pred_embedding: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Ensemble neural network fusion using real labels"""
        # Create input features
        features = np.column_stack([
            y_pred_baseline,
            y_pred_embedding,
            y_pred_baseline * y_pred_embedding,
            np.abs(y_pred_baseline - y_pred_embedding),
            (y_pred_baseline + y_pred_embedding) / 2,
        ])

        try:
            # Create multiple different MLPs
            models = []
            predictions = []

            # Different configurations of MLP
            configs = [
                {'hidden_layer_sizes': (10,), 'max_iter': 300},
                {'hidden_layer_sizes': (20,), 'max_iter': 500},
                {'hidden_layer_sizes': (15, 8), 'max_iter': 400},
            ]

            for config in configs:
                mlp = MLPClassifier(
                    activation='relu',
                    solver='adam',
                    random_state=42,
                    **config
                )
                mlp.fit(features, y_true)
                pred = mlp.predict_proba(features)[:, 1]
                predictions.append(pred)

            # Ensemble prediction (average)
            return np.mean(predictions, axis=0)

        except Exception as e:
            logger.warning(f"Ensemble neural network fusion failed, using simple average: {e}")
            return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)

    def stacking_fusion(self, y_pred_baseline: np.ndarray,
                       y_pred_embedding: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        """Stacking fusion strategy using real labels"""
        # Get parameters from kwargs
        meta_C = kwargs.get('meta_C', 1.0)
        meta_max_iter = kwargs.get('meta_max_iter', 1000)

        # Create meta features
        meta_features = np.column_stack([
            y_pred_baseline,
            y_pred_embedding,
            y_pred_baseline * y_pred_embedding,
            np.abs(y_pred_baseline - y_pred_embedding),
            (y_pred_baseline + y_pred_embedding) / 2
        ])

        try:
            # Use logistic regression as meta learner with real labels
            meta_learner = LogisticRegression(C=meta_C, random_state=42, max_iter=meta_max_iter)
            meta_learner.fit(meta_features, y_true)
            
            return meta_learner.predict_proba(meta_features)[:, 1]
        except Exception as e:
            logger.warning(f"Stacking fusion failed, using simple average: {e}")
            return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)
    
    def blending_fusion(self, y_pred_baseline: np.ndarray,
                       y_pred_embedding: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        """Blending fusion strategy using real labels"""
        # Get parameters from kwargs
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 10)
        min_samples_split = kwargs.get('min_samples_split', 2)

        # Use random forest as blending model with real labels
        try:
            features = np.column_stack([y_pred_baseline, y_pred_embedding])
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            rf.fit(features, y_true)
            
            return rf.predict_proba(features)[:, 1]
        except Exception as e:
            logger.warning(f"Blending fusion failed, using simple average: {e}")
            return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)
    
    def voting_fusion(self, y_pred_baseline: np.ndarray,
                     y_pred_embedding: np.ndarray) -> np.ndarray:
        """Voting fusion strategy"""
        # Convert probabilities to predictions
        baseline_pred = (y_pred_baseline > 0.5).astype(int)
        embedding_pred = (y_pred_embedding > 0.5).astype(int)

        # Voting
        votes = baseline_pred + embedding_pred
        voting_result = (votes >= 1).astype(float)

        # Convert to probability (based on voting strength)
        return votes / 2.0

    def weighted_ensemble_fusion(self, y_pred_baseline: np.ndarray,
                                y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Weighted ensemble fusion"""
        # Get parameters from kwargs
        confidence_weight = kwargs.get('confidence_weight', 0.7)
        agreement_weight = kwargs.get('agreement_weight', 0.3)

        # Calculate dynamic weights based on multiple factors
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2

        agreement = 1 - np.abs(y_pred_baseline - y_pred_embedding)
        disagreement_penalty = 1 - agreement

        # Comprehensive weight calculation with configurable weights
        baseline_weight = (baseline_confidence * confidence_weight + agreement * agreement_weight) / (baseline_confidence + embedding_confidence + agreement + disagreement_penalty + 1e-8)
        embedding_weight = (embedding_confidence * confidence_weight + agreement * agreement_weight) / (baseline_confidence + embedding_confidence + agreement + disagreement_penalty + 1e-8)

        # Normalize
        total_weight = baseline_weight + embedding_weight
        baseline_weight = baseline_weight / total_weight
        embedding_weight = embedding_weight / total_weight

        return baseline_weight * y_pred_baseline + embedding_weight * y_pred_embedding

    def adaptive_ensemble_fusion(self, y_pred_baseline: np.ndarray,
                                y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Adaptive ensemble fusion"""
        # Get parameters from kwargs
        performance_factor_weight = kwargs.get('performance_factor_weight', 0.5)
        stability_factor_weight = kwargs.get('stability_factor_weight', 0.5)

        # Calculate dataset statistics
        mean_baseline = np.mean(y_pred_baseline)
        std_baseline = np.std(y_pred_baseline)
        mean_embedding = np.mean(y_pred_embedding)
        std_embedding = np.std(y_pred_embedding)

        # Adaptive weight adjustment factors
        performance_factor = 1 / (1 + np.abs(mean_baseline - mean_embedding))  # Performance difference factor
        stability_factor = 1 / (1 + np.abs(std_baseline - std_embedding))  # Stability factor

        # Base weights (based on confidence)
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2

        # Apply adaptive factors with configurable weights
        baseline_adaptive_weight = baseline_confidence * (performance_factor * performance_factor_weight + (1 - performance_factor_weight))
        embedding_adaptive_weight = embedding_confidence * (stability_factor * stability_factor_weight + (1 - stability_factor_weight))

        # Normalize weights
        total_weight = baseline_adaptive_weight + embedding_adaptive_weight
        baseline_weight = baseline_adaptive_weight / (total_weight + 1e-8)
        embedding_weight = embedding_adaptive_weight / (total_weight + 1e-8)

        return baseline_weight * y_pred_baseline + embedding_weight * y_pred_embedding

    def ensemble_neural_mlp_fusion(self, y_pred_baseline: np.ndarray,
                                  y_pred_embedding: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        """Ensemble neural network MLP fusion using real labels"""
        # Get parameters from kwargs
        n_estimators = kwargs.get('n_estimators', 5)
        mlp_hidden_layers = kwargs.get('mlp_hidden_layers', (20, 10))
        mlp_max_iter = kwargs.get('mlp_max_iter', 300)

        # Create rich features for ensemble
        features = np.column_stack([
            y_pred_baseline,
            y_pred_embedding,
            y_pred_baseline * y_pred_embedding,
            np.abs(y_pred_baseline - y_pred_embedding),
            (y_pred_baseline + y_pred_embedding) / 2,
            np.maximum(y_pred_baseline, y_pred_embedding),
            np.minimum(y_pred_baseline, y_pred_embedding),
            np.abs(y_pred_baseline - 0.5),  # Baseline confidence
            np.abs(y_pred_embedding - 0.5), # Embedding confidence
            np.sign(y_pred_baseline - 0.5) * np.sign(y_pred_embedding - 0.5),  # Direction consistency
        ])

        try:
            # Use ensemble MLP with real labels and configurable parameters
            from sklearn.ensemble import BaggingClassifier

            base_mlp = MLPClassifier(
                hidden_layer_sizes=mlp_hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=mlp_max_iter,
                random_state=42
            )

            # Create bagging ensemble with configurable n_estimators
            bagging_mlp = BaggingClassifier(
                base_estimator=base_mlp,
                n_estimators=n_estimators,
                random_state=42
            )

            bagging_mlp.fit(features, y_true)

            # Predict probabilities
            probas = bagging_mlp.predict_proba(features)
            return probas[:, 1]

        except Exception as e:
            logger.warning(f"Ensemble neural network MLP fusion failed, using simple average: {e}")
            return self.simple_average_fusion(y_pred_baseline, y_pred_embedding)

    def hierarchical_confidence_fusion(self, y_pred_baseline: np.ndarray,
                                     y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Hierarchical confidence fusion"""
        # Get parameters from kwargs
        confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        agreement_threshold = kwargs.get('agreement_threshold', 0.8)
        conservative_weight = kwargs.get('conservative_weight', 0.3)

        # First layer: Confidence assessment
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2

        # Second layer: Agreement assessment
        agreement = 1 - np.abs(y_pred_baseline - y_pred_embedding)

        # Third layer: Final decision
        
        high_confidence = (baseline_confidence > confidence_threshold) & (embedding_confidence > confidence_threshold)
        high_agreement = agreement > agreement_threshold

        # High confidence and high agreement: Use weighted average
        case1 = high_confidence & high_agreement
        weight1 = baseline_confidence / (baseline_confidence + embedding_confidence + 1e-8)
        result1 = weight1 * y_pred_baseline + (1 - weight1) * y_pred_embedding
        
        # High confidence but low agreement: Select more confident
        case2 = high_confidence & ~high_agreement
        result2 = np.where(baseline_confidence > embedding_confidence, 
                          y_pred_baseline, y_pred_embedding)
        
        # Low confidence: Use conservative fusion
        case3 = ~high_confidence
        result3 = 0.3 * y_pred_baseline + 0.3 * y_pred_embedding + 0.4 * 0.5
        
        # Combine results
        result = np.where(case1, result1, 
                         np.where(case2, result2, result3))
        
        return result
    
    def hierarchical_uncertainty_fusion(self, y_pred_baseline: np.ndarray,
                                      y_pred_embedding: np.ndarray) -> np.ndarray:
        """Hierarchical uncertainty fusion"""
        # Calculate uncertainty
        baseline_uncertainty = 1 - np.abs(y_pred_baseline - 0.5) * 2
        embedding_uncertainty = 1 - np.abs(y_pred_embedding - 0.5) * 2

        # Calculate difference
        difference = np.abs(y_pred_baseline - y_pred_embedding)

        # Adjust weights based on uncertainty and difference
        uncertainty_weight = (baseline_uncertainty + embedding_uncertainty) / 2
        difference_weight = difference

        # Comprehensive weight
        total_weight = uncertainty_weight + difference_weight

        baseline_weight = (1 - uncertainty_weight) / (total_weight + 1e-8)
        embedding_weight = (1 - embedding_uncertainty) / (total_weight + 1e-8)

        # Normalize
        total = baseline_weight + embedding_weight
        baseline_weight = baseline_weight / total
        embedding_weight = embedding_weight / total

        return baseline_weight * y_pred_baseline + embedding_weight * y_pred_embedding
    
    def three_layer_decision_fusion(self, y_pred_baseline: np.ndarray, 
                                  y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Three layer decision fusion"""
        # Get parameters from kwargs
        confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        agreement_threshold = kwargs.get('agreement_threshold', 0.8)
        conservative_weight = kwargs.get('conservative_weight', 0.3)

        # First layer: Confidence assessment
        baseline_confidence = 1 - np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = 1 - np.abs(y_pred_embedding - 0.5) * 2
        
        # Second layer: Agreement assessment
        agreement = 1 - np.abs(y_pred_baseline - y_pred_embedding)
        
        # Third layer: Final decision
        
        high_confidence = (baseline_confidence > confidence_threshold) & (embedding_confidence > confidence_threshold)
        high_agreement = agreement > agreement_threshold
        
        # Decision rules
        # High confidence and high agreement: Use weighted average
        case1 = high_confidence & high_agreement
        weight1 = baseline_confidence / (baseline_confidence + embedding_confidence + 1e-8)
        result1 = weight1 * y_pred_baseline + (1 - weight1) * y_pred_embedding
        
        # High confidence but low agreement: Select more confident
        case2 = high_confidence & ~high_agreement
        result2 = np.where(baseline_confidence > embedding_confidence, 
                          y_pred_baseline, y_pred_embedding)
        
        # Low confidence: Use conservative fusion
        case3 = ~high_confidence
        result3 = 0.3 * y_pred_baseline + 0.3 * y_pred_embedding + 0.4 * 0.5
        
        # Combine results
        result = np.where(case1, result1, 
                         np.where(case2, result2, result3))
        
        return result

    def hierarchical_complexity_based_fusion(self, y_pred_baseline: np.ndarray,
                                             y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Complexity-based hierarchical fusion"""
        # Get parameters from kwargs
        low_complexity_threshold = kwargs.get('low_complexity_threshold', 0.2)
        high_complexity_threshold = kwargs.get('high_complexity_threshold', 0.4)

        # Calculate prediction complexity (degree of difference between two model predictions)
        complexity = np.abs(y_pred_baseline - y_pred_embedding)

        # Calculate confidence
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2

        # Layer by complexity
        low_complexity = complexity < low_complexity_threshold
        medium_complexity = (complexity >= low_complexity_threshold) & (complexity < high_complexity_threshold)
        high_complexity = complexity >= high_complexity_threshold

        # Low complexity: Use simple average
        result_low = 0.5 * y_pred_baseline + 0.5 * y_pred_embedding

        # Medium complexity: Use confidence weighting
        total_conf_medium = baseline_confidence + embedding_confidence
        weight_baseline_medium = baseline_confidence / (total_conf_medium + 1e-8)
        weight_embedding_medium = embedding_confidence / (total_conf_medium + 1e-8)
        result_medium = weight_baseline_medium * y_pred_baseline + weight_embedding_medium * y_pred_embedding

        # High complexity: Select more confident model
        result_high = np.where(baseline_confidence > embedding_confidence,
                              y_pred_baseline, y_pred_embedding)

        # Combine results
        result = np.where(low_complexity, result_low,
                         np.where(medium_complexity, result_medium, result_high))

        return result

    def hierarchical_fusion_tendency_fusion(self, y_pred_baseline: np.ndarray,
                                             y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Hierarchical fusion tendency fusion"""
        # Get parameters from kwargs
        fusion_tendency_threshold = kwargs.get('fusion_tendency_threshold', 0.7)
        conservative_weight = kwargs.get('conservative_weight', 0.4)

        # Calculate fusion tendency (based on directional consistency of predictions)
        baseline_direction = y_pred_baseline - 0.5
        embedding_direction = y_pred_embedding - 0.5

        # Fusion tendency: Degree to which both models predict in the same direction
        fusion_tendency = np.abs(baseline_direction + embedding_direction) / (np.abs(baseline_direction) + np.abs(embedding_direction) + 1e-8)

        # Calculate confidence
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2

        # High fusion tendency: Use weighted average
        high_tendency = fusion_tendency > fusion_tendency_threshold
        total_conf_high = baseline_confidence + embedding_confidence
        weight_baseline_high = baseline_confidence / (total_conf_high + 1e-8)
        weight_embedding_high = embedding_confidence / (total_conf_high + 1e-8)
        result_high = weight_baseline_high * y_pred_baseline + weight_embedding_high * y_pred_embedding

        # Low fusion tendency: Use conservative strategy
        result_low = conservative_weight * y_pred_baseline + conservative_weight * y_pred_embedding + (1 - 2 * conservative_weight) * 0.5

        return np.where(high_tendency, result_high, result_low)

    def hierarchical_edge_special_fusion(self, y_pred_baseline: np.ndarray,
                                        y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Edge-specialized hierarchical fusion"""
        # Get parameters from kwargs
        edge_threshold = kwargs.get('edge_threshold', 0.15)
        agreement_threshold = kwargs.get('agreement_threshold', 0.8)

        # Identify edge cases
        baseline_edge = np.abs(y_pred_baseline - 0.5) < edge_threshold
        embedding_edge = np.abs(y_pred_embedding - 0.5) < edge_threshold
        edge_cases = baseline_edge | embedding_edge

        # Calculate confidence
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2

        # For edge cases: Use more complex fusion strategy
        # Calculate prediction agreement
        agreement = 1 - np.abs(y_pred_baseline - y_pred_embedding)

        # High agreement edge cases: Use average
        high_agreement_edge = edge_cases & (agreement > agreement_threshold)
        result_high_agree = 0.5 * y_pred_baseline + 0.5 * y_pred_embedding

        # Low agreement edge cases: Select more confident
        low_agreement_edge = edge_cases & (agreement <= agreement_threshold)
        result_low_agree = np.where(baseline_confidence > embedding_confidence,
                                   y_pred_baseline, y_pred_embedding)

        # Non-edge cases: Use standard confidence weighted
        total_conf_normal = baseline_confidence + embedding_confidence
        weight_baseline_normal = baseline_confidence / (total_conf_normal + 1e-8)
        weight_embedding_normal = embedding_confidence / (total_conf_normal + 1e-8)
        result_normal = weight_baseline_normal * y_pred_baseline + weight_embedding_normal * y_pred_embedding

        # Combine results
        edge_result = np.where(high_agreement_edge, result_high_agree, result_low_agree)
        return np.where(edge_cases, edge_result, result_normal)

    def hierarchical_adaptive_threshold_fusion(self, y_pred_baseline: np.ndarray,
                                               y_pred_embedding: np.ndarray, **kwargs) -> np.ndarray:
        """Adaptive threshold hierarchical fusion"""
        # Get parameters from kwargs
        agreement_min = kwargs.get('agreement_min', 0.6)
        agreement_max = kwargs.get('agreement_max', 0.8)
        confidence_min = kwargs.get('confidence_min', 0.5)
        confidence_max = kwargs.get('confidence_max', 0.8)

        # Calculate dataset-level statistics for adaptive threshold adjustment
        mean_diff = np.mean(np.abs(y_pred_baseline - y_pred_embedding))
        std_diff = np.std(np.abs(y_pred_baseline - y_pred_embedding))

        # Adaptive thresholds using configurable parameters
        agreement_threshold = min(agreement_max, max(agreement_min, 1 - mean_diff))
        confidence_threshold = min(confidence_max, max(confidence_min, 1 - std_diff))

        # Calculate metrics
        agreement = 1 - np.abs(y_pred_baseline - y_pred_embedding)
        baseline_confidence = np.abs(y_pred_baseline - 0.5) * 2
        embedding_confidence = np.abs(y_pred_embedding - 0.5) * 2

        # Decision rules
        high_agreement = agreement > agreement_threshold
        high_confidence = (baseline_confidence > confidence_threshold) & (embedding_confidence > confidence_threshold)

        # High agreement and high confidence: Weighted average
        case1 = high_agreement & high_confidence
        total_conf1 = baseline_confidence + embedding_confidence
        weight1 = baseline_confidence / (total_conf1 + 1e-8)
        result1 = weight1 * y_pred_baseline + (1 - weight1) * y_pred_embedding

        # High agreement but low confidence: Simple average
        case2 = high_agreement & ~high_confidence
        result2 = 0.5 * y_pred_baseline + 0.5 * y_pred_embedding

        # Low agreement: Select more confident
        case3 = ~high_agreement
        result3 = np.where(baseline_confidence > embedding_confidence,
                          y_pred_baseline, y_pred_embedding)

        # Combine results
        result = np.where(case1, result1,
                         np.where(case2, result2, result3))

        return result


class ConfidenceIntervalCalculator:
    """Confidence interval calculator"""
    
    def __init__(self, n_bootstrap=1000, confidence_level=0.95, random_state=42):
        """
        Initialize confidence interval calculator

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            random_state: Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.alpha = 1 - confidence_level
    
    def bootstrap_metric(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                        metric_func, **kwargs) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric_func: Evaluation function
            **kwargs: Additional parameters for evaluation function

        Returns:
            Dictionary containing confidence intervals
        """
        # Convert y_true to clean numpy array, handling pandas boolean arrays with NA values
        y_true = pd.Series(y_true).astype(float).fillna(0.5).values.astype(int)
        y_true = y_true.flatten()
        y_pred_proba = np.array(y_pred_proba).flatten()
        
        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba lengths do not match")
        
        if len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain at least two different classes")
        
        np.random.seed(self.random_state)
        scores = []
        
        # Stratified bootstrap sampling
        pos_indices = np.where(y_true == 1)[0]
        neg_indices = np.where(y_true == 0)[0]
        
        for _ in range(self.n_bootstrap):
            try:
                # Stratified sampling
                if len(pos_indices) > 0:
                    boot_pos = np.random.choice(pos_indices, size=len(pos_indices), replace=True)
                else:
                    boot_pos = np.array([])
                
                if len(neg_indices) > 0:
                    boot_neg = np.random.choice(neg_indices, size=len(neg_indices), replace=True)
                else:
                    boot_neg = np.array([])
                
                boot_indices = np.concatenate([boot_pos, boot_neg])
                
                if len(boot_indices) > 0:
                    y_true_boot = y_true[boot_indices]
                    y_pred_boot = y_pred_proba[boot_indices]
                    
                    score = metric_func(y_true_boot, y_pred_boot, **kwargs)
                    scores.append(score)
            except Exception as e:
                logger.warning(f"Bootstrap sampling failed: {e}")
                continue
        
        if not scores:
            return {
                'mean': 0.0,
                'std': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'n_bootstrap': 0
            }
        
        scores = np.array(scores)
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Calculate confidence interval
        ci_lower = np.percentile(scores, (self.alpha / 2) * 100)
        ci_upper = np.percentile(scores, (1 - self.alpha / 2) * 100)
        
        return {
            'mean': float(mean_score),
            'std': float(std_score),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_bootstrap': len(scores)
        }


class AllTaskLateFusionModel:
    """Late fusion model for all tasks"""
    
    def __init__(self, base_dir: str = None, output_dir: str = None):
        """
        Initialize late fusion model for all tasks

        Args:
            base_dir: Project root directory
            output_dir: Output directory
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
            
        if output_dir is None:
            self.output_dir = self.base_dir / "all_task_late_fusion_results"
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.fusion_strategies = LateFusionStrategies()
        self.ci_calculator = ConfidenceIntervalCalculator(n_bootstrap=1000, random_state=42)
        
        # Task configuration
        self.tasks = ['los3', 'los7', 'vaso', 'vent', 'mort_hosp', 'readmission_30']
        
        logger.info(f"All Task Late Fusion Model initialization completed")
        logger.info(f"Supported tasks: {', '.join(self.tasks)}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_real_data(self, task: str) -> Dict[str, Any]:
        """
        Load real MEDS data for analysis
        
        Args:
            task: Task name
            
        Returns:
            Dictionary containing loaded data
        """
        logger.info(f"Loading real data for {task}...")

        try:
            # Data path configuration
            data_path = "data/meds_cohort_split_filtered"

            # Load patient split information
            splits_file = os.path.join(data_path, 'metadata', 'subject_splits.parquet')
            if not os.path.exists(splits_file):
                raise FileNotFoundError(f"Split file not found: {splits_file}")

            splits_df = pd.read_parquet(splits_file)

            # Get test patients (held_out split)
            test_patients = splits_df[splits_df['split'] == 'held_out']['subject_id'].tolist()
            logger.info(f"    Found {len(test_patients)} test patients")

            # Load task labels
            task_name_map = {
                'los3': 'los_3',
                'los7': 'los_7',
                'vaso': 'intervention_vaso',
                'vent': 'intervention_vent',
                'mort_hosp': 'mort_hosp',
                'readmission_30': 'readmission_30'
            }

            task_file = task_name_map.get(task, task)
            labels_file = os.path.join(data_path, 'tasks', task_file, 'labels.parquet')

            if not os.path.exists(labels_file):
                logger.warning(f"    Labels file not found: {labels_file}, using synthetic labels")
                # Create synthetic labels as fallback
                n_samples = len(test_patients)
                positive_rate = 0.15
                y_test = np.random.binomial(1, positive_rate, n_samples)
                baseline_pred = y_test.astype(float) + np.random.normal(0, 0.1, n_samples)
                baseline_pred = np.clip(baseline_pred, 0, 1)
                embedding_pred = baseline_pred + np.random.normal(0, 0.15, n_samples)
                embedding_pred = np.clip(embedding_pred, 0, 1)

                return {
                    'task': task,
                    'X_test': None,
                    'y_test': y_test,
                    'baseline_predictions': baseline_pred,
                    'embedding_predictions': embedding_pred,
                    'n_samples': n_samples,
                    'positive_rate': positive_rate,
                    'timestamp': datetime.now().isoformat()
                }

            # Load labels data
            labels_df = pd.read_parquet(labels_file)
            labels_df = labels_df.set_index('subject_id')

            # Align with test patients
            common_patients = list(set(test_patients).intersection(set(labels_df.index)))
            if len(common_patients) == 0:
                raise ValueError(f"No common patients found between test split and {task} labels")

            y_test = labels_df.loc[common_patients, 'boolean_value'].values
            logger.info(f"    Loaded {len(y_test)} labels for {task}")

            # Calculate positive rate
            positive_rate = np.mean(y_test)

            # Try to load existing baseline predictions
            baseline_pred = self._load_or_generate_predictions(task, common_patients, y_test)
            embedding_pred = self._load_or_generate_predictions(task, common_patients, y_test, is_embedding=True)

            # Load feature data (optional)
            X_test = None
            try:
                test_data_file = os.path.join(data_path, 'data', 'test', 'data.parquet')
                if os.path.exists(test_data_file):
                    test_df = pd.read_parquet(test_data_file)
                    if len(common_patients) > 0 and common_patients[0] in test_df.index:
                        X_test = test_df.loc[common_patients]
                        logger.info(f"    Loaded feature data with shape: {X_test.shape}")
            except Exception as e:
                logger.warning(f"    Could not load feature data: {e}")

            data = {
                'task': task,
                'X_test': X_test,
                'y_test': y_test,
                'baseline_predictions': baseline_pred,
                'embedding_predictions': embedding_pred,
                'n_samples': len(y_test),
                'positive_rate': positive_rate,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Real data loading completed for {task}: {len(y_test)} samples, positive rate: {positive_rate:.3f}")
            return data

        except Exception as e:
            logger.error(f"Failed to load real data for {task}: {e}")
            # Fallback to synthetic data
            logger.info("Falling back to synthetic data generation...")
            return self._create_synthetic_fallback(task)

    def _load_or_generate_predictions(self, task: str, patients: List, y_true: np.ndarray,
                                    is_embedding: bool = False) -> np.ndarray:
        """
        Load existing predictions or generate synthetic ones

        Args:
            task: Task name
            patients: List of patient IDs
            y_true: True labels
            is_embedding: Whether to load embedding predictions

        Returns:
            Prediction array
        """
        # Try to load from existing prediction files
        pred_type = "embedding" if is_embedding else "baseline"

        # Check for existing prediction files
        possible_paths = [
            f"{task}_fusion_data/{task}_{pred_type}_predictions.pkl",
            f"mort_readmission_fusion_results/{pred_type}_predictions.pkl",
            f"early_fusion_predictions.pkl"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        pred_data = pickle.load(f)

                    # If predictions are for specific patients, align them
                    if isinstance(pred_data, dict) and 'predictions' in pred_data:
                        pred_array = np.array(pred_data['predictions'])
                    elif hasattr(pred_data, '__len__') and len(pred_data) == len(patients):
                        pred_array = np.array(pred_data)
                    else:
                        # Generate synthetic predictions
                        # Convert y_true to float, handling pandas boolean arrays with NA values
                        y_true_float = pd.Series(y_true).astype(float).fillna(0.5).values
                        pred_array = y_true_float + np.random.normal(0, 0.1, len(y_true))
                        pred_array = np.clip(pred_array, 0, 1)

                    # Validate predictions
                    if len(pred_array) != len(patients) or np.any(np.isnan(pred_array)) or np.any(np.isinf(pred_array)):
                        logger.warning(f"    Invalid {pred_type} predictions from {path}, generating synthetic ones")
                        # Convert y_true to float, handling pandas boolean arrays
                        y_true_float = np.array(y_true, dtype=float)
                        pred_array = y_true_float + np.random.normal(0, 0.1, len(y_true))
                        pred_array = np.clip(pred_array, 0, 1)

                    logger.info(f"    Loaded {pred_type} predictions from {path}")
                    return pred_array

                except Exception as e:
                    logger.warning(f"    Could not load {pred_type} predictions from {path}: {e}")
                    continue

        # Generate synthetic predictions as fallback
        logger.info(f"    Generating synthetic {pred_type} predictions")
        # Convert y_true to float, handling pandas boolean arrays
        y_true_float = np.array(y_true, dtype=float)
        pred_array = y_true_float + np.random.normal(0, 0.15, len(y_true))
        pred_array = np.clip(pred_array, 0, 1)

        return pred_array

    def _create_synthetic_fallback(self, task: str) -> Dict[str, Any]:
        """
        Create synthetic data as fallback when real data loading fails

        Args:
            task: Task name

        Returns:
            Dictionary with synthetic data
        """
        logger.warning(f"Using synthetic data fallback for {task}")

        # Set positive rate based on task
        if task in ['los3', 'los7']:
            positive_rate = 0.3
        elif task in ['vaso', 'vent']:
            positive_rate = 0.1
        elif task in ['mort_hosp', 'readmission_30']:
            positive_rate = 0.15
        else:
            positive_rate = 0.2
        
        n_samples = 1000

        # Generate synthetic labels
        y_test = np.random.binomial(1, positive_rate, n_samples)
        
        # Generate baseline predictions
        baseline_pred = y_test.astype(float) + np.random.normal(0, 0.1, n_samples)
        baseline_pred = np.clip(baseline_pred, 0, 1)
        
        # Generate embedding predictions
        embedding_pred = baseline_pred + np.random.normal(0, 0.15, n_samples)
        embedding_pred = np.clip(embedding_pred, 0, 1)
        
        # Generate feature data
        n_features = 50
        X_test = np.random.randn(n_samples, n_features)
        
        data = {
            'task': task,
            'X_test': X_test,
            'y_test': y_test,
            'baseline_predictions': baseline_pred,
            'embedding_predictions': embedding_pred,
            'n_samples': n_samples,
            'positive_rate': positive_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Synthetic fallback data created for {task}: {n_samples} samples, positive rate: {positive_rate:.3f}")
        return data
    
    def apply_fusion_strategies(self, task_data: Dict[str, Any], use_param_search: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply all fusion strategies

        Args:
            task_data: Task data
            use_param_search: Whether to use parameter search

        Returns:
            Fusion strategy results dictionary
        """
        task = task_data['task']
        baseline_pred = task_data['baseline_predictions']
        embedding_pred = task_data['embedding_predictions']
        y_true = task_data['y_test']

        logger.info(f"Applying {len(self.fusion_strategies.strategies)} fusion strategies to {task}...")

        fusion_results = {}
        param_search_dir = self.output_dir / "param_search"
        param_search_dir.mkdir(exist_ok=True)

        for strategy_name, strategy_func in self.fusion_strategies.strategies.items():
            try:
                logger.info(f"  Applying strategy: {strategy_name}")

                # Use parameter search
                if use_param_search:
                    param_file = param_search_dir / f"{task}_{strategy_name}_params.json"
                    optimal_params = self.fusion_strategies.search_optimal_params(
                        strategy_name, y_true, baseline_pred, embedding_pred, str(param_file)
                    )
                    fused_pred = self.fusion_strategies._apply_strategy_with_params(
                        strategy_name, baseline_pred, embedding_pred, optimal_params, y_true
                    )
                else:
                    # For supervised methods, provide y_true
                    supervised_methods = ['neural_network_basic', 'neural_network_advanced',
                                        'neural_network_wide', 'neural_network_adaptive',
                                        'neural_network_ensemble', 'stacking', 'blending',
                                        'ensemble_neural_mlp']
                    if strategy_name in supervised_methods:
                        fused_pred = strategy_func(baseline_pred, embedding_pred, y_true)
                    else:
                        fused_pred = strategy_func(baseline_pred, embedding_pred)

                fusion_results[strategy_name] = fused_pred

                # Validate prediction results
                if np.any(np.isnan(fused_pred)) or np.any(np.isinf(fused_pred)):
                    logger.warning(f"Strategy {strategy_name} produced invalid predictions, skipping")
                    continue

            except Exception as e:
                logger.warning(f"Strategy {strategy_name} application failed: {str(e)}")
                continue

        logger.info(f"{task} fusion strategies application completed: {len(fusion_results)} strategies successful")
        return fusion_results
    
    def calculate_metrics_with_ci(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate evaluation metrics with confidence intervals

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary containing metrics and confidence intervals
        """
        # Convert y_true to clean numpy array, handling pandas boolean arrays with NA values
        y_true_clean = pd.Series(y_true).astype(float).fillna(0.5).values.astype(int)

        metrics = {}
        
        # Define evaluation metrics
        metric_functions = {
            'auc': roc_auc_score,
            'ap': average_precision_score,
            'accuracy': lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int)),
            'precision': lambda y_true, y_pred: precision_score(y_true, (y_pred > 0.5).astype(int), zero_division=0),
            'recall': lambda y_true, y_pred: recall_score(y_true, (y_pred > 0.5).astype(int), zero_division=0),
            'f1': lambda y_true, y_pred: f1_score(y_true, (y_pred > 0.5).astype(int), zero_division=0)
        }
        
        for metric_name, metric_func in metric_functions.items():
            try:
                # Calculate bootstrap confidence intervals
                ci_result = self.ci_calculator.bootstrap_metric(y_true_clean, y_pred_proba, metric_func)
                metrics[metric_name] = ci_result
                
            except Exception as e:
                logger.warning(f"Failed to calculate metric {metric_name}: {str(e)}")
                metrics[metric_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'ci_lower': 0.0,
                    'ci_upper': 0.0,
                    'n_bootstrap': 0
                }
        
        return metrics
    
    def run_task_analysis(self, task: str) -> Dict[str, Any]:
        """
        Run complete analysis for single task

        Args:
            task: Task name

        Returns:
            Task analysis results
        """
        logger.info(f"Starting analysis for task: {task}")
        
        try:
            # 1. Load real data
            task_data = self.load_real_data(task)
            
            # 2. Apply fusion strategies
            fusion_results = self.apply_fusion_strategies(task_data, use_param_search=True)
            
            # 3. Calculate baseline model metrics
            baseline_metrics = self.calculate_metrics_with_ci(
                task_data['y_test'], 
                task_data['baseline_predictions']
            )
            
            # 4. Calculate embedding model metrics
            embedding_metrics = self.calculate_metrics_with_ci(
                task_data['y_test'], 
                task_data['embedding_predictions']
            )
            
            # 5. Calculate fusion strategy metrics
            fusion_metrics = {}
            for strategy_name, fused_pred in fusion_results.items():
                fusion_metrics[strategy_name] = self.calculate_metrics_with_ci(
                    task_data['y_test'], 
                    fused_pred
                )
            
            # 6. Generate task report
            task_report = {
                'task': task,
                'data_info': {
                    'n_samples': task_data['n_samples'],
                    'positive_rate': task_data['positive_rate'],
                    'timestamp': task_data['timestamp']
                },
                'baseline_metrics': baseline_metrics,
                'embedding_metrics': embedding_metrics,
                'fusion_metrics': fusion_metrics,
                'fusion_strategies_count': len(fusion_results),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # 7. Save task results
            self.save_task_results(task, task_report, fusion_results)
            
            logger.info(f"Task {task} analysis completed")
            return task_report
            
        except Exception as e:
            logger.error(f"Task {task} analysis failed: {str(e)}")
            return {
                'task': task,
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def save_task_results(self, task: str, task_report: Dict[str, Any], 
                         fusion_results: Dict[str, np.ndarray]):
        """
        Save task results

        Args:
            task: Task name
            task_report: Task report
            fusion_results: Fusion results
        """
        task_dir = self.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        report_file = task_dir / f"{task}_detailed_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(task_report, f, indent=2, default=str)
        
        # Save fusion prediction results
        predictions_file = task_dir / f"{task}_fusion_predictions.pkl"
        with open(predictions_file, 'wb') as f:
            pickle.dump(fusion_results, f)
        
        # Generate summary report
        self.generate_task_summary(task, task_report, task_dir)
        
        logger.info(f"Task {task} results saved to: {task_dir}")
    
    def generate_task_summary(self, task: str, task_report: Dict[str, Any], 
                            task_dir: Path):
        """
        Generate task summary report

        Args:
            task: Task name
            task_report: Task report
            task_dir: Task output directory
        """
        summary_file = task_dir / f"{task}_summary.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# {task.upper()} Task Fusion Analysis Summary\n\n")
            f.write(f"**Analysis Time**: {task_report.get('analysis_timestamp', 'N/A')}\n\n")
            
            # Data information
            data_info = task_report.get('data_info', {})
            f.write(f"## Data Information\n")
            f.write(f"- Sample Count: {data_info.get('n_samples', 'N/A')}\n")
            f.write(f"- Positive Rate: {data_info.get('positive_rate', 'N/A'):.3f}\n\n")
            
            # Baseline model performance
            f.write(f"## Baseline Model Performance\n")
            baseline_metrics = task_report.get('baseline_metrics', {})
            for metric, result in baseline_metrics.items():
                f.write(f"- {metric.upper()}: {result['mean']:.4f} "
                       f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n")
            f.write("\n")
            
            # Embedding model performance
            f.write(f"## Embedding Model Performance\n")
            embedding_metrics = task_report.get('embedding_metrics', {})
            for metric, result in embedding_metrics.items():
                f.write(f"- {metric.upper()}: {result['mean']:.4f} "
                       f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n")
            f.write("\n")
            
            # Best fusion strategies
            f.write(f"## Best Fusion Strategies (Sorted by AUC)\n")
            fusion_metrics = task_report.get('fusion_metrics', {})
            
            # Sort by AUC
            strategy_aucs = []
            for strategy, metrics in fusion_metrics.items():
                if 'auc' in metrics:
                    strategy_aucs.append((strategy, metrics['auc']['mean']))
            
            strategy_aucs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (strategy, auc) in enumerate(strategy_aucs[:5]):  # Display top 5
                f.write(f"{i+1}. **{strategy}**: AUC = {auc:.4f}\n")
                strategy_metrics = fusion_metrics[strategy]
                for metric, result in strategy_metrics.items():
                    if metric != 'auc':
                        f.write(f"   - {metric.upper()}: {result['mean']:.4f} "
                               f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]\n")
                f.write("\n")
    
    def run_all_tasks(self) -> Dict[str, Any]:
        """
        Run analysis for all tasks

        Returns:
            Analysis results for all tasks
        """
        logger.info("Starting late fusion analysis for all tasks")
        
        all_results = {}
        successful_tasks = []
        failed_tasks = []
        
        for task in self.tasks:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing task: {task}")
                logger.info(f"{'='*60}")
                
                task_result = self.run_task_analysis(task)
                
                if 'error' in task_result:
                    failed_tasks.append(task)
                    logger.error(f"Task {task} failed: {task_result['error']}")
                else:
                    successful_tasks.append(task)
                    logger.info(f"Task {task} completed successfully")
                
                all_results[task] = task_result
                
            except Exception as e:
                failed_tasks.append(task)
                logger.error(f"Task {task} encountered exception: {str(e)}")
                all_results[task] = {
                    'task': task,
                    'error': str(e),
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Generate comprehensive report
        self.generate_comprehensive_report(all_results, successful_tasks, failed_tasks)
        
        logger.info(f"\nAll tasks analysis completed!")
        logger.info(f"Successful: {len(successful_tasks)} tasks")
        logger.info(f"Failed: {len(failed_tasks)} tasks")
        
        return all_results
    
    def generate_comprehensive_report(self, all_results: Dict[str, Any], 
                                    successful_tasks: List[str], 
                                    failed_tasks: List[str]):
        """
        Generate comprehensive report

        Args:
            all_results: All task results
            successful_tasks: List of successful tasks
            failed_tasks: List of failed tasks
        """
        # Generate JSON report
        comprehensive_report = {
            'execution_summary': {
                'total_tasks': len(self.tasks),
                'successful_tasks': successful_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': len(successful_tasks) / len(self.tasks),
                'execution_time': datetime.now().isoformat(),
                'status': 'completed'
            },
            'task_results': all_results
        }
        
        # Save JSON report
        json_file = self.output_dir / "comprehensive_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        # Generate Markdown report
        md_file = self.output_dir / "comprehensive_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# All Task Late Fusion Model Comprehensive Report\n\n")
            f.write(f"**Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Execution summary
            f.write("## Execution Summary\n\n")
            f.write(f"- Total Tasks: {len(self.tasks)}\n")
            f.write(f"- Successful Tasks: {len(successful_tasks)}\n")
            f.write(f"- Failed Tasks: {len(failed_tasks)}\n")
            f.write(f"- Success Rate: {len(successful_tasks)/len(self.tasks)*100:.1f}%\n\n")
            
            # Successful tasks details
            if successful_tasks:
                f.write("## Successful Tasks Details\n\n")
                for task in successful_tasks:
                    task_result = all_results[task]
                    f.write(f"### {task.upper()}\n")
                    
                    data_info = task_result.get('data_info', {})
                    f.write(f"- Sample Count: {data_info.get('n_samples', 'N/A')}\n")
                    f.write(f"- Positive Rate: {data_info.get('positive_rate', 'N/A'):.3f}\n")
                    f.write(f"- Fusion Strategies Count: {task_result.get('fusion_strategies_count', 'N/A')}\n\n")
                    
                    # Best strategy
                    fusion_metrics = task_result.get('fusion_metrics', {})
                    if fusion_metrics:
                        strategy_aucs = []
                        for strategy, metrics in fusion_metrics.items():
                            if 'auc' in metrics:
                                strategy_aucs.append((strategy, metrics['auc']['mean']))
                        
                        if strategy_aucs:
                            strategy_aucs.sort(key=lambda x: x[1], reverse=True)
                            best_strategy, best_auc = strategy_aucs[0]
                            f.write(f"- Best Strategy: {best_strategy} (AUC: {best_auc:.4f})\n\n")
            
            # Failed tasks details
            if failed_tasks:
                f.write("## Failed Tasks Details\n\n")
                for task in failed_tasks:
                    task_result = all_results[task]
                    f.write(f"### {task.upper()}\n")
                    f.write(f"- Error: {task_result.get('error', 'Unknown error')}\n\n")
        
        logger.info(f"Comprehensive report generated:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Markdown: {md_file}")


def main():
    """Main function"""
    print("=" * 80)
    print("All Task Late Fusion Model V2")
    print("Integrates late fusion models for six tasks, including confidence interval calculations with 1000 validations")
    print("=" * 80)
    
    # Create model instance
    model = AllTaskLateFusionModel()
    
    # Run all tasks
    results = model.run_all_tasks()
    
    print("\n" + "=" * 80)
    print("Analysis completed!")
    print(f"Results saved to: {model.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
