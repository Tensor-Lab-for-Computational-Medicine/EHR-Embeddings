# phase_2_classifiers.py

import pandas as pd
import numpy as np
import logging
import os
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report, 
    confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
import optuna
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CLASSIFIER CONFIGURATION
# =============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for classifier training and evaluation."""
    name: str
    description: str
    use_hyperparameter_tuning: bool = True
    cv_folds: int = 5
    random_state: int = 42
    scale_features: bool = True
    n_bootstrap_samples: int = 1000

# Available classifier configurations
CLASSIFIER_CONFIGS = {
    'logistic_regression': ClassifierConfig(
        name='logistic_regression',
        description='Logistic Regression with L2 regularization',
        use_hyperparameter_tuning=True,
        scale_features=True
    ),
    'logistic_regression_simple': ClassifierConfig(
        name='logistic_regression_simple',
        description='Simple Logistic Regression without tuning',
        use_hyperparameter_tuning=False,
        scale_features=True
    ),
    'xgboost': ClassifierConfig(
        name='xgboost',
        description='XGBoost Gradient Boosting',
        use_hyperparameter_tuning=True,
        scale_features=False
    ),
    'xgboost_simple': ClassifierConfig(
        name='xgboost_simple',
        description='Simple XGBoost without extensive tuning',
        use_hyperparameter_tuning=False,
        scale_features=False
    )
}

# =============================================================================
# CLASSIFIER TRAINING FUNCTIONS
# =============================================================================

class EmbeddingClassifier:
    """Main class for training and evaluating classifiers on embeddings."""
    
    def __init__(self, config_name: str = 'logistic_regression'):
        """
        Initialize the classifier.
        
        Args:
            config_name: Name of the classifier configuration to use
        """
        if config_name not in CLASSIFIER_CONFIGS:
            raise ValueError(f"Unknown classifier config: {config_name}. "
                           f"Available configs: {list(CLASSIFIER_CONFIGS.keys())}")
        
        self.config = CLASSIFIER_CONFIGS[config_name]
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        logging.info(f"Initialized {self.config.description}")
    
    def _create_base_model(self) -> Any:
        """Create the base model based on configuration."""
        if 'logistic' in self.config.name:
            return LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif 'xgboost' in self.config.name:
            return xgb.XGBClassifier(
                random_state=self.config.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type for config: {self.config.name}")
    
    def _tune_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune hyperparameters for Logistic Regression using Optuna."""
        
        def objective(trial):
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            model = LogisticRegression(
                C=C,
                penalty='elasticnet',
                l1_ratio=l1_ratio,
                solver='saga',
                random_state=self.config.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.config.cv_folds, 
                scoring='roc_auc',
                n_jobs=-1
            )
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, timeout=300)
        
        return study.best_params
    
    def _tune_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune hyperparameters for XGBoost using Optuna."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': self.config.random_state,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
            
            # Calculate scale_pos_weight
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            params['scale_pos_weight'] = scale_pos_weight
            
            model = xgb.XGBClassifier(**params)
            
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.config.cv_folds, 
                scoring='roc_auc',
                n_jobs=-1
            )
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, timeout=600)
        
        return study.best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the classifier on embedding features.
        
        Args:
            X_train: Training features (embeddings)
            y_train: Training targets
            
        Returns:
            Dictionary with training results and metadata
        """
        logging.info(f"Training {self.config.description} on {X_train.shape[0]} samples")
        
        # Scale features if required
        if self.config.scale_features:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
        
        training_results = {
            'config_name': self.config.name,
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'class_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
            'hyperparameters': {},
            'cv_scores': {}
        }
        
        # Hyperparameter tuning
        if self.config.use_hyperparameter_tuning:
            logging.info("Performing hyperparameter tuning...")
            
            if 'logistic' in self.config.name:
                best_params = self._tune_logistic_regression(X_train_scaled, y_train)
                self.model = LogisticRegression(
                    **best_params,
                    penalty='elasticnet',
                    solver='saga',
                    random_state=self.config.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                )
            elif 'xgboost' in self.config.name:
                best_params = self._tune_xgboost(X_train_scaled, y_train)
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                best_params['scale_pos_weight'] = scale_pos_weight
                self.model = xgb.XGBClassifier(**best_params)
            
            training_results['hyperparameters'] = best_params
            logging.info(f"Best hyperparameters: {best_params}")
        
        else:
            # Use default parameters
            self.model = self._create_base_model()
            if 'xgboost' in self.config.name:
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Cross-validation before final training
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        training_results['cv_scores'] = {
            'roc_auc_mean': cv_scores.mean(),
            'roc_auc_std': cv_scores.std(),
            'roc_auc_scores': cv_scores.tolist()
        }
        
        # Final training
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        logging.info(f"Training complete. CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return training_results
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Tuple of (predicted_probabilities, predicted_classes)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features if required
        if self.config.scale_features and self.scaler is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        return y_pred_proba, y_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data with uncertainty quantification.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred_proba, y_pred = self.predict(X_test)
        
        # Basic metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        
        # Bootstrap confidence intervals
        def bootstrap_metric(metric_func, y_true, y_pred, n_bootstrap=1000):
            np.random.seed(self.config.random_state)
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Stratified bootstrap
                pos_indices = np.where(y_true == 1)[0]
                neg_indices = np.where(y_true == 0)[0]
                
                n_pos = len(pos_indices)
                n_neg = len(neg_indices)
                
                boot_pos = np.random.choice(pos_indices, size=n_pos, replace=True)
                boot_neg = np.random.choice(neg_indices, size=n_neg, replace=True)
                boot_indices = np.concatenate([boot_pos, boot_neg])
                
                try:
                    score = metric_func(y_true[boot_indices], y_pred[boot_indices])
                    bootstrap_scores.append(score)
                except:
                    continue
            
            return np.array(bootstrap_scores)
        
        # Calculate bootstrap CIs
        roc_auc_bootstrap = bootstrap_metric(roc_auc_score, y_test, y_pred_proba)
        auprc_bootstrap = bootstrap_metric(average_precision_score, y_test, y_pred_proba)
        
        evaluation_results = {
            'n_test_samples': len(y_test),
            'test_class_distribution': dict(zip(*np.unique(y_test, return_counts=True))),
            'metrics': {
                'roc_auc': {
                    'point_estimate': roc_auc,
                    'ci_lower': np.percentile(roc_auc_bootstrap, 2.5),
                    'ci_upper': np.percentile(roc_auc_bootstrap, 97.5),
                    'std': np.std(roc_auc_bootstrap)
                },
                'auprc': {
                    'point_estimate': auprc,
                    'ci_lower': np.percentile(auprc_bootstrap, 2.5),
                    'ci_upper': np.percentile(auprc_bootstrap, 97.5),
                    'std': np.std(auprc_bootstrap)
                }
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'predictions': {
                'y_pred_proba': y_pred_proba.tolist(),
                'y_pred': y_pred.tolist(),
                'y_true': y_test.tolist()
            }
        }
        
        return evaluation_results

# =============================================================================
# EMBEDDING CLASSIFIER EXPERIMENT FUNCTIONS
# =============================================================================

def run_classifier_experiment(
    embeddings_results: Dict[str, Any],
    target_data: pd.Series,
    classifier_configs: List[str],
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run classifier experiments on all embedding combinations.
    
    Args:
        embeddings_results: Results from embedding experiment
        target_data: Target outcomes for classification
        classifier_configs: List of classifier configuration names to test
        output_dir: Directory to save results
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with complete experiment results
    """
    from .phase_2_embeddings import get_embedding_matrix
    
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'classifier_configs': classifier_configs,
            'test_size': test_size,
            'val_size': val_size,
            'random_state': random_state
        },
        'embedding_combinations': list(embeddings_results['embeddings'].keys()),
        'results': {}
    }
    
    total_experiments = len(embeddings_results['embeddings']) * len(classifier_configs)
    experiment_count = 0
    
    # Process each embedding combination
    for combo_key, embeddings_dict in embeddings_results['embeddings'].items():
        logging.info(f"Processing embedding combination: {combo_key}")
        
        # Get patient IDs and convert embeddings to matrix
        patient_ids = list(embeddings_dict.keys())
        
        try:
            X, valid_patient_ids = get_embedding_matrix(embeddings_dict, patient_ids)
            
            # Align target data
            y = target_data.loc[[int(pid) for pid in valid_patient_ids]].values
            
            if len(np.unique(y)) < 2:
                logging.warning(f"Skipping {combo_key}: insufficient class diversity")
                continue
            
            # Split data
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            if val_size > 0:
                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, 
                    random_state=random_state, stratify=y_temp
                )
            else:
                X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
            
            # Test each classifier configuration
            combo_results = {
                'n_total_samples': len(X),
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val) if X_val is not None else 0,
                'n_test_samples': len(X_test),
                'embedding_dim': X.shape[1],
                'train_class_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
                'test_class_distribution': dict(zip(*np.unique(y_test, return_counts=True))),
                'classifiers': {}
            }
            
            for config_name in classifier_configs:
                experiment_count += 1
                logging.info(f"Training {config_name} on {combo_key} "
                           f"({experiment_count}/{total_experiments})")
                
                try:
                    # Initialize and train classifier
                    classifier = EmbeddingClassifier(config_name)
                    
                    # Train on training set
                    training_results = classifier.train(X_train, y_train)
                    
                    # Evaluate on test set
                    evaluation_results = classifier.evaluate(X_test, y_test)
                    
                    # Store results
                    combo_results['classifiers'][config_name] = {
                        'training_results': training_results,
                        'evaluation_results': evaluation_results
                    }
                    
                    # Log key metrics
                    roc_auc = evaluation_results['metrics']['roc_auc']['point_estimate']
                    auprc = evaluation_results['metrics']['auprc']['point_estimate']
                    
                    logging.info(f"  ROC-AUC: {roc_auc:.4f}, AUPRC: {auprc:.4f}")
                    
                    # Save individual classifier
                    classifier_filename = os.path.join(
                        output_dir, f"classifier_{combo_key}_{config_name}.pkl"
                    )
                    with open(classifier_filename, 'wb') as f:
                        pickle.dump(classifier, f)
                
                except Exception as e:
                    logging.error(f"Failed to train {config_name} on {combo_key}: {str(e)}")
                    combo_results['classifiers'][config_name] = {'error': str(e)}
            
            experiment_results['results'][combo_key] = combo_results
            
        except Exception as e:
            logging.error(f"Failed to process embedding combination {combo_key}: {str(e)}")
            experiment_results['results'][combo_key] = {'error': str(e)}
    
    # Save complete results
    results_filename = os.path.join(output_dir, "classifier_experiment_results.pkl")
    with open(results_filename, 'wb') as f:
        pickle.dump(experiment_results, f)
    
    # Create summary report
    summary = create_experiment_summary(experiment_results)
    summary_filename = os.path.join(output_dir, "classifier_experiment_summary.json")
    with open(summary_filename, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    logging.info(f"Classifier experiment complete. Results saved to {output_dir}")
    
    return experiment_results

def create_experiment_summary(experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of the classifier experiment results."""
    summary = {
        'metadata': experiment_results['metadata'],
        'total_combinations': len(experiment_results['embedding_combinations']),
        'successful_combinations': 0,
        'performance_summary': {},
        'best_performers': {},
        'detailed_results': []
    }
    
    all_results = []
    
    for combo_key, combo_results in experiment_results['results'].items():
        if 'error' in combo_results:
            continue
        
        summary['successful_combinations'] += 1
        
        for classifier_name, classifier_results in combo_results.get('classifiers', {}).items():
            if 'error' in classifier_results:
                continue
            
            eval_results = classifier_results['evaluation_results']
            roc_auc = eval_results['metrics']['roc_auc']['point_estimate']
            auprc = eval_results['metrics']['auprc']['point_estimate']
            
            result_record = {
                'combination': combo_key,
                'classifier': classifier_name,
                'roc_auc': roc_auc,
                'auprc': auprc,
                'n_test_samples': eval_results['n_test_samples'],
                'embedding_dim': combo_results['embedding_dim']
            }
            
            all_results.append(result_record)
    
    # Sort by performance
    all_results.sort(key=lambda x: x['roc_auc'], reverse=True)
    
    summary['detailed_results'] = all_results
    summary['best_performers'] = {
        'best_roc_auc': all_results[0] if all_results else None,
        'best_auprc': max(all_results, key=lambda x: x['auprc']) if all_results else None
    }
    
    # Performance statistics
    if all_results:
        roc_aucs = [r['roc_auc'] for r in all_results]
        auprcs = [r['auprc'] for r in all_results]
        
        summary['performance_summary'] = {
            'roc_auc': {
                'mean': np.mean(roc_aucs),
                'std': np.std(roc_aucs),
                'min': np.min(roc_aucs),
                'max': np.max(roc_aucs)
            },
            'auprc': {
                'mean': np.mean(auprcs),
                'std': np.std(auprcs),
                'min': np.min(auprcs),
                'max': np.max(auprcs)
            }
        }
    
    return summary

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_classifier_results(results_path: str) -> Dict[str, Any]:
    """Load classifier experiment results from pickle file."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)

def compare_classifier_performance(
    results: Dict[str, Any],
    metric: str = 'roc_auc'
) -> pd.DataFrame:
    """
    Create a comparison table of classifier performance across all combinations.
    
    Args:
        results: Classifier experiment results
        metric: Metric to compare ('roc_auc' or 'auprc')
        
    Returns:
        DataFrame with performance comparison
    """
    comparison_data = []
    
    for combo_key, combo_results in results['results'].items():
        if 'error' in combo_results:
            continue
        
        for classifier_name, classifier_results in combo_results.get('classifiers', {}).items():
            if 'error' in classifier_results:
                continue
            
            eval_results = classifier_results['evaluation_results']
            metric_result = eval_results['metrics'][metric]
            
            comparison_data.append({
                'combination': combo_key,
                'classifier': classifier_name,
                f'{metric}_point_estimate': metric_result['point_estimate'],
                f'{metric}_ci_lower': metric_result['ci_lower'],
                f'{metric}_ci_upper': metric_result['ci_upper'],
                f'{metric}_std': metric_result['std'],
                'n_test_samples': eval_results['n_test_samples']
            })
    
    return pd.DataFrame(comparison_data)

def plot_performance_comparison(
    results: Dict[str, Any],
    output_path: str,
    metric: str = 'roc_auc'
):
    """
    Create visualization of classifier performance comparison.
    
    Args:
        results: Classifier experiment results
        output_path: Path to save the plot
        metric: Metric to plot ('roc_auc' or 'auprc')
    """
    df = compare_classifier_performance(results, metric)
    
    if df.empty:
        logging.warning("No results to plot")
        return
    
    plt.figure(figsize=(15, 8))
    
    # Create pivot table for heatmap
    pivot_df = df.pivot(index='combination', columns='classifier', 
                       values=f'{metric}_point_estimate')
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', 
                cbar_kws={'label': metric.upper()})
    
    plt.title(f'Classifier Performance Comparison ({metric.upper()})')
    plt.xlabel('Classifier Configuration')
    plt.ylabel('Embedding Combination')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Performance comparison plot saved to {output_path}") 