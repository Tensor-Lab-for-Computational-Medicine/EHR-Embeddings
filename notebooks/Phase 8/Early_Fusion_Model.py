
"""
Embeddings Data Loading (lines ~240-250)
Numerical Data Loading (lines ~180-190)
FilePath  (lines ~76-80)
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

# GPU Configuration for IHC H200-1 Server
def configure_gpu_for_ihc_h200():
    """
    Configure GPU settings optimized for IHC H200-1 server

    """
    try:
        # Set GPU device
        if torch.cuda.is_available():
            # Configure for IHC H200-1
            torch.cuda.set_device(0)  # Primary GPU
            device = torch.device('cuda:0')
            
            # Set memory optimization for H200
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Configure for mixed precision if available
            if hasattr(torch.cuda, 'amp') and torch.cuda.amp.is_autocast_available():
                torch.cuda.amp.set_autocast_enabled(True)
            
            print(f"[GPU] Configured IHC H200-1 GPU: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return device
        else:
            print("[GPU] CUDA not available, using CPU")
            return torch.device('cpu')
    except Exception as e:
        print(f"[GPU] Error configuring IHC H200-1: {e}")
        return torch.device('cpu')

# Initialize GPU configuration
GPU_DEVICE = configure_gpu_for_ihc_h200()

def discover_all_combinations_for_task(task):

    combinations = []
    
    # Base path for embeddings 
    embeddings_base_path = "../notebooks/Phase 4"
    
    # Model directory mapping 
    model_dir_mapping = {
        "text-embedding-large-exp-03-07": "embeddings_text-embedding-large-exp-03-07",
        "text-embedding-004": "embeddings_models_text-embedding-004",
        "text-embedding-005": "embeddings_text-embedding-005",
        "embedding-001": "embeddings_models_embedding-001",
        "MedEmbed-small": "embeddings_abhinand_MedEmbed-small-v0.1",
        "text-embedding-004-classification": "embeddings_models_text-embedding-004",  # Add more mappings as needed
    }
    
    # Reverse mapping for directory to model name
    dir_to_model_mapping = {v: k for k, v in model_dir_mapping.items()}
    
    try:
        if not os.path.exists(embeddings_base_path):
            print(f"[WARNING] Embeddings base path not found: {embeddings_base_path}")
            return []
        
        # Scan all model directories
        for item in os.listdir(embeddings_base_path):
            model_dir_path = os.path.join(embeddings_base_path, item)
            
            if not os.path.isdir(model_dir_path):
                continue
            
            # Get model name from directory
            if item in dir_to_model_mapping:
                model_name = dir_to_model_mapping[item]
            else:
                # Try to infer model name from directory name
                model_name = item.replace("embeddings_", "").replace("embeddings_models_", "")
            
            # Check if this model directory has the specific task
            task_path = os.path.join(model_dir_path, task)
            if not os.path.exists(task_path):
                continue
            
            # Scan all arm directories within the task
            for arm_item in os.listdir(task_path):
                arm_path = os.path.join(task_path, arm_item)
                
                if not os.path.isdir(arm_path):
                    continue
                
                # Check if this arm has the required data splits
                has_train = os.path.exists(os.path.join(arm_path, "train"))
                has_val = os.path.exists(os.path.join(arm_path, "val"))
                has_test = os.path.exists(os.path.join(arm_path, "test"))
                
                if has_train and has_val and has_test:
                    # Try to get baseline score from existing results
                    baseline_score = 0.5  # Default baseline
                    
                    # Try to find baseline score from previous results
                    try:
                        # Look for existing results file
                        results_file = f"{task}_complete_1000_bootstrap_results/{task}_summary.json"
                        if os.path.exists(results_file):
                            with open(results_file, 'r') as f:
                                results_data = json.load(f)
                                # Find matching configuration
                                for result in results_data.get('all_results', []):
                                    if (result.get('model') == model_name and 
                                        result.get('arm') == arm_item):
                                        baseline_score = result.get('auc', 0.5)
                                        break
                    except:
                        pass
                    
                    combinations.append({
                        "model": model_name,
                        "arm": arm_item,
                        "baseline_score": baseline_score
                    })
                    
                    print(f"[DISCOVERY] Found combination: {model_name} / {arm_item} for {task}")
    
    except Exception as e:
        print(f"[ERROR] Failed to discover combinations for {task}: {e}")
    
    # Sort by baseline score (descending)
    combinations.sort(key=lambda x: x['baseline_score'], reverse=True)
    
    print(f"[DISCOVERY] Found {len(combinations)} combinations for {task}")
    return combinations

# Task configurations - NOW DYNAMICALLY GENERATED
TASK_CONFIGS = {}

# Define all tasks
ALL_TASKS = [
    "intervention_vaso",
    "intervention_vent", 
    "los_3",
    "los_7",
    "mort_hosp",
    "readmission_30"
]

def initialize_task_configs():
    """Initialize task configurations by discovering all available combinations"""
    global TASK_CONFIGS
    
    print(" Discovering all available combinations for each task...")
    
    for task in ALL_TASKS:
        combinations = discover_all_combinations_for_task(task)
        
        if combinations:
            TASK_CONFIGS[task] = {
                "task": task,
                "all_combinations": combinations  # Changed from top_combinations to all_combinations
            }
            print(f" {task}: Found {len(combinations)} combinations")
        else:
            print(f"  {task}: No combinations found")
            # Fallback to original top combinations if discovery fails
            TASK_CONFIGS[task] = {
                "task": task,
                "all_combinations": []  # Will be handled in the code
            }

class EarlyFusionConfig:
    def __init__(self, task_config):
        self.task = task_config["task"]
        self.all_combinations = task_config["all_combinations"]
        self.output_dir = f"{self.task}_complete_all_combinations_1000_bootstrap_results"
        os.makedirs(self.output_dir, exist_ok=True)

        # Hyperparameters for 1000 bootstrap
        self.n_bootstrap = 1000
        self.confidence_level = 0.95
        self.n_trials = 50
        self.epochs = 100
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.patience = 20

def calculate_bootstrap_ci(y_true, y_pred_proba, n_iterations=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for AUROC and AUPRC"""
    auc_scores = []
    auprc_scores = []
    accuracy_scores = []
    f1_scores = []

    n_samples = len(y_true)

    for i in range(n_iterations):
        # Bootstrap resampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]

        try:
            # Calculate metrics
            auc = roc_auc_score(y_true_boot, y_pred_boot)
            auprc = average_precision_score(y_true_boot, y_pred_boot)
            y_pred_binary = (y_pred_boot >= 0.5).astype(int)
            accuracy = accuracy_score(y_true_boot, y_pred_binary)
            f1 = f1_score(y_true_boot, y_pred_binary)

            auc_scores.append(auc)
            auprc_scores.append(auprc)
            accuracy_scores.append(accuracy)
            f1_scores.append(f1)

        except Exception as e:
            print(f"[BOOTSTRAP] Error in iteration {i}: {e}")
            continue

    if len(auc_scores) == 0:
        return {
            'auc_ci_lower': np.nan, 'auc_ci_upper': np.nan,
            'auprc_ci_lower': np.nan, 'auprc_ci_upper': np.nan,
            'accuracy_ci_lower': np.nan, 'accuracy_ci_upper': np.nan,
            'f1_ci_lower': np.nan, 'f1_ci_upper': np.nan,
            'n_bootstrap': 0
        }

    alpha = 1 - confidence_level
    auc_ci_lower = np.percentile(auc_scores, alpha/2 * 100)
    auc_ci_upper = np.percentile(auc_scores, (1 - alpha/2) * 100)
    auprc_ci_lower = np.percentile(auprc_scores, alpha/2 * 100)
    auprc_ci_upper = np.percentile(auprc_scores, (1 - alpha/2) * 100)
    accuracy_ci_lower = np.percentile(accuracy_scores, alpha/2 * 100)
    accuracy_ci_upper = np.percentile(accuracy_scores, (1 - alpha/2) * 100)
    f1_ci_lower = np.percentile(f1_scores, alpha/2 * 100)
    f1_ci_upper = np.percentile(f1_scores, (1 - alpha/2) * 100)

    return {
        'auc_ci_lower': auc_ci_lower, 'auc_ci_upper': auc_ci_upper,
        'auprc_ci_lower': auprc_ci_lower, 'auprc_ci_upper': auprc_ci_upper,
        'accuracy_ci_lower': accuracy_ci_lower, 'accuracy_ci_upper': accuracy_ci_upper,
        'f1_ci_lower': f1_ci_lower, 'f1_ci_upper': f1_ci_upper,
        'n_bootstrap': len(auc_scores)
    }

def load_numerical_features_for_task(task):
    """Load numerical features for specific task
    
    DATA LOADING LOCATION - MODIFY THESE PATHS IF YOUR DATA LOCATION CHANGES:
    """
    try:
        # ===== MODIFY THESE PATHS IF YOUR NUMERICAL DATA LOCATION CHANGES =====
        # Current location: ../notebooks/Phase 1 and 2/phase_1_outputs/
        numerical_data_base_path = "../notebooks/Phase 1 and 2/phase_1_outputs"
        base_name = "preprocessed_mort_hosp_los_3_los_7_readmission_30_intervention_vent_intervention_vaso_trends_True_window_24_gap_6_seed_42"
        
        # Feature file paths - MODIFY THESE IF YOUR FILE NAMING CHANGES
        X_train_path = f"{numerical_data_base_path}/{base_name}_X_train.pkl"
        X_val_path = f"{numerical_data_base_path}/{base_name}_X_val.pkl"
        X_test_path = f"{numerical_data_base_path}/{base_name}_X_test.pkl"
        
        # Label file paths - MODIFY THESE IF YOUR FILE NAMING CHANGES
        y_train_path = f"{numerical_data_base_path}/{base_name}_y_train.pkl"
        y_val_path = f"{numerical_data_base_path}/{base_name}_y_val.pkl"
        y_test_path = f"{numerical_data_base_path}/{base_name}_y_test.pkl"
        # ===== END OF PATH MODIFICATION SECTION =====

        # Load data using pickle
        import pickle
        with open(X_train_path, 'rb') as f:
            X_train_full = pickle.load(f)
        with open(X_val_path, 'rb') as f:
            X_val = pickle.load(f)
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)

        with open(y_train_path, 'rb') as f:
            y_train_full = pickle.load(f)
        with open(y_val_path, 'rb') as f:
            y_val = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)

        print(f"[INFO] Loaded numerical features from Phase 1 outputs")
        print(f"[INFO] Feature shapes - Train: {X_train_full.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Extract task-specific labels from DataFrame
        if hasattr(y_train_full, task):
            y_train = y_train_full[task].values
            y_val = y_val[task].values
            y_test = y_test[task].values
        else:
            print(f"[ERROR] Task {task} not found in labels DataFrame")
            print(f"[INFO] Available tasks: {list(y_train_full.columns)}")
            return None

        # Convert features to numpy if needed
        if hasattr(X_train_full, 'values'):
            X_train_full = X_train_full.values
            X_val = X_val.values
            X_test = X_test.values

        # Convert to numpy arrays
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

        # Remove NaN values
        train_valid = ~np.isnan(y_train)
        val_valid = ~np.isnan(y_val)
        test_valid = ~np.isnan(y_test)

        return {
            'X_train': X_train_full[train_valid],
            'y_train': y_train[train_valid],
            'X_val': X_val[val_valid],
            'y_val': y_val[val_valid],
            'X_test': X_test[test_valid],
            'y_test': y_test[test_valid]
        }
    except Exception as e:
        print(f"[ERROR] Failed to load numerical features for {task}: {e}")
        return None

def load_embeddings_for_task(model, arm, task):
    """Load semantic embeddings for specific model, arm, and task
    
    DATA LOADING LOCATION - MODIFY THESE PATHS IF YOUR EMBEDDINGS LOCATION CHANGES:
    """
    try:
        # ===== MODIFY THESE PATHS IF YOUR EMBEDDINGS LOCATION CHANGES =====
        # Current location: ../notebooks/Phase 4/
        embeddings_base_path = "../notebooks/Phase 4"
        
        # Model directory mapping - ADD NEW MODELS HERE
        model_dir_mapping = {
            "text-embedding-large-exp-03-07": "embeddings_text-embedding-large-exp-03-07",
            "text-embedding-004": "embeddings_models_text-embedding-004",
            "text-embedding-005": "embeddings_text-embedding-005",
            "embedding-001": "embeddings_models_embedding-001",
            "MedEmbed-small": "embeddings_abhinand_MedEmbed-small-v0.1",
            "text-embedding-004-classification": "embeddings_models_text-embedding-004",
        }
        
        # Get model directory
        if model in model_dir_mapping:
            model_dir = model_dir_mapping[model]
        else:
            # Try to infer directory name
            model_dir = f"embeddings_{model}"
        
        # Construct full path - MODIFY THIS IF YOUR DIRECTORY STRUCTURE CHANGES
        base_path = f"{embeddings_base_path}/{model_dir}/{arm}"
        # ===== END OF PATH MODIFICATION SECTION =====

        # Load train embeddings
        train_embeddings = []
        train_path = os.path.join(base_path, "train")
        if os.path.exists(train_path):
            for file in os.listdir(train_path):
                if file.endswith('.npy') and not file.startswith('._'):
                    file_path = os.path.join(train_path, file)
                    emb = np.load(file_path, allow_pickle=True)
                    if emb.ndim > 1:
                        emb = emb.squeeze()
                    train_embeddings.append(emb)
        X_train_emb = np.array(train_embeddings) if train_embeddings else None

        # Load val embeddings
        val_embeddings = []
        val_path = os.path.join(base_path, "val")
        if os.path.exists(val_path):
            for file in os.listdir(val_path):
                if file.endswith('.npy') and not file.startswith('._'):
                    file_path = os.path.join(val_path, file)
                    emb = np.load(file_path, allow_pickle=True)
                    if emb.ndim > 1:
                        emb = emb.squeeze()
                    val_embeddings.append(emb)
        X_val_emb = np.array(val_embeddings) if val_embeddings else None

        # Load test embeddings
        test_embeddings = []
        test_path = os.path.join(base_path, "test")
        if os.path.exists(test_path):
            for file in os.listdir(test_path):
                if file.endswith('.npy') and not file.startswith('._'):
                    file_path = os.path.join(test_path, file)
                    emb = np.load(file_path, allow_pickle=True)
                    if emb.ndim > 1:
                        emb = emb.squeeze()
                    test_embeddings.append(emb)
        X_test_emb = np.array(test_embeddings) if test_embeddings else None

        return {
            'X_train_emb': X_train_emb,
            'X_val_emb': X_val_emb,
            'X_test_emb': X_test_emb
        }
    except Exception as e:
        print(f"[ERROR] Failed to load embeddings for {model}/{arm}: {e}")
        return None

class LearnableFusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super(LearnableFusionLayer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

def train_fusion_layer(X_train_combined, epochs=100, batch_size=256, learning_rate=1e-3, patience=20):
    """Train the learnable fusion layer"""
    # Use IHC H200-1 GPU configuration
    device = GPU_DEVICE

    # Create autoencoder
    input_dim = X_train_combined.shape[1]
    model = LearnableFusionLayer(input_dim).to(device)

    # Create DataLoader
    dataset = TensorDataset(torch.FloatTensor(X_train_combined))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model

def run_experiment_for_config(config, combination, fusion_method, numerical_data):
    """Run single experiment configuration"""
    try:
        task = config.task
        model = combination['model']
        arm = combination['arm']
        baseline_score = combination['baseline_score']

        # Load embeddings
        embeddings = load_embeddings_for_task(model, arm, task)
        if embeddings is None:
            return None

        # Align data samples
        X_train_num = numerical_data['X_train']
        y_train = numerical_data['y_train']
        X_train_emb = embeddings['X_train_emb']

        X_val_num = numerical_data['X_val']
        y_val = numerical_data['y_val']
        X_val_emb = embeddings['X_val_emb']

        X_test_num = numerical_data['X_test']
        y_test = numerical_data['y_test']
        X_test_emb = embeddings['X_test_emb']

        # Find minimum samples across all splits
        train_samples = min(len(X_train_num), len(X_train_emb))
        val_samples = min(len(X_val_num), len(X_val_emb))
        test_samples = min(len(X_test_num), len(X_test_emb))

        # Truncate to minimum samples
        X_train_num = X_train_num[:train_samples]
        y_train = y_train[:train_samples]
        X_train_emb = X_train_emb[:train_samples]

        X_val_num = X_val_num[:val_samples]
        y_val = y_val[:val_samples]
        X_val_emb = X_val_emb[:val_samples]

        X_test_num = X_test_num[:test_samples]
        y_test = y_test[:test_samples]
        X_test_emb = X_test_emb[:test_samples]

        # Standardize numerical features
        scaler = StandardScaler()
        X_train_num_scaled = scaler.fit_transform(X_train_num)
        X_val_num_scaled = scaler.transform(X_val_num)
        X_test_num_scaled = scaler.transform(X_test_num)

        # Fuse features based on method
        if fusion_method == "Concatenation":
            X_train_combined = np.concatenate([X_train_num_scaled, X_train_emb], axis=1)
            X_val_combined = np.concatenate([X_val_num_scaled, X_val_emb], axis=1)
            X_test_combined = np.concatenate([X_test_num_scaled, X_test_emb], axis=1)
        elif fusion_method == "FusionLayer":
            # Train fusion layer
            X_train_temp = np.concatenate([X_train_num_scaled, X_train_emb], axis=1)
            fusion_model = train_fusion_layer(X_train_temp, epochs=config.epochs)

            # Apply fusion
            device = GPU_DEVICE
            fusion_model.eval()

            X_train_combined = fusion_model(torch.FloatTensor(X_train_temp).to(device)).cpu().detach().numpy()
            X_val_combined = fusion_model(torch.FloatTensor(
                np.concatenate([X_val_num_scaled, X_val_emb], axis=1)).to(device)).cpu().detach().numpy()
            X_test_combined = fusion_model(torch.FloatTensor(
                np.concatenate([X_test_num_scaled, X_test_emb], axis=1)).to(device)).cpu().detach().numpy()

        # ===== IHC H200-1 GPU OPTIMIZATION FOR XGBOOST =====
        def objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                # GPU configuration optimized for IHC H200-1
                'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
                'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
                'gpu_id': 0 if torch.cuda.is_available() else None,
                'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor',
                # Memory optimization for H200
                'max_bin': 256,  # Optimized for H200 memory
                'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
            }

            model = xgb.XGBClassifier(**param)
            model.fit(X_train_combined, y_train,
                     eval_set=[(X_val_combined, y_val)],
                     early_stopping_rounds=50,
                     verbose=False)

            y_pred_proba = model.predict_proba(X_val_combined)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)

        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=config.n_trials)

        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            # IHC H200-1 GPU configuration
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'gpu_id': 0 if torch.cuda.is_available() else None,
            'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor',
            'max_bin': 256,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        })

        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train_combined, y_train,
                       eval_set=[(X_val_combined, y_val)],
                       early_stopping_rounds=50,
                       verbose=False)

        # Make predictions
        y_pred_proba = final_model.predict_proba(X_test_combined)[:, 1]
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)

        # Calculate bootstrap confidence intervals
        bootstrap_results = calculate_bootstrap_ci(y_test, y_pred_proba,
                                                n_iterations=config.n_bootstrap,
                                                confidence_level=config.confidence_level)

        return {
            'task': task,
            'config_name': f"{model}_{arm}",
            'fusion_method': fusion_method,
            'model': model,
            'arm': arm,
            'baseline_score': baseline_score,
            'auc': auc,
            'auprc': auprc,
            'accuracy': accuracy,
            'f1': f1,
            'numerical_features': X_train_num.shape[1],
            'semantic_features': X_train_emb.shape[1],
            'fused_features': X_train_combined.shape[1],
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            **bootstrap_results
        }

    except Exception as e:
        print(f"[ERROR] Failed to run experiment for {combination} + {fusion_method}: {e}")
        return None

def run_complete_experiment(task_config):
    """Run complete experiment for a task"""
    config = EarlyFusionConfig(task_config)
    task = config.task

    print(f"\n{'='*50}")
    print(f"Running complete experiment for {task}")
    print(f"{'='*50}")

    # Load numerical data
    numerical_data = load_numerical_features_for_task(task)
    if numerical_data is None:
        print(f"[ERROR] Could not load numerical data for {task}")
        return None

    all_results = []

    # Run experiments for ALL combinations and fusion methods
    combinations = config.all_combinations
    if not combinations:
        print(f"[WARNING] No combinations found for {task}, skipping...")
        return None

    print(f"[INFO] Running {len(combinations)} combinations for {task}")

    for i, combination in enumerate(combinations):
        group_name = f"Config{i+1}"

        for fusion_method in ["Concatenation", "FusionLayer"]:
            print(f"\n[{task}] Testing {group_name} + {fusion_method}")
            print(f"Model: {combination['model']}, Arm: {combination['arm']}")

            result = run_experiment_for_config(config, combination, fusion_method, numerical_data)

            if result:
                result['group'] = group_name
                all_results.append(result)
                print(".4f")
            else:
                print(f"[ERROR] Failed to run {group_name} + {fusion_method}")

    if not all_results:
        print(f"[ERROR] No results generated for {task}")
        return None

    # Create summary statistics
    fusion_method_summary = {}
    experiment_group_summary = {}

    # Group by fusion method
    for method in ["Concatenation", "FusionLayer"]:
        method_results = [r for r in all_results if r['fusion_method'] == method]
        if method_results:
            fusion_method_summary[method] = {
                'mean': np.mean([r['auc'] for r in method_results]),
                'std': np.std([r['auc'] for r in method_results]),
                'max': np.max([r['auc'] for r in method_results])
            }

    # Group by experiment group (now we have all combinations, not just top 3)
    unique_groups = sorted(set([r['group'] for r in all_results]))
    for group in unique_groups:
        group_results = [r for r in all_results if r['group'] == group]
        if group_results:
            experiment_group_summary[group] = {
                'mean': np.mean([r['auc'] for r in group_results]),
                'std': np.std([r['auc'] for r in group_results]),
                'max': np.max([r['auc'] for r in group_results])
            }

    # Find best configuration
    best_result = max(all_results, key=lambda x: x['auc'])

    # Create summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'task': task,
        'total_experiments': len(all_results),
        'total_combinations_tested': len(combinations),
        'best_configuration': best_result,
        'fusion_method_summary': {
            'mean': {k: v['mean'] for k, v in fusion_method_summary.items()},
            'std': {k: v['std'] for k, v in fusion_method_summary.items()},
            'max': {k: v['max'] for k, v in fusion_method_summary.items()}
        },
        'experiment_group_summary': {
            'mean': {k: v['mean'] for k, v in experiment_group_summary.items()},
            'std': {k: v['std'] for k, v in experiment_group_summary.items()},
            'max': {k: v['max'] for k, v in experiment_group_summary.items()}
        },
        'bootstrap_config': {
            'n_iterations': config.n_bootstrap,
            'confidence_level': config.confidence_level,
            'random_state': 42
        },
        'gpu_config': {
            'device': str(GPU_DEVICE),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        },
        'all_results': all_results
    }

    # Save summary
    summary_path = os.path.join(config.output_dir, f"{task}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed results as CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(config.output_dir, f"{task}_complete_all_combinations_1000_bootstrap_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'='*50}")
    print(f"SUMMARY FOR {task.upper()}")
    print(f"{'='*50}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Total combinations tested: {len(combinations)}")
    print(".4f")
    print(".4f")
    print(".4f")
    print(f"\nBest configuration: {best_result['config_name']} + {best_result['fusion_method']}")
    print(".4f")
    print(f"Results saved to: {config.output_dir}")

    return summary

def main():
    """Main function to run all tasks"""
    print(" Starting Complete Early Fusion Experiments with ALL Combinations")
    print(" Testing all available model/arm combinations instead of just top 3")
    print("  Optimized for IHC H200-1 GPU Server")
    print("=" * 70)

    # Initialize task configurations by discovering all combinations
    initialize_task_configs()

    all_summaries = {}

    for task_name, task_config in TASK_CONFIGS.items():
        print(f"\n Processing task: {task_name}")
        summary = run_complete_experiment(task_config)

        if summary:
            all_summaries[task_name] = summary
            print(f" {task_name} completed successfully")
        else:
            print(f" {task_name} failed")

    print(f"\n{'='*70}")
    print(" ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*70}")

    # Print final summary
    print("\n FINAL RESULTS SUMMARY:")
    print("-" * 50)

    for task_name, summary in all_summaries.items():
        best_config = summary['best_configuration']
        print("30"
              "4.1f")

    print("\n Results saved in individual task directories")
    print(" Each task has:")
    print("   - {task}_summary.json (detailed results with confidence intervals)")
    print("   - {task}_complete_all_combinations_1000_bootstrap_results.csv (CSV format)")
    print("\n Data loading locations are marked in the code for easy modification")
    print("  GPU configuration optimized for IHC H200-1 server")

if __name__ == '__main__':
    main()