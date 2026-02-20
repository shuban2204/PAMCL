# ABOUTME: Utility functions for PAMCL training and evaluation
# ABOUTME: Configuration loading, metrics computation, checkpointing

import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml
import numpy as np
import torch
from sklearn.metrics import roc_curve


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str, name: str = "pamcl") -> logging.Logger:
    """Setup logging to file and console.
    
    Args:
        log_dir: Directory for log files
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER).
    
    The EER is the point where False Acceptance Rate equals False Rejection Rate.
    
    Args:
        y_true: Ground truth binary labels (0 = real, 1 = fake)
        y_scores: Predicted scores (higher = more likely fake)
        
    Returns:
        Tuple of (EER value, threshold at EER)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find the point where FPR equals FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return float(eer), float(eer_threshold)


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted scores
        
    Returns:
        Dictionary with EER, AUC, and accuracy
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    eer, threshold = compute_eer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    # Accuracy at EER threshold
    y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'eer': eer,
        'eer_threshold': threshold,
        'auc': auc,
        'accuracy': acc
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: Optional[str] = None
) -> str:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoints
        filename: Optional custom filename
        
    Returns:
        Path to saved checkpoint
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch{epoch}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, Dict[str, float]]:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load to
        
    Returns:
        Tuple of (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return model, optimizer, epoch, metrics


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get the appropriate device based on config and availability.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device object
    """
    device_str = config.get('training', {}).get('device', 'cuda')
    
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
