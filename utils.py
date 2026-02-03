"""
Utility functions for ODMR dataset processing and evaluation.
"""

import numpy as np
import torch
from pathlib import Path


def load_normalization_stats(dataset_dir):
    """
    Load label normalization statistics from dataset.
    
    Parameters:
        dataset_dir : str or Path
            Path to dataset directory
            
    Returns:
        dict with keys 'labels_mean' and 'labels_std'
    """
    stats_path = Path(dataset_dir) / 'normalization_stats.npy'
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found at {stats_path}")
    
    stats = np.load(stats_path, allow_pickle=True).item()
    return stats


def denormalize_labels(labels_norm, labels_mean, labels_std):
    """
    Denormalize labels back to original scale.
    
    Parameters:
        labels_norm : array or tensor (N, 3)
            Normalized labels (Ax, Ay, Az)
        labels_mean : array (3,)
            Mean used for normalization
        labels_std : array (3,)
            Std used for normalization
            
    Returns:
        Denormalized labels in original scale
    """
    if isinstance(labels_norm, torch.Tensor):
        labels_mean = torch.tensor(labels_mean, device=labels_norm.device, dtype=labels_norm.dtype)
        labels_std = torch.tensor(labels_std, device=labels_norm.device, dtype=labels_norm.dtype)
        return labels_norm * labels_std + labels_mean
    else:
        return labels_norm * labels_std + labels_mean


def normalize_labels(labels, labels_mean, labels_std):
    """
    Normalize labels to zero mean and unit variance.
    
    Parameters:
        labels : array or tensor (N, 3)
            Original labels (Ax, Ay, Az)
        labels_mean : array (3,)
            Mean to use for normalization
        labels_std : array (3,)
            Std to use for normalization
            
    Returns:
        Normalized labels
    """
    if isinstance(labels, torch.Tensor):
        labels_mean = torch.tensor(labels_mean, device=labels.device, dtype=labels.dtype)
        labels_std = torch.tensor(labels_std, device=labels.device, dtype=labels.dtype)
        return (labels - labels_mean) / (labels_std + 1e-8)
    else:
        return (labels - labels_mean) / (labels_std + 1e-8)
