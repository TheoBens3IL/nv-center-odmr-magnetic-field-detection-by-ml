"""
Utility functions for ODMR dataset processing, training and evaluation.
"""

import numpy as np
import torch
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt


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


def spherical_to_cartesian(spherical):
    """
    Convert spherical coordinates (Ar, theta, phi) to cartesian (Ax, Ay, Az).
    Parameters:
        spherical: array-like or tensor (..., 3) with [Ar, theta, phi]
    Returns:
        cartesian: same shape (..., 3) with [Ax, Ay, Az]
    """
    if isinstance(spherical, torch.Tensor):
        Ar = spherical[..., 0]
        theta = spherical[..., 1]
        phi = spherical[..., 2]
        Ax = Ar * torch.sin(theta) * torch.cos(phi)
        Ay = Ar * torch.sin(theta) * torch.sin(phi)
        Az = Ar * torch.cos(theta)
        return torch.stack([Ax, Ay, Az], dim=-1)
    else:
        Ar = spherical[..., 0]
        theta = spherical[..., 1]
        phi = spherical[..., 2]
        Ax = Ar * np.sin(theta) * np.cos(phi)
        Ay = Ar * np.sin(theta) * np.sin(phi)
        Az = Ar * np.cos(theta)
        return np.stack([Ax, Ay, Az], axis=-1)


def plot_training_history(history, label_names=['Ax', 'Ay', 'Az'], show=True):
    """Plot training and validation metrics over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics Evolution', fontsize=16, fontweight='bold')

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. MAE per axis
    ax = axes[0, 1]
    mae_keys = [f"mae_{name.lower()}" for name in label_names]
    for i, k in enumerate(mae_keys):
        ax.plot(epochs, history[k], label=label_names[i], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (unit)')
    ax.set_title('Mean Absolute Error by Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. MAE moyen
    ax = axes[1, 0]
    avg_mae = [sum(history[k][i] for k in mae_keys) / len(mae_keys) for i in range(len(epochs))]
    ax.plot(epochs, avg_mae, label='Avg MAE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (unit)')
    ax.set_title('Average MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Physics loss
    ax = axes[1, 1]
    ax.plot(epochs, history['physics_loss'], label='Physics Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Physics Loss')
    ax.set_title('Physics Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# def compute_direction_zones(vectors, n_theta_bins=4, n_phi_bins=12):
#     """
#     Discretize 3D vectors into directional zones on the sphere.

#     The sphere is partitioned into n_theta_bins in polar angle θ (0→π)
#     and n_phi_bins in azimuth φ (-π→π), giving n_theta_bins * n_phi_bins zones.

#     This is intended to approximate different alignment regions of the
#     magnetic field with respect to the NV axes.

#     Parameters
#     ----------
#     vectors : array-like or tensor of shape (..., 3)
#         Cartesian components (Ax, Ay, Az) or (Bx, By, Bz) in any units.
#     n_theta_bins : int
#         Number of bins in polar angle θ (default 4).
#     n_phi_bins : int
#         Number of bins in azimuth φ (default 12).

#     Returns
#     -------
#     zones : ndarray or tensor of dtype int64, shape (...)
#         Integer zone indices in [0, n_theta_bins * n_phi_bins - 1].
#     """
#     # Handle both numpy arrays and torch tensors
#     is_tensor = isinstance(vectors, torch.Tensor)

#     if is_tensor:
#         v = vectors.detach().cpu().numpy()
#     else:
#         v = np.asarray(vectors, dtype=np.float64)

#     if v.shape[-1] != 3:
#         raise ValueError(f"Expected last dimension of size 3 for vectors, got shape {v.shape}")

#     # Compute spherical angles from Cartesian components
#     x = v[..., 0]
#     y = v[..., 1]
#     z = v[..., 2]

#     r = np.sqrt(x * x + y * y + z * z)
#     theta = np.zeros_like(r)
#     phi = np.zeros_like(r)

#     # Avoid division by zero for near-zero vectors
#     mask = r > 1e-8
#     if np.any(mask):
#         theta[mask] = np.arccos(np.clip(z[mask] / r[mask], -1.0, 1.0))
#         phi[mask] = np.arctan2(y[mask], x[mask])

#     # Bin θ in [0, π] into n_theta_bins
#     theta_norm = theta / np.pi  # in [0, 1]
#     theta_bins = np.floor(theta_norm * n_theta_bins).astype(int)
#     theta_bins = np.clip(theta_bins, 0, n_theta_bins - 1)

#     # Bin φ in [-π, π] into n_phi_bins
#     phi_norm = (phi + np.pi) / (2.0 * np.pi)  # in [0, 1]
#     phi_bins = np.floor(phi_norm * n_phi_bins).astype(int)
#     phi_bins = np.clip(phi_bins, 0, n_phi_bins - 1)

#     zones = theta_bins * n_phi_bins + phi_bins

#     if is_tensor:
#         return torch.from_numpy(zones).to(dtype=torch.long, device=vectors.device)
#     return zones.astype(np.int64)


def compute_direction_zones(vectors):
    """
    Discretize 3D vectors into 48 cubic-symmetry zones on the sphere.

    This follows the O_h symmetry of the diamond lattice and the NV axes:
    - The sphere is first divided into 8 octants by the coordinate planes
      (sign of Bx, By, Bz).
    - Within each octant, planes Bx = ±By, By = ±Bz, Bx = ±Bz order the
      absolute values |Bx|, |By|, |Bz|. The 6 possible orderings correspond
      to 6 permutations.

    Total zones = 8 octants × 6 permutations = 48 identical spherical
    triangles, which is the standard partition for cubic symmetry and
    matches the 48 fundamental sectors you described.

    Parameters
        vectors : array-like or tensor of shape (..., 3)
            Cartesian components (Ax, Ay, Az) or (Bx, By, Bz) in any units.
        n_theta_bins, n_phi_bins :
            Kept for backward compatibility but ignored; the number of zones
            is fixed to 48 by construction.

    Returns
        zones : ndarray or tensor of dtype int64, shape (...)
            Integer zone indices in [0, 47].
    """
    # If vectors is a tensor, convert to numpy array on CPU
    is_tensor = isinstance(vectors, torch.Tensor)
    if is_tensor:
        v = vectors.detach().cpu().numpy()
    else:
        v = np.asarray(vectors, dtype=np.float64)

    if v.shape[-1] != 3:
        raise ValueError(f"Expected last dimension of size 3 for vectors, got shape {v.shape}")

    # Extract Cartesian components
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]

    # Handle near-zero vectors: assign them to a default zone (0)
    r2 = x * x + y * y + z * z
    valid = r2 > 1e-14

    # Octant index from signs of components (3 bits → 8 octants)
    # sign_bit = 1 if component < 0 else 0
    sx = (x < 0).astype(np.int64)
    sy = (y < 0).astype(np.int64)
    sz = (z < 0).astype(np.int64)
    octant = (sx << 2) | (sy << 1) | sz  # 0..7

    # Permutation index from ordering of |Bx|, |By|, |Bz|
    abs_vals = np.stack([np.abs(x), np.abs(y), np.abs(z)], axis=-1)  # (..., 3)

    # argsort descending: indices of components from largest to smallest
    order = np.argsort(-abs_vals, axis=-1)  # (..., 3)

    # Map each of the 6 possible permutations to an ID 0..5
    perm_to_id = {
        (0, 1, 2): 0,
        (0, 2, 1): 1,
        (1, 0, 2): 2,
        (1, 2, 0): 3,
        (2, 0, 1): 4,
        (2, 1, 0): 5,
    }

    flat_order = order.reshape(-1, 3)
    perm_ids_flat = np.zeros(flat_order.shape[0], dtype=np.int64)
    for i, triplet in enumerate(flat_order):
        perm_ids_flat[i] = perm_to_id[tuple(triplet)]
    perm_id = perm_ids_flat.reshape(order.shape[:-1])  # same shape as octant

    zones = octant * 6 + perm_id  # 0..47

    # For invalid (near-zero) vectors, force zone 0
    zones = np.where(valid, zones, 0)

    if is_tensor:
        return torch.from_numpy(zones).to(dtype=torch.long, device=vectors.device)
    return zones.astype(np.int64)


def compute_zones_for_dataset(dataset_dir):
    """
    Compute per-experiment zone indices from normalized labels in metadata.csv.
    This is a convenience helper used by some training scripts. It:
    - loads metadata.csv,
    - denormalizes label columns using normalization_stats.npy,
    - maps each (Ax,Ay,Az) to one of the 48 cubic-symmetry zones.
    """
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path)

    norm_stats = load_normalization_stats(dataset_dir)
    labels_mean = norm_stats["labels_mean"]
    labels_std = norm_stats["labels_std"]
    label_cols = list(metadata.columns[:3]) # ['Ax', 'Ay', 'Az'] or ['Ar', 'theta', 'phi']
    labels_norm = metadata[label_cols].values.astype(np.float32)
    labels_denorm = denormalize_labels(labels_norm, labels_mean, labels_std)

    zones = compute_direction_zones(labels_denorm)
    return zones, labels_mean, labels_std