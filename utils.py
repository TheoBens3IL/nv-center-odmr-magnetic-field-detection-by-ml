"""
Utility functions for ODMR dataset processing, training and evaluation.
"""

import numpy as np
import torch
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class EarlyStopping:
    '''
    Early stopping to halt training when validation loss doesn't improve after a set number (patience) of epochs.
    If no improvement after 'patience' epochs, training stops.
    '''
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience       # epochs to wait for improvement
        self.min_delta = min_delta     # minimum change to qualify as improvement
        self.best_loss = float('inf')  # best validation loss observed
        self.counter = 0               # epochs since last improvement
        self.best_state = None         # best model state

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta: # improvement observed
            self.best_loss = val_loss                  # update best loss
            self.counter = 0                           # reset counter
            self.best_state = model.state_dict()       # save best model state
        else:
            self.counter += 1                          # if no improvement, increment counter
        return self.counter >= self.patience           # return True if early stopping criterion met


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


# def compute_direction_zones(vectors):
#     """
#     Discretize 3D magnetic field vectors into 48 cubic-symmetry zones on the sphere.

#     This follows the O_h symmetry of the diamond lattice and the NV axes:
#     - The sphere is first divided into 8 octants by the coordinate planes
#       (sign of Bx, By, Bz).
#     - Within each octant, planes Bx = ±By, By = ±Bz, Bx = ±Bz order the
#       absolute values |Bx|, |By|, |Bz|. The 6 possible orderings correspond
#       to 6 permutations.
#     Total zones = 8 octants * 6 permutations = 48 identical spherical triangles,
#     which is the standard partition for cubic symmetry.

#     Parameters
#         vectors : array-like or tensor of shape (..., 3)
#             Cartesian components (Ax, Ay, Az) or (Bx, By, Bz) in any units.
#     Returns
#         zones : ndarray or tensor of dtype int64, shape (...)
#             Integer zone indices in [0, 47].
#     """
#     # Convert vectors to numpy array
#     v = np.asarray(vectors, dtype=np.float64)

#     # Extract Cartesian components
#     x = v[..., 0]  # Extract the entire first line
#     y = v[..., 1]  # Extract the entire second line
#     z = v[..., 2]  # Extract the entire third line

#     # Octant index from signs of components (3 bits → 8 octants)
#     # sign_bit = 1 if component < 0 else 0
#     sx = (x < 0).astype(np.int64)
#     sy = (y < 0).astype(np.int64)
#     sz = (z < 0).astype(np.int64)
#     octant = (sx << 2) | (sy << 1) | sz  # 0..7 (binary encoding of signs)

#     # Permutation index from ordering of |Bx|, |By|, |Bz|
#     abs_vals = np.stack([np.abs(x), np.abs(y), np.abs(z)], axis=-1)  # (..., 3)

#     # argsort descending: indices of components from largest to smallest
#     order = np.argsort(-abs_vals, axis=-1)  # (..., 3)

#     # Map each of the 6 possible permutations to an ID 0..5
#     perm_to_id = {
#         (0, 1, 2): 0,
#         (0, 2, 1): 1,
#         (1, 0, 2): 2,
#         (1, 2, 0): 3,
#         (2, 0, 1): 4,
#         (2, 1, 0): 5,
#     }

#     flat_order = order.reshape(-1, 3)
#     perm_ids_flat = np.zeros(flat_order.shape[0], dtype=np.int64)
#     for i, triplet in enumerate(flat_order):
#         perm_ids_flat[i] = perm_to_id[tuple(triplet)]
#     perm_id = perm_ids_flat.reshape(order.shape[:-1])  # same shape as octant

#     zones = octant * 6 + perm_id  # 0..47
#     return zones.astype(np.int64)


def nv_axes():
    """
    Return the 4 NV axes as unit vectors in Cartesian coordinates.
    These correspond to the <111> directions in the diamond lattice.
    """
    axes = np.array([
        [ 0.5,  0.5,  0.70710678],
        [-0.5, -0.5,  0.70710678],
        [ 0.5, -0.5, -0.70710678],
        [-0.5,  0.5, -0.70710678]
    ], dtype=np.float64)
    return axes / np.linalg.norm(axes, axis=1, keepdims=True)


def compute_direction_zones(vectors, nv_axes=nv_axes()):
    """
    Partition vectors by NV peak ordering (as in ESRpeakOrderPlotterv2.py).
    Each vector is projected onto the 4 NV axes, the order of projections is used as a configuration string.
    Each unique ordering is mapped to a unique zone index.
    Parameters:
        vectors: array-like (..., 3)
        nv_axes: array-like (4, 3), NV axes (defaults to diamond <111> directions)
    Returns:
        zones: ndarray of shape (...,) with integer zone indices
    """
    v = np.asarray(vectors, dtype=np.float64)
    if nv_axes is None:
        nv_axes = np.array([
            [1, -1, 1],
            [-1, 1, 1],
            [-1, -1, -1],
            [1, 1, -1]
        ], dtype=np.float64)
        nv_axes = nv_axes / np.linalg.norm(nv_axes, axis=1, keepdims=True)
    # Project each vector onto NV axes
    projections = np.dot(v, nv_axes.T)  # e.g [[0.1, 0.2, 0.5, 0.8], [0.5, 0.4, 0.6, 0.7], [0.6, 0.3, 0.2, 0.1]] 
    # For each vector, get the order of projections (descending)
    orderings = np.argsort(-projections, axis=1)  # e.g [[3, 2, 1, 0], [3, 2, 0, 1], [0, 1, 2, 3]]
    # Convert ordering to string for unique config
    configs = [''.join(map(str, ordering)) for ordering in orderings] # e.g ['3210', '3201', '0123']
    # Map each unique config to a zone index
    unique_configs = sorted(set(configs)) # e.g ['0123', '3201', '3210']
    config_to_zone = {cfg: idx for idx, cfg in enumerate(unique_configs)} # e.g {'0123': 0, '3201': 1, '3210': 2}
    zones = np.array([config_to_zone[cfg] for cfg in configs], dtype=np.int64) # e.g [0, 1, 2]
    return zones


def compute_direction_zones_split_opposite_sign(vectors, nv_axes=nv_axes()):
    """
    Same *ordering* logic as `compute_direction_zones()` (ESRpeakOrderPlotterv2-style: 24 permutations),
    but split each ordering into two disconnected sphere regions ("opposite sign") to obtain 48 zones.

    How the split works:
    - Compute projections p_i = B·n_i onto the 4 NV axes
    - Ordering is defined by sorting p_i descending (24 possibilities)
    - Find the dominant axis by magnitude: argmax_i |p_i|
    - Split by the sign of that dominant-by-|p| projection (>=0 vs <0)

    Returns:
        zones: ndarray of shape (...,) with integer zone indices in [0, 47]
                Zone color can be grouped back into 24 colors via (zones // 2).
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError("Last dimension of vectors must be 3.")

    # Projections onto NV axes: (..., 4)
    projections = np.dot(v, nv_axes.T)

    # ESR-style ordering: sort projections descending, get permutation of indices 0..3
    orderings = np.argsort(-projections, axis=-1)  # (..., 4)
    # Convert to 1..4 labels to match the CSV convention (e.g. "1234")
    flat_order = orderings.reshape(-1, 4)
    configs = [''.join(map(str, (o + 1).tolist())) for o in flat_order]

    # Stable mapping for the 24 permutations of "1234"
    from itertools import permutations
    all_cfgs = [''.join(map(str, p)) for p in permutations([1, 2, 3, 4])]
    all_cfgs = sorted(all_cfgs)
    cfg_to_ord = {cfg: i for i, cfg in enumerate(all_cfgs)}  # 0..23
    ordering_idx = np.array([cfg_to_ord[cfg] for cfg in configs], dtype=np.int64).reshape(orderings.shape[:-1])

    # Opposite-sign split: sign of dominant-by-|projection| axis
    abs_orderings = np.argsort(-np.abs(projections), axis=-1)  # (..., 4)
    proj_dominant_abs = np.take_along_axis(projections, abs_orderings[..., :1], axis=-1).squeeze(-1)  # (...,)
    sign_bit = (proj_dominant_abs < 0).astype(np.int64)  # 0/1

    zones = ordering_idx * 2 + sign_bit  # 0..47
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

    if {"Ax", "Ay", "Az"}.issubset(metadata.columns):
        label_cols = ["Ax", "Ay", "Az"]
    else:
        raise ValueError("metadata.csv must contain Ax/Ay/Az columns.")
    labels_norm = metadata[label_cols].values.astype(np.float32)
    labels_denorm = denormalize_labels(labels_norm, labels_mean, labels_std)
    zones = compute_direction_zones_split_opposite_sign(labels_denorm)

    return zones, labels_mean, labels_std


def visualize_dataset_direction_zones(dataset_dir):
    """
    Visualize how compute_direction_zones partitions the real current directions in a dataset.
    Loads Ax, Ay, Az from metadata.csv in dataset_dir, computes zones, and plots them in 3D.
    """
    # Load metadata
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    
    metadata = pd.read_csv(metadata_path)
    if not {'Ax', 'Ay', 'Az'}.issubset(metadata.columns):
        raise ValueError("metadata.csv must contain Ax, Ay, Az columns")
    
    # Load normalized labels from metadata and denormalize them
    labels = metadata[['Ax', 'Ay', 'Az']].values
    norm_stats = load_normalization_stats(dataset_dir)
    labels_mean = norm_stats["labels_mean"]
    labels_std = norm_stats["labels_std"]
    directions = denormalize_labels(labels, labels_mean, labels_std)

    # Compute zones for these directions
    zones = compute_direction_zones_split_opposite_sign(directions)
    print(f"Number of unique zones: {len(np.unique(zones))}")

    # Print number of points 
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    print(f"Number of points (current vectors): {len(directions)}")

    # Plot in 3D colored by zone
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=zones, cmap='tab20', s=20, alpha=0.8)
    ax.set_title(f'Dataset directions partitioned into zones ({dataset_dir})')
    ax.set_xlabel('Ax')
    ax.set_ylabel('Ay')
    ax.set_zlabel('Az')
    fig.colorbar(scatter, ax=ax, label='Zone index')
    plt.show()


def visualize_zone_vectors_for_dataset(dataset_dir):
    """
    Visualize current vectors grouped by zone for a dataset.
    Loads Ax, Ay, Az from metadata.csv, computes zones, and plots them in 3D with different colors per zone.
    """
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    metadata = pd.read_csv(metadata_path)
    if not {'Ax', 'Ay', 'Az'}.issubset(metadata.columns):
        raise ValueError("metadata.csv must contain Ax, Ay, Az columns")
    directions = metadata[['Ax', 'Ay', 'Az']].values

    zones, _, _ = compute_zones_for_dataset(dataset_dir)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=zones, cmap='tab20', s=30, alpha=0.8)
    ax.set_title(f'Current vectors grouped by zone ({dataset_dir})')
    ax.set_xlabel('Ax')
    ax.set_ylabel('Ay')
    ax.set_zlabel('Az')
    fig.colorbar(scatter, ax=ax, label='Zone index')
    # Show zone distribution
    unique_zones, counts = np.unique(zones, return_counts=True)
    print("Zone distribution:")
    for uz, cnt in zip(unique_zones, counts):
        print(f"Zone {uz}: {cnt} samples")
    plt.show()


def visualize_vectors_on_sphere(dataset_dir):
    """
    Visualize current vectors on a unit sphere to identify distribution and missing zones.
    """
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")

    metadata = pd.read_csv(metadata_path)
    if not {'Ax', 'Ay', 'Az'}.issubset(metadata.columns):
        raise ValueError("metadata.csv must contain Ax, Ay, Az columns")

    directions = metadata[['Ax', 'Ay', 'Az']].values
    # Normalize to unit vectors
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_vectors = directions / norms

    # Load zones to color points by zone
    zones, _, _ = compute_zones_for_dataset(dataset_dir)

    x, y, z = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw unit sphere for reference
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.3)

    scatter = ax.scatter(x, y, z, c=zones, cmap='tab20', s=30, alpha=0.8)
    ax.set_title(f'Unit vectors on sphere ({dataset_dir})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(scatter, ax=ax, label='Zone index')

    # Show zone distribution
    unique_zones, counts = np.unique(zones, return_counts=True)
    print("Zone distribution:")
    for uz, cnt in zip(unique_zones, counts):
        print(f"Zone {uz}: {cnt} samples")

    plt.show()


def visualize_sphere_zones_surface(n_theta=100, n_phi=100, split_opposite_sign=True, same_config_same_color=True):
    """
    Visualize a fully filled, smooth sphere surface colored by zone using plot_surface and a meshgrid.
    """
    # Create a meshgrid on the sphere
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi)
    # Convert to cartesian coordinates
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)
    # Flatten for zone computation
    directions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    if split_opposite_sign:
        zones = compute_direction_zones_split_opposite_sign(directions)  # 0..47
    else:
        zones = compute_direction_zones(directions)  # 0..23
    # Reshape zones to grid
    zones_grid = zones.reshape(Phi.shape)
    
    if split_opposite_sign and same_config_same_color:
        # zones48 = config*2 + sign_bit  -> collapse back to config index for coloring
        zones_for_color = zones_grid // 2  # 0..23
        colorbar_label = "Configuration index (24 colors)"
    else:
        zones_for_color = zones_grid
        colorbar_label = "Zone index"

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Use a colormap with more colors
    n_zones = len(np.unique(zones_for_color))
    cmap = plt.get_cmap('nipy_spectral', n_zones)
    norm = plt.Normalize(zones_for_color.min(), zones_for_color.max())
    facecolors = cmap(norm(zones_for_color))
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_title('Filled sphere surface partitioned into zones')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(zones_for_color)
    fig.colorbar(mappable, ax=ax, label=colorbar_label)
    plt.show()


def visualize_zone_sample_counts_on_sphere(dataset_dir, n_theta=100, n_phi=100, split_opposite_sign=True):
    """
    Visualize the number of samples per zone as a 3D sphere heatmap.
    Each zone on the sphere surface is colored by the number of samples in that zone.
    """
    # Load metadata
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    metadata = pd.read_csv(metadata_path)
    if not {'Ax', 'Ay', 'Az'}.issubset(metadata.columns):
        raise ValueError("metadata.csv must contain Ax, Ay, Az columns")

    # Denormalize labels
    norm_stats = load_normalization_stats(dataset_dir)
    labels_mean = norm_stats["labels_mean"]
    labels_std = norm_stats["labels_std"]
    labels_norm = metadata[['Ax', 'Ay', 'Az']].values.astype(np.float32)
    labels_denorm = denormalize_labels(labels_norm, labels_mean, labels_std)

    # Compute zones for dataset
    dataset_zones = compute_direction_zones_split_opposite_sign(labels_denorm)

    # Count samples per zone
    n_zones = 48 if split_opposite_sign else 24
    zone_counts = np.zeros(n_zones, dtype=int)
    for z in dataset_zones:
        zone_counts[z] += 1

    # Create a meshgrid on the sphere
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi)

    # Convert to cartesian coordinates
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)

    # Flatten for zone computation
    directions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    if split_opposite_sign:
        zones_grid = compute_direction_zones_split_opposite_sign(directions)
    else:
        zones_grid = compute_direction_zones(directions)

    # Reshape zones to grid
    zones_grid = zones_grid.reshape(Phi.shape)

    # Create sample count grid
    sample_count_grid = np.zeros_like(zones_grid, dtype=float)
    for i in range(zones_grid.shape[0]):
        for j in range(zones_grid.shape[1]):
            zone_idx = zones_grid[i, j]
            sample_count_grid[i, j] = zone_counts[int(zone_idx)]

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create colormap
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(sample_count_grid.min(), sample_count_grid.max())
    facecolors = cmap(norm(sample_count_grid))

    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_title('Sample Counts per Zone (on Sphere)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(sample_count_grid)
    fig.colorbar(mappable, ax=ax, label='Sample Count')

    plt.show()

    print("Zone sample counts:")
    for z in range(n_zones):
        print(f"Zone {z}: {zone_counts[z]} samples")


def visualize_zone_sample_counts_heatmap(dataset_dir):
    """
    Visualize the number of samples per zone in a dataset as a heatmap.
    Uses compute_direction_zones_split_opposite_sign for zone assignment.
    """
    # Load metadata
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    metadata = pd.read_csv(metadata_path)
    if not {'Ax', 'Ay', 'Az'}.issubset(metadata.columns):
        raise ValueError("metadata.csv must contain Ax, Ay, Az columns")
    # Denormalize labels
    norm_stats = load_normalization_stats(dataset_dir)
    labels_mean = norm_stats["labels_mean"]
    labels_std = norm_stats["labels_std"]
    labels_norm = metadata[['Ax', 'Ay', 'Az']].values.astype(np.float32)
    labels_denorm = denormalize_labels(labels_norm, labels_mean, labels_std)
    # Compute zones
    zones = compute_direction_zones_split_opposite_sign(labels_denorm)
    # Count samples per zone
    n_zones = 48
    zone_counts = np.zeros(n_zones, dtype=int)
    for z in zones:
        zone_counts[z] += 1
    # Reshape for heatmap (6 rows x 8 columns for easier visualization)
    heatmap = zone_counts.reshape(6, 8)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, cmap='viridis', aspect='auto')
    ax.set_title('Sample Counts per Zone (6x8 grid)')
    ax.set_xlabel('Octant (0-7)')
    ax.set_ylabel('Permutation (0-5)')
    # Annotate counts
    for i in range(6):
        for j in range(8):
            ax.text(j, i, heatmap[i, j], ha='center', va='center', color='white' if heatmap[i, j] > heatmap.max()/2 else 'black')
    fig.colorbar(im, ax=ax, label='Sample Count')
    plt.tight_layout()
    plt.show()
    print("Zone sample counts:")
    for z in range(n_zones):
        print(f"Zone {z}: {zone_counts[z]} samples")


if __name__ =="__main__":
    visualize_zone_sample_counts_on_sphere(dataset_dir="datasets_pytorch/dataset_multi_mw_2")
    # visualize_zone_sample_counts_heatmap(dataset_dir="datasets_pytorch/dataset_multi_mw_2")
    # visualize_dataset_direction_zones(dataset_dir="datasets_pytorch/dataset_multi_mw_2")
    # visualize_zone_vectors_for_dataset(dataset_dir="datasets_pytorch/dataset_multi_mw_2")
    # visualize_vectors_on_sphere(dataset_dir="datasets_pytorch/dataset_multi_mw_2")
    # visualize_sphere_zones_surface()