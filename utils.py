"""
Utility functions for ODMR dataset processing, training and evaluation.
"""

import numpy as np
import torch
from pathlib import Path
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations


class EarlyStopping:
    '''
    Early stopping to halt training when validation loss doesn't improve after a set number (patience) of epochs.
    If no improvement after 'patience' epochs, training stops.
    '''
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_state = None

    def step(self, metric, model):
        if self.mode == 'min':
            improved = metric < self.best_metric - self.min_delta
        else:
            improved = metric > self.best_metric + self.min_delta

        if improved:
            self.best_metric = metric
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
        return self.counter >= self.patience

    @property
    def best_loss(self):
        """Backward-compatible alias for code that reads early_stopping.best_loss."""
        return self.best_metric


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

    # 3. MAE mean
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


def mw_count_subdir(mw_indices):
    """Subdirectory name for a given MW config selection (e.g. [1, 3, 4, 6] -> '4mw')."""
    return f"{len(mw_indices)}mw"


def resolve_model_output_dir(dataset_dir, model_name, mw_indices):
    """
    Locate trained model artifacts for a dataset / MW selection.

    Prefers models_trained/<dataset>/<N>mw/<model>/ when present, otherwise
    falls back to models_trained/<dataset>/<model>/.
    """
    dataset_name = Path(dataset_dir).name
    per_count_dir = get_model_output_dir(dataset_name, model_name, mw_indices, per_mw_count=True)
    global_dir = get_model_output_dir(dataset_name, model_name, mw_indices, per_mw_count=False)
    if per_count_dir.exists():
        return per_count_dir
    return global_dir


def get_model_output_dir(dataset_dir, model_name, mw_indices=None, per_mw_count=False):
    """
    Return output directory for trained model artifacts.

    Global (default): models_trained/<dataset>/<model>/
    Per MW count:      models_trained/<dataset>/<N>mw/<model>/
    """
    base = Path("models_trained") / Path(dataset_dir)
    if per_mw_count and mw_indices is not None:
        base = base / mw_count_subdir(mw_indices)
    return base / model_name.lower()


def read_train_log_mae(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    try:
        with open(log_path, 'r') as f:
            prev_log = json.load(f)
        prev_mae = prev_log.get('metrics', {}).get('mae')
        if prev_mae is not None:
            return sum(prev_mae) / len(prev_mae)
    except Exception:
        pass
    return None


def try_save_best_checkpoint(model, model_dir, checkpoint_name, record_name, metric_value, higher_is_better=False):
    """Save checkpoint if metric improves compared to a record file in model_dir."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    record_path = model_dir / record_name
    best_prev = None
    if record_path.exists():
        try:
            best_prev = float(record_path.read_text().strip())
        except Exception:
            pass

    improved = best_prev is None
    if not improved:
        improved = metric_value > best_prev if higher_is_better else metric_value < best_prev

    if not improved:
        cmp_word = "higher" if higher_is_better else "lower"
        print(f"Model not saved to {model_dir}: current={metric_value:.4f}, best={best_prev:.4f} (need {cmp_word})")
        return False

    torch.save(model.state_dict(), model_dir / checkpoint_name)
    record_path.write_text(str(metric_value))
    print(f"Best model saved to {model_dir / checkpoint_name} (metric: {metric_value:.4f})")
    return True


def save_cnn_training_run_if_improved(
    model,
    model_dir,
    model_name,
    metric_value,
    history,
    label_names,
    labels_mean,
    labels_std,
    config,
    early_stopping,
    mae_keys,
    use_physics_loss,
    physics_loss_weight,
    plot_training_history_fn,
):
    """Save CNN training artifacts if validation MAE improved for this output directory."""
    from datetime import datetime

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name.lower()}_best_model.pth"
    log_path = model_dir / f"{model_name.lower()}_train_log.json"
    plot_path = model_dir / f"{model_name.lower()}_training_plot.png"

    best_mae = read_train_log_mae(log_path)
    if best_mae is not None:
        print(f"[{model_dir}] current_mae_mean = {metric_value:.4f}, best_mae = {best_mae:.4f}")
    else:
        print(f"[{model_dir}] current_mae_mean = {metric_value:.4f}, best_mae = None")

    if best_mae is not None and metric_value >= best_mae:
        print(f"Model not saved to {model_dir}: previous best MAE={best_mae:.4f} is better or equal.")
        return False

    torch.save(model.state_dict(), model_path)
    print(f"Best model saved as {model_path} (MAE: {metric_value:.4f})")

    fig = plot_training_history_fn(history, label_names=label_names, show=False)
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved as {plot_path}")

    norm_stats_path = model_dir / f"{model_name.lower()}_scaler.json"
    norm_stats = {
        'labels_mean': labels_mean.tolist() if hasattr(labels_mean, 'tolist') else list(labels_mean),
        'labels_std': labels_std.tolist() if hasattr(labels_std, 'tolist') else list(labels_std),
    }
    with open(norm_stats_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)

    current_mae = [float(history[mae_keys[i]][-1]) for i in range(len(label_names))]
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'model_structure': str(model),
        'config': config,
        'val_loss': round(float(early_stopping.best_loss), 3),
        'train_loss': round(float(history['train_loss'][-1]), 3),
        'metrics': {
            'mae': [round(float(v), 4) for v in current_mae],
            'physics_loss': round(float(history['physics_loss'][-1]), 4) if use_physics_loss else None,
        },
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved as {log_path}")
    return True


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


# def compute_direction_zones(vectors, nv_axes=nv_axes()):
#     """
#     Partition vectors by NV peak ordering.
#     Each vector is projected onto the 4 NV axes, the order of projections is used as a configuration string.
#     Each unique ordering is mapped to a unique zone index.
#     Parameters:
#         vectors: array-like (..., 3)
#         nv_axes: array-like (4, 3), NV axes (defaults to diamond <111> directions)
#     Returns:
#         zones: ndarray of shape (...,) with integer zone indices
#     """
#     v = np.asarray(vectors, dtype=np.float64)
#     if nv_axes is None:
#         nv_axes = np.array([
#             [1, -1, 1],
#             [-1, 1, 1],
#             [-1, -1, -1],
#             [1, 1, -1]
#         ], dtype=np.float64)
#         nv_axes = nv_axes / np.linalg.norm(nv_axes, axis=1, keepdims=True)
#     # Project each vector onto NV axes
#     projections = np.dot(v, nv_axes.T)  # e.g [[0.1, 0.2, 0.5, 0.8], [0.5, 0.4, 0.6, 0.7], [0.6, 0.3, 0.2, 0.1]] 
#     # For each vector, get the order of projections (descending)
#     orderings = np.argsort(-projections, axis=1)  # e.g [[3, 2, 1, 0], [3, 2, 0, 1], [0, 1, 2, 3]]
#     # Convert ordering to string for unique config
#     configs = [''.join(map(str, ordering)) for ordering in orderings] # e.g ['3210', '3201', '0123']
#     # Map each unique config to a zone index
#     unique_configs = sorted(set(configs)) # e.g ['0123', '3201', '3210']
#     config_to_zone = {cfg: idx for idx, cfg in enumerate(unique_configs)} # e.g {'0123': 0, '3201': 1, '3210': 2}
#     zones = np.array([config_to_zone[cfg] for cfg in configs], dtype=np.int64) # e.g [0, 1, 2]
#     return zones


def split_zones(vectors, nv_axes=nv_axes()):
    """
    Partition vectors by NV peak ordering :
    - Each vector is projected onto the 4 NV axes (p_i = B·n_i)
    - Ordering by descending projection magnitude p_i (24 possibilities)
    - The order of projections is used as a configuration string.
    - Find the dominant axis by magnitude: argmax_i |p_i|
    - Split each ordering by the sign of that dominant-by-|p_i| projection (>=0 vs <0) (two disconnected sphere regions) -> 48 zones.
    - Each unique ordering is mapped to a unique zone index.

    Returns:
        zones: ndarray of shape (...,) with integer zone indices in [0, 47]
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError("Last dimension of vectors must be 3.")

    # Project each vector onto NV axes
    projections = np.dot(v, nv_axes.T)                        # Shape: (..., 4) e.g [[0.1, 0.2, 0.5, 0.8], [0.5, 0.4, 0.6, 0.7], [0.6, 0.3, 0.2, 0.1]] 
    # For each vector, get the order of projections (descending)
    orderings = np.argsort(-projections, axis=-1)             # Shape: (..., 4) e.g [[3, 2, 1, 0], [3, 2, 0, 1], [0, 1, 2, 3]]
    # Convert ordering to string for unique config to match the CSV convention
    configs = [''.join(map(str, (o + 1).tolist())) for o in orderings]  # e.g ['3210', '3201', '0123']

    # Stable mapping for the 24 permutations of "1234"
    all_configs = [''.join(map(str, p)) for p in permutations([1, 2, 3, 4])]
    all_configs = sorted(all_configs)
    cfg_to_ord = {cfg: i for i, cfg in enumerate(all_configs)}  # 0..23
    ordering_idx = np.array([cfg_to_ord[cfg] for cfg in configs], dtype=np.int64).reshape(orderings.shape[:-1])

    # Opposite-sign split: sign of dominant-by-|projection| axis
    abs_orderings = np.argsort(-np.abs(projections), axis=-1)  # (..., 4)
    proj_dominant_abs = np.take_along_axis(projections, abs_orderings[..., :1], axis=-1).squeeze(-1)  # (...,)
    sign_bit = (proj_dominant_abs < 0).astype(np.int64)  # e.g [0, 1, 0, ..., 1]

    zones = ordering_idx * 2 + sign_bit  # e.g [0, 1, 2, ..., 47]
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
    zones = split_zones(labels_denorm)

    return zones, labels_mean, labels_std


def visualize_sphere_zones_surface(n_theta=100, n_phi=100, same_config_same_color=True):
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

    zones = split_zones(directions)

    # Reshape zones to grid
    zones_grid = zones.reshape(Phi.shape)
    
    if same_config_same_color:
        # zones = config*2 + sign_bit  -> collapse back to config index for coloring
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


def visualize_dataset_vectors_on_sphere(dataset_dir):
    """
    Visualize dataset B-vector directions coloured by zone, in two complementary views.

    Left  — physical space: scatter of (Ax, Ay, Az) at their actual magnitudes.
            Shows the full distribution in current/field space.
    Right — unit sphere: each vector projected onto the unit sphere with a
            wireframe for reference.  Shows angular (directional) coverage
            independently of magnitude — useful to spot missing zones.

    Both panels use the same zone colour coding.
    Also prints per-zone sample counts to the console.
    """
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    metadata = pd.read_csv(metadata_path)
    if not {'Ax', 'Ay', 'Az'}.issubset(metadata.columns):
        raise ValueError("metadata.csv must contain Ax, Ay, Az columns")

    # Denormalize to physical units
    norm_stats = load_normalization_stats(dataset_dir)
    directions = denormalize_labels(
        metadata[['Ax', 'Ay', 'Az']].values,
        norm_stats["labels_mean"], norm_stats["labels_std"],
    )

    zones = split_zones(directions)
    unique_zones, counts = np.unique(zones, return_counts=True)
    print(f"Samples: {len(directions)}  |  Unique zones: {len(unique_zones)} / 48")
    for uz, cnt in zip(unique_zones, counts):
        print(f"  Zone {uz:2d}: {cnt} samples")

    # Unit vectors for the sphere panel
    unit_dirs = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f'Dataset direction zones — {dataset_dir}', fontsize=13)

    # Left: physical space
    ax_phys = fig.add_subplot(121, projection='3d')
    sc1 = ax_phys.scatter(*directions.T, c=zones, cmap='tab20', s=20, alpha=0.8)
    ax_phys.set_title('Physical space  (actual magnitudes)')
    ax_phys.set_xlabel('Ax'); ax_phys.set_ylabel('Ay'); ax_phys.set_zlabel('Az')
    fig.colorbar(sc1, ax=ax_phys, label='Zone index', shrink=0.6)

    # Right: unit sphere
    ax_sph = fig.add_subplot(122, projection='3d')
    ax_sph.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.25, linewidth=0.5)
    sc2 = ax_sph.scatter(*unit_dirs.T, c=zones, cmap='tab20', s=20, alpha=0.8)
    ax_sph.set_title('Unit sphere  (directions only)')
    ax_sph.set_xlabel('X'); ax_sph.set_ylabel('Y'); ax_sph.set_zlabel('Z')
    fig.colorbar(sc2, ax=ax_sph, label='Zone index', shrink=0.6)

    plt.tight_layout()
    plt.show()


def visualize_zone_sample_counts(dataset_dir, n_theta=100, n_phi=100):
    """
    Show how many dataset samples fall in each of the 48 sphere zones.

    Produces a two-panel figure:
      Left  — 3D sphere surface coloured by per-zone sample count (viridis).
      Right — 2D heatmap with axes that reflect the zone structure:
                rows   = ordering index (0..23, the 24 NV-axis ranking permutations)
                columns= sign bit (0 = B toward dominant axis, 1 = away)
              This layout directly mirrors zone = ordering_idx * 2 + sign_bit.

    Also prints per-zone counts to the console.
    """
    # --- load and denormalize labels ---
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    metadata = pd.read_csv(metadata_path)
    if not {'Ax', 'Ay', 'Az'}.issubset(metadata.columns):
        raise ValueError("metadata.csv must contain Ax, Ay, Az columns")
    stats_path = os.path.join(dataset_dir, 'normalization_stats.npy')
    if os.path.isfile(stats_path):
        norm_stats = load_normalization_stats(dataset_dir)
        labels_denorm = denormalize_labels(
            metadata[['Ax', 'Ay', 'Az']].values.astype(np.float32),
            norm_stats["labels_mean"], norm_stats["labels_std"],
        )
    else:
        # Raw physical labels (e.g. build_synthetic_dataset output before normalization)
        labels_denorm = metadata[['Ax', 'Ay', 'Az']].values.astype(np.float64)

    # --- count samples per zone ---
    n_zones = 48
    dataset_zones = split_zones(labels_denorm)
    zone_counts = np.bincount(dataset_zones, minlength=n_zones)  # shape (48,)

    print(f"Samples: {len(dataset_zones)}  |  Zones with ≥1 sample: "
          f"{(zone_counts > 0).sum()} / {n_zones}")
    for z in range(n_zones):
        print(f"  Zone {z:2d}: {zone_counts[z]} samples")

    # --- 3D sphere: colour each surface patch by its zone's sample count ---
    theta = np.linspace(0, np.pi, n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi)
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)

    sphere_dirs  = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    sphere_zones = split_zones(sphere_dirs).reshape(Phi.shape)
    count_grid   = zone_counts[sphere_zones]          # broadcast lookup

    cmap_sphere = plt.get_cmap('viridis')
    norm_sphere = plt.Normalize(count_grid.min(), count_grid.max())

    # --- 2D heatmap: ordering_idx (rows 0..23) × sign_bit (cols 0..1) ---
    # zone = ordering_idx * 2 + sign_bit  →  reshape to (24, 2)
    heatmap = zone_counts.reshape(24, 2)

    # --- plot ---
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f'Sample counts per zone — {dataset_dir}', fontsize=13)

    # left: sphere
    ax3d = fig.add_subplot(121, projection='3d')
    fc = cmap_sphere(norm_sphere(count_grid))
    ax3d.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                      linewidth=0, antialiased=False, shade=False)
    ax3d.set_title('3D sphere')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    sm = plt.cm.ScalarMappable(cmap=cmap_sphere, norm=norm_sphere)
    sm.set_array(count_grid)
    fig.colorbar(sm, ax=ax3d, label='Sample count', shrink=0.6)

    # right: heatmap
    ax2d = fig.add_subplot(122)
    im = ax2d.imshow(heatmap, cmap='viridis', aspect='auto')
    ax2d.set_title('2D heatmap  (ordering × sign bit)')
    ax2d.set_xlabel('Sign bit  (0 = toward dominant axis, 1 = away)')
    ax2d.set_ylabel('Ordering index (0–23)')
    ax2d.set_xticks([0, 1])
    ax2d.set_xticklabels(['0 (toward)', '1 (away)'])
    for i in range(24):
        for j in range(2):
            val = heatmap[i, j]
            color = 'white' if val < heatmap.max() * 0.6 else 'black'
            ax2d.text(j, i, str(val), ha='center', va='center',
                      fontsize=7, color=color)
    fig.colorbar(im, ax=ax2d, label='Sample count')

    plt.tight_layout()
    plt.show()


def compute_zone_mae(y_pred, y_true, labels_mean, labels_std, zones=None, axis='mean', n_zones=48):
    """
    Compute mean absolute error per sphere zone on a test set.

    Args:
        y_pred, y_true: model outputs and targets (normalized)
        labels_mean, labels_std: denormalization stats
        zones: optional zone indices per sample; if None, zones are inferred from true B-field
        axis: 'mean' (avg over Ax,Ay,Az) or 'Ax' / 'Ay' / 'Az'
        n_zones: number of zones (48 by default)

    Returns:
        zone_mae: array (n_zones,) with NaN for zones without test samples
        zones_np: zone index per sample
    """
    y_pred_denorm = denormalize_labels(y_pred, labels_mean, labels_std)
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std)
    if hasattr(y_pred_denorm, 'numpy'):
        y_pred_denorm = y_pred_denorm.numpy()
        y_true_denorm = y_true_denorm.numpy()

    errors = np.abs(y_pred_denorm - y_true_denorm)
    n_samples = len(errors)

    if zones is None:
        zones_np = split_zones(y_true_denorm)
    else:
        zones_np = zones.cpu().numpy() if hasattr(zones, 'cpu') else np.array(zones)
        if len(zones_np.shape) > 1:
            zones_np = zones_np.squeeze()
        if len(zones_np) != n_samples:
            zones_np = zones_np[-n_samples:]

    axis_map = {'Ax': 0, 'Ay': 1, 'Az': 2}
    if axis == 'mean':
        sample_mae = errors.mean(axis=1)
    elif axis in axis_map:
        sample_mae = errors[:, axis_map[axis]]
    else:
        raise ValueError(f"axis must be 'mean', 'Ax', 'Ay' or 'Az', got {axis!r}")

    zone_mae = np.full(n_zones, np.nan)
    for z in range(n_zones):
        mask = zones_np == z
        if mask.any():
            zone_mae[z] = sample_mae[mask].mean()
    return zone_mae, zones_np


def plot_zone_mae_on_sphere(
    y_pred, y_true, labels_mean, labels_std, zones=None,
    axis='mean', n_theta=100, n_phi=100, title=None, save_path=None, show=True,
):
    """
    Visualize per-zone MAE on the unit sphere and as a structured 2D heatmap.

    Left  — 3D sphere coloured by mean MAE in each zone (viridis).
    Right — 2D heatmap (24 ordering indices × 2 sign bits), mirroring zone = ordering*2 + sign.

    Zones with no test samples appear in grey on the sphere and as empty cells in the heatmap.
    """
    zone_mae, zones_np = compute_zone_mae(
        y_pred, y_true, labels_mean, labels_std, zones=zones, axis=axis,
    )
    valid = zone_mae[~np.isnan(zone_mae)]
    if len(valid) == 0:
        raise ValueError("No zone MAE values to plot (empty test set or no zone overlap).")

    n_covered = int(np.sum(~np.isnan(zone_mae)))
    print(f"Zone MAE ({axis}): {n_covered}/{len(zone_mae)} zones covered on test set")
    for z in range(len(zone_mae)):
        if not np.isnan(zone_mae[z]):
            print(f"  Zone {z:2d}: MAE = {zone_mae[z]:.4f} A  ({np.sum(zones_np == z)} samples)")

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi)
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)

    sphere_dirs = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    sphere_zones = split_zones(sphere_dirs).reshape(Phi.shape)
    mae_grid = zone_mae[sphere_zones]

    heatmap = zone_mae.reshape(24, 2)
    axis_label = 'Mean MAE (Ax,Ay,Az)' if axis == 'mean' else f'MAE ({axis})'

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='0.85')
    vmin, vmax = valid.min(), valid.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(16, 7))
    suptitle = title or 'Model prediction MAE per zone'
    fig.suptitle(f'{suptitle} — {axis_label}', fontsize=13)

    ax3d = fig.add_subplot(121, projection='3d')
    mae_grid_masked = np.ma.masked_invalid(mae_grid)
    fc = cmap(norm(mae_grid_masked))
    ax3d.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                      linewidth=0, antialiased=False, shade=False)
    ax3d.set_title('MAE on unit sphere')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(valid)
    fig.colorbar(sm, ax=ax3d, label=f'{axis_label} (A)', shrink=0.6)

    ax2d = fig.add_subplot(122)
    heatmap_masked = np.ma.masked_invalid(heatmap)
    im = ax2d.imshow(heatmap_masked, cmap=cmap, norm=norm, aspect='auto')
    ax2d.set_title('2D heatmap  (ordering × sign bit)')
    ax2d.set_xlabel('Sign bit  (0 = toward dominant axis, 1 = away)')
    ax2d.set_ylabel('Ordering index (0–23)')
    ax2d.set_xticks([0, 1])
    ax2d.set_xticklabels(['0 (toward)', '1 (away)'])
    for i in range(24):
        for j in range(2):
            val = heatmap[i, j]
            if np.isnan(val):
                continue
            color = 'white' if val < vmax * 0.6 else 'black'
            ax2d.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=color)
    fig.colorbar(im, ax=ax2d, label=f'{axis_label} (A)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zone MAE sphere plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, zone_mae


def select_extreme_zones(zone_mae, zones_np, min_samples=1):
    """
    Pick best (lowest MAE) and worst (highest MAE) zones — same criterion as the sphere heatmap.

    Args:
        min_samples: minimum test samples in a zone to be eligible (default 1 = match sphere extrema)
    """
    counts = np.bincount(zones_np, minlength=len(zone_mae))
    eligible = [
        z for z in range(len(zone_mae))
        if not np.isnan(zone_mae[z]) and counts[z] >= min_samples
    ]
    if len(eligible) < 2:
        raise ValueError(
            f"Need at least 2 zones with >={min_samples} test samples, found {len(eligible)}."
        )
    best_zone = min(eligible, key=lambda z: zone_mae[z])
    worst_zone = max(eligible, key=lambda z: zone_mae[z])
    return best_zone, worst_zone, counts


def _zone_description(zone_id):
    ordering = zone_id // 2
    sign_bit = zone_id % 2
    sign_label = 'toward' if sign_bit == 0 else 'away'
    return f'zone {zone_id} (ordering {ordering}, {sign_label} dominant axis)'


def plot_extreme_zone_signals(
    signals,
    y_pred,
    y_true,
    zones,
    labels_mean,
    labels_std,
    freq_axis_hz,
    zone_mae=None,
    mw_indices=None,
    axis='mean',
    min_samples=1,
    title=None,
    save_path=None,
    show=True,
):
    """
    Compare ODMR spectra between the best- and worst-performing zones (by mean test MAE).

    For each zone, plots the first test sample only (one spectrum per MW config).
    """
    if hasattr(signals, 'numpy'):
        signals = signals.numpy()
    zone_mae, zones_np = compute_zone_mae(
        y_pred, y_true, labels_mean, labels_std, zones=zones, axis=axis,
    )
    best_zone, worst_zone, counts = select_extreme_zones(zone_mae, zones_np, min_samples=min_samples)

    y_pred_denorm = denormalize_labels(y_pred, labels_mean, labels_std)
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std)
    if hasattr(y_pred_denorm, 'numpy'):
        y_pred_denorm = y_pred_denorm.numpy()
        y_true_denorm = y_true_denorm.numpy()

    sample_mae = np.abs(y_pred_denorm - y_true_denorm).mean(axis=1)

    freq_axis_hz = np.asarray(freq_axis_hz)
    freq_plot = freq_axis_hz / 1e9 if freq_axis_hz.max() > 1e6 else freq_axis_hz

    n_mw = signals.shape[1]
    if mw_indices is None:
        mw_labels = [f'MW {i}' for i in range(n_mw)]
    else:
        mw_labels = [f'MW {i}' for i in mw_indices]

    zone_pairs = [
        ('Best zone', best_zone, 'tab:green'),
        ('Worst zone', worst_zone, 'tab:red'),
    ]

    print(f"\nExtreme zones ({axis} MAE — same metric as sphere heatmap, min {min_samples} sample(s)/zone):")
    first_indices = {}
    for label, zid, _ in zone_pairs:
        sample_idx = int(np.where(zones_np == zid)[0][0])
        first_indices[zid] = sample_idx
        b_true = y_true_denorm[sample_idx]
        b_pred = y_pred_denorm[sample_idx]
        err = np.abs(b_pred - b_true)
        print(f"  {label}: {_zone_description(zid)} — first test sample #{sample_idx}")
        print(f"    zone MAE={zone_mae[zid]:.4f} A | n={counts[zid]} samples in zone")
        print(f"    sample MAE={sample_mae[sample_idx]:.4f} A")
        print(f"    B true (Ax,Ay,Az)=({b_true[0]:.4f}, {b_true[1]:.4f}, {b_true[2]:.4f}) A")
        print(f"    B pred (Ax,Ay,Az)=({b_pred[0]:.4f}, {b_pred[1]:.4f}, {b_pred[2]:.4f}) A")
        print(f"    abs err (Ax,Ay,Az)=({err[0]:.4f}, {err[1]:.4f}, {err[2]:.4f}) A")

    fig, axes = plt.subplots(n_mw, 2, figsize=(14, 3.0 * n_mw), sharex=True, sharey='row')
    if n_mw == 1:
        axes = np.array([axes])

    for col, (col_title, zid, color) in enumerate(zone_pairs):
        sample_idx = first_indices[zid]
        sample_signals = signals[sample_idx]  # (n_mw, n_freq)
        zone_mae_val = zone_mae[zid]
        sample_mae_val = sample_mae[sample_idx]

        for row in range(n_mw):
            ax = axes[row, col]
            ax.plot(freq_plot, sample_signals[row], color=color, linewidth=1.8)
            ax.set_ylabel('Signal (a.u.)')
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(
                    f"{col_title}: {_zone_description(zid)}\n"
                    f"sample #{sample_idx} MAE={sample_mae_val:.4f} A | "
                    f"zone MAE={zone_mae_val:.4f} A (n={counts[zid]})",
                    fontsize=10, pad=12,
                )
            if col == 0:
                ax.text(-0.18, 0.5, mw_labels[row], transform=ax.transAxes,
                        rotation=90, va='center', ha='center', fontsize=10)

        axes[-1, col].set_xlabel('Frequency (GHz)', labelpad=8)

    suptitle = title or 'ODMR spectra: best vs worst zone'
    fig.suptitle(suptitle, fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.12, right=0.98, hspace=0.55)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Extreme zone signals plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, best_zone, worst_zone


if __name__ == "__main__":
    # visualize_sphere_zones_surface()
    visualize_zone_sample_counts(dataset_dir="datasets_pytorch/dataset_multi_mw_2")
    # visualize_dataset_vectors_on_sphere(dataset_dir="datasets_pytorch/dataset_multi_mw_2")