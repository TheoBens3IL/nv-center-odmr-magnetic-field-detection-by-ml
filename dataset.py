import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split


class ODMRDatasetMultiConfig(Dataset):
    """
    PyTorch Dataset for ODMR signals that returns all 10 MW configurations together.

    Each item:
        X : Tensor (10, N_freq) → all MW configurations as channels
        y : Tensor (3,) → (Ax, Ay, Az) : Labels
    """

    def __init__(self, dataset_dir, transform=None):
        # Initialize dataset paths and load metadata
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.signals_dir = os.path.join(self.dataset_dir, "signals")
        self.metadata = pd.read_csv(os.path.join(self.dataset_dir, "metadata.csv"))
        self.transform = transform
        
        # Detect label columns (cartesian or spherical)
        if 'Ax' in self.metadata.columns:
            self.label_cols = ['Ax', 'Ay', 'Az']
        elif 'Ar' in self.metadata.columns:
            self.label_cols = ['Ar', 'theta', 'phi']
        else:
            raise ValueError("Could not find label columns (Ax/Ay/Az or Ar/theta/phi)")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        '''
        Get all MW configurations and label for a given experiment.
        Returns:
            spectrum: Tensor of shape (10, N_freq) - all MW configs as channels
            label: Tensor of shape (3,) corresponding to (Ax, Ay, Az)
        '''
        row = self.metadata.iloc[idx]
        config_id = int(row["experiment_id"])

        # Load all 10 MW configurations for this experiment
        signals = np.load(os.path.join(self.signals_dir, f"config_{config_id:04d}.npy"))  # (10, 201)
        spectrum = torch.from_numpy(signals).float()  # (10, N_freq)

        label = torch.tensor([row[self.label_cols[0]], row[self.label_cols[1]], row[self.label_cols[2]]], dtype=torch.float32)

        return spectrum, label


def train_val_test_split(dataset_dir, val_size=0.15, test_size=0.15, random_state=1):
    """
    Split the dataset by experiment_ids into train, val, and test sets.
    
    Args:
        dataset_dir: Path to dataset directory
        val_size: Validation set size (fraction)
        test_size: Test set size (fraction)
        random_state: Random seed for reproducibility
    """
    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
    config_ids = metadata["experiment_id"].values

    train_ids, test_ids = train_test_split(config_ids, test_size=test_size, random_state=random_state)
    train_ids, val_ids  = train_test_split(train_ids, test_size=val_size/(1-test_size), random_state=random_state)

    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    full_dataset = ODMRDatasetMultiConfig(dataset_dir)

    train_idx = [i for i, row in metadata.iterrows() if row["experiment_id"] in train_set]
    val_idx   = [i for i, row in metadata.iterrows() if row["experiment_id"] in val_set]
    test_idx  = [i for i, row in metadata.iterrows() if row["experiment_id"] in test_set]

    return (
        Subset(full_dataset, train_idx),
        Subset(full_dataset, val_idx),
        Subset(full_dataset, test_idx),
    )

def get_data_loaders(train_set, val_set, test_set, batch_size=32, device="cpu"):
    """
    Utility to create DataLoaders from Subsets.
    Returns:
        train_loader, val_loader, test_loader
    """
    pin_memory = True if device == "cuda" else False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def get_frequency_axis(dataset_dir):
    """
    Extract frequency axis from frequencies.npy in the dataset directory.
    Returns:
        freq_axis (np.ndarray): Frequency axis in Hz
    """
    freq_path = os.path.join(dataset_dir, "frequencies.npy")

    # Extract frequency axis (in Hz) from frequencies.npy if it exists, otherwise create it based on known range
    if os.path.exists(freq_path):
        freq_axis = np.load(freq_path)
    else:
        freq_axis = np.linspace(2.7, 3.1, 201)

    # Convert to Hz if in GHz if needed (assuming if max < 1000, it's in GHz)
    if freq_axis.max() < 1000:
        freq_axis_Hz = freq_axis * 1e9
    else:
        freq_axis_Hz = freq_axis

    return freq_axis_Hz


def print_dataset_statistics(train_set, val_set, test_set, label_names, labels_mean, labels_std, coord_system):
    """
    Print statistics for the dataset: label stats (normalized and denormalized), signal stats, and frequency axis.
    Args:
        full_dataset: Dataset object (should support __getitem__ returning (signal, label))
        label_names: List of label names (e.g., ['Ax', 'Ay', 'Az'])
        labels_mean: Mean used for normalization
        labels_std: Std used for normalization
        coord_system: 'cartesian' or 'spherical'
        train_set, val_set, test_set: Subsets for printing sizes
    """
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Dataset name: {train_set.dataset.dataset_dir.split(os.sep)[-1]}")
    print(f"Dataset sizes:  Total: {len(train_set) + len(val_set) + len(test_set)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"Coordinate system: {coord_system} ({', '.join(label_names)})")
    print()

    # Access the full dataset from the Subset to compute statistics
    full_dataset = train_set.dataset
    all_labels = torch.stack([full_dataset[i][1] for i in range(len(full_dataset))], dim=0)
    all_labels_denorm = all_labels * torch.tensor(labels_std) + torch.tensor(labels_mean)

    # Print normalized label stats
    print("Normalized labels stats:")
    for i, name in enumerate(label_names):
        print(f"  {name}: mean={all_labels[:,i].mean():.3f}, std={all_labels[:,i].std():.3f}, min={all_labels[:,i].min():.3f}, max={all_labels[:,i].max():.3f}")
    print()

    # Print denormalized label stats
    print("Labels phys stats (dataset):")
    for i, name in enumerate(label_names):
        print(f"  {name}: mean={all_labels_denorm[:,i].mean():.3f}, std={all_labels_denorm[:,i].std():.3f}, min={all_labels_denorm[:,i].min():.3f}, max={all_labels_denorm[:,i].max():.3f}")
    print()

    # Print signal stats
    all_signals = torch.cat([full_dataset[i][0] for i in range(len(full_dataset))], dim=0)
    print("Signals stats:")
    print(f"  mean={all_signals.mean():.3f}, std={all_signals.std():.3f}, min={all_signals.min():.3f}, max={all_signals.max():.3f}")
    print("=" * 60)
    print()