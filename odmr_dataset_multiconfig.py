import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
