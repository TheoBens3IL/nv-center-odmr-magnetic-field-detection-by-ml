import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ODMRDataset(Dataset):
    """
    PyTorch Dataset for ODMR signals.

    Each item:
        X : Tensor (1, N_freq) → 1D spectrum
        y : Tensor (3,) → (Ax, Ay, Az) : Labels
    """

    def __init__(self, dataset_dir, transform=None):
        # Initialize dataset paths and load metadata
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.signals_dir = os.path.join(self.dataset_dir, "signals")
        self.metadata = pd.read_csv(os.path.join(self.dataset_dir, "metadata.csv"))
        self.transform = transform
        
        # Auto-detect label column names (Ax/Ay/Az or Bx/By/Bz)
        if 'Ax' in self.metadata.columns:
            self.label_cols = ['Ax', 'Ay', 'Az']
        elif 'Bx' in self.metadata.columns:
            self.label_cols = ['Bx', 'By', 'Bz']
        else:
            raise ValueError("Metadata must contain either Ax/Ay/Az or Bx/By/Bz columns")

        # Create mapping: global index -> (config_id, signal_id) , to associate each signal to its current configuration (label)
        self.index_map = []
        for _, row in self.metadata.iterrows(): # iterate over configurations
            config_id_raw = row["config_id"]
            # Handle both int and string formats (e.g., "config_000000")
            if isinstance(config_id_raw, str):
                config_id = config_id_raw  # Keep as string for filename matching
            else:
                config_id = int(config_id_raw)
            
            # Build filename based on format
            if isinstance(config_id, str):
                filename = f"{config_id}.npy"
            else:
                filename = f"config_{config_id:03d}.npy"
            
            signals = np.load(os.path.join(self.signals_dir, filename)) # load signals for this configuration (n_mw_configs, n_freq)
            for mw_idx in range(signals.shape[0]): # iterate over MW configurations
                self.index_map.append((config_id, mw_idx)) # map global index to (config_id, mw_idx)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        '''
        Get spectrum and label for a given index.
        Returns:
            spectrum: Tensor of shape (N_freq,)
            label: Tensor of shape (3,) corresponding to (Ax, Ay, Az)
        '''
        config_id, mw_idx = self.index_map[idx]  # get config_id and mw_idx for this idx

        # Build filename based on config_id format
        if isinstance(config_id, str):
            filename = f"{config_id}.npy"
        else:
            filename = f"config_{config_id:03d}.npy"
            
        signals = np.load(os.path.join(self.signals_dir, filename))  # load the signals file for this idx configuration
        spectrum = signals[mw_idx, :]  # get the specific spectrum for this mw_idx
        spectrum = torch.from_numpy(spectrum).unsqueeze(0)  # add channel dimension → (1, N_freq))

        # Find metadata row matching this config_id
        if isinstance(config_id, str):
            row = self.metadata[self.metadata["config_id"] == config_id].iloc[0]
        else:
            row = self.metadata.iloc[config_id]  # get the metadata row for this configuration
            
        label = torch.tensor([row[self.label_cols[0]], row[self.label_cols[1]], row[self.label_cols[2]]], dtype=torch.float32)  # get the label

        return spectrum, label