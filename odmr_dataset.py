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
        y : Tensor (1,) → Magnitude sqrt(Ax^2 + Ay^2 + Az^2) : Label
    """

    def __init__(self, dataset_dir, transform=None):
        # Initialize dataset paths and load metadata
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.signals_dir = os.path.join(self.dataset_dir, "signals")
        self.metadata = pd.read_csv(os.path.join(self.dataset_dir, "metadata.csv"))
        self.transform = transform

        # Create mapping: global index -> (config_id, signal_id) , to associate each signal to its current configuration (label)
        self.index_map = []
        for _, row in self.metadata.iterrows(): # iterate over configurations
            config_id = int(row["config_id"])   # get configuration ID
            signals = np.load(os.path.join(self.signals_dir, f"config_{config_id:03d}.npy")) # load signals for this configuration (n_mw_configs, n_freq)
            for mw_idx in range(signals.shape[0]): # iterate over MW configurations
                self.index_map.append((config_id, mw_idx)) # map global index to (config_id, mw_idx)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        '''
        Get spectrum and label for a given index.
        Returns:
            spectrum: Tensor of shape (N_freq,)
            label: Scalar tensor representing the magnetic field magnitude sqrt(Ax^2 + Ay^2 + Az^2)
        '''
        config_id, mw_idx = self.index_map[idx]  # get config_id and mw_idx for this idx

        signals = np.load(os.path.join(self.signals_dir, f"config_{config_id:03d}.npy"))  # load the signals file for this idx configuration
        spectrum = signals[mw_idx, :]  # get the specific spectrum for this mw_idx
        spectrum = torch.from_numpy(spectrum).unsqueeze(0)  # add channel dimension → (1, N_freq))

        row = self.metadata.iloc[config_id]  # get the metadata row for this configuration
        magnitude = np.sqrt(row["Ax"]**2 + row["Ay"]**2 + row["Az"]**2)  # calculate magnetic field magnitude
        label = torch.tensor([magnitude], dtype=torch.float32)  # get the label (magnitude)

        return spectrum, label