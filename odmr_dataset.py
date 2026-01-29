import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ODMRDataset(Dataset):
    """
    PyTorch Dataset for ODMR signals.

    Each line in a config .npy file is a single spectre (201 values), corresponding to one example.
    Label is the current triplet (Ax, Ay, Az).
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
            signals_file = os.path.join(self.signals_dir, f"config_{config_id:03d}.npy") # load the signals file for this configuration
            signals = np.load(signals_file)     # load signals array for this configuration (num_signals, num_freq)
            for signal_id in range(signals.shape[0]): # iterate over signals
                self.index_map.append((config_id, signal_id)) # map global index to (config_id, signal_id)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        '''
        Get spectrum and label for a given index.
        Returns:
            spectrum: Tensor of shape (201,)
            label: Tensor of shape (3,) corresponding to (Ax, Ay, Az)
        '''
        config_id, signal_id = self.index_map[idx]  # get config_id and signal_id for this idx
        signals_file = os.path.join(self.signals_dir, f"config_{config_id:03d}.npy")  # load the signals file for this idx configuration 
        signals = np.load(signals_file)  # load signals array for this configuration (num_signals, num_freq)
        spectrum = signals[signal_id]    # get the specific spectrum (201,)

        row = self.metadata.iloc[config_id]  # get the metadata row for this configuration
        label = np.array([row["Ax"], row["Ay"], row["Az"]], dtype=np.float32)  # get the label (Ax, Ay, Az)

        if self.transform:
            spectrum = self.transform(spectrum)

        return torch.tensor(spectrum, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
