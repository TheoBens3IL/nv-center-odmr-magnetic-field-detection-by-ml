import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
from odmr_dataset import ODMRDataset
from odmr_dataset_multiconfig import ODMRDatasetMultiConfig


def train_val_test_split(dataset_dir, val_size=0.15, test_size=0.15, random_state=1, multi_config=False):
    """
    Split the dataset by experiment_ids into train, val, and test sets.
    
    Args:
        dataset_dir: Path to dataset directory
        val_size: Validation set size (fraction)
        test_size: Test set size (fraction)
        random_state: Random seed for reproducibility
        multi_config: If True, use ODMRDatasetMultiConfig (returns all 10 MW configs together)
                      If False, use ODMRDataset (returns individual spectra)
    """
    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
    config_ids = metadata["experiment_id"].values

    train_ids, test_ids = train_test_split(config_ids, test_size=test_size, random_state=random_state)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size/(1-test_size), random_state=random_state)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    # Use appropriate dataset class
    if multi_config:
        full_dataset = ODMRDatasetMultiConfig(dataset_dir)
        # For multi-config, we split by experiments directly (no index_map needed)
        train_idx = [i for i, row in metadata.iterrows() if row["experiment_id"] in train_set]
        val_idx = [i for i, row in metadata.iterrows() if row["experiment_id"] in val_set]
        test_idx = [i for i, row in metadata.iterrows() if row["experiment_id"] in test_set]
    else:
        full_dataset = ODMRDataset(dataset_dir)
        # For single-config, we split by individual spectra
        train_idx, val_idx, test_idx = [], [], []
        for i, (cfg_id, _) in enumerate(full_dataset.index_map):
            if cfg_id in train_set:
                train_idx.append(i)
            elif cfg_id in val_set:
                val_idx.append(i)
            elif cfg_id in test_set:
                test_idx.append(i)

    return (
        Subset(full_dataset, train_idx),
        Subset(full_dataset, val_idx),
        Subset(full_dataset, test_idx),
    )