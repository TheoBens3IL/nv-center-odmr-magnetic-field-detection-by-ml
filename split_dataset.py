import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
from odmr_dataset import ODMRDataset


def train_val_test_split(dataset_dir, val_size=0.15, test_size=0.15, random_state=1):
    """
    Split the dataset by config_ids into train, val, and test sets.
    """
    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
    config_ids = metadata["config_id"].values

    train_ids, test_ids = train_test_split(config_ids, test_size=test_size, random_state=random_state)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size/(1-test_size), random_state=random_state)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    full_dataset = ODMRDataset(dataset_dir)

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