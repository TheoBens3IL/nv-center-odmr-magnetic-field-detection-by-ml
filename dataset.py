import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict
from utils import compute_zones_for_dataset


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
        # Detect label columns
        if 'Ax' in self.metadata.columns:
            self.label_cols = ['Ax', 'Ay', 'Az']
        else:
            raise ValueError("Could not find label columns (Ax/Ay/Az) in metadata.csv")

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


# New: Dataset for synthetic data (5 MW configs, 400 freq)
class ODMRDatasetSynthetic(Dataset):
    """
    PyTorch Dataset for synthetic ODMR signals (5 MW configs, 400 freq).
    Each item:
        X : Tensor (5, 400)
        y : Tensor (3,)
    """
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.signals_dir = os.path.join(self.dataset_dir, "signals")
        self.metadata = pd.read_csv(os.path.join(self.dataset_dir, "metadata.csv"))
        self.transform = transform
        if 'Ax' in self.metadata.columns:
            self.label_cols = ['Ax', 'Ay', 'Az']
        else:
            raise ValueError("Could not find label columns (Ax/Ay/Az) in metadata.csv")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        config_id = int(row["experiment_id"])
        signals = np.load(os.path.join(self.signals_dir, f"config_{config_id:04d}.npy"))  # (5, 400)
        spectrum = torch.from_numpy(signals).float()  # (5, 400)
        label = torch.tensor([row[self.label_cols[0]], row[self.label_cols[1]], row[self.label_cols[2]]], dtype=torch.float32)
        return spectrum, label


def resolve_dataset_path(dataset_dir):
    """
    Resolve a CLI dataset name or path to an existing PyTorch dataset directory.

    Accepts bare names (e.g. dataset_new_1), paths under datasets_pytorch/, or
    absolute paths.
    """
    if dataset_dir is None:
        raise ValueError("dataset_dir is required")

    candidates = []
    if os.path.isabs(dataset_dir):
        candidates.append(dataset_dir)
    elif dataset_dir.replace("\\", "/").startswith("datasets_pytorch"):
        candidates.append(dataset_dir)
    else:
        candidates.append(os.path.join("datasets_pytorch", dataset_dir))
        candidates.append(dataset_dir)

    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(os.path.join(path, "metadata.csv")):
            return path

    tried = ", ".join(os.path.abspath(c) for c in candidates)
    raise FileNotFoundError(f"Dataset not found (no metadata.csv). Tried: {tried}")


def _first_signal_path(dataset_dir):
    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
    exp_id = int(metadata.iloc[0]["experiment_id"])
    return os.path.join(dataset_dir, "signals", f"config_{exp_id:04d}.npy")


_signal_shape_cache = {}


def detect_signal_shape(dataset_dir, synthetic=False):
    """
    Infer (n_mw_configs, n_freq) from the first signal file in the dataset.

    Falls back to legacy defaults (5/10 configs, 400/201 freq) if the dataset
    path is missing or unreadable.
    """
    if dataset_dir is None:
        return (5 if synthetic else 10, 400 if synthetic else 201)

    dataset_dir = resolve_dataset_path(dataset_dir)
    dataset_dir = os.path.abspath(dataset_dir)
    if dataset_dir in _signal_shape_cache:
        return _signal_shape_cache[dataset_dir]

    signal_path = _first_signal_path(dataset_dir)
    if not os.path.exists(signal_path):
        shape = (5 if synthetic else 10, 400 if synthetic else 201)
    else:
        signals = np.load(signal_path)
        if signals.ndim != 2:
            raise ValueError(
                f"Expected 2D signal array (n_mw, n_freq), got shape {signals.shape} in {signal_path}"
            )
        shape = (int(signals.shape[0]), int(signals.shape[1]))

    _signal_shape_cache[dataset_dir] = shape
    return shape


def detect_num_mw_configs(dataset_dir=None, synthetic=False):
    """Number of MW configuration channels stored in the dataset."""
    return detect_signal_shape(dataset_dir, synthetic=synthetic)[0]


def detect_n_freq(dataset_dir=None, synthetic=False):
    """Number of frequency bins per MW channel in the dataset."""
    return detect_signal_shape(dataset_dir, synthetic=synthetic)[1]


def default_num_mw_configs(synthetic=False, dataset_dir=None):
    return detect_num_mw_configs(dataset_dir, synthetic=synthetic)


def resolve_mw_indices(synthetic=False, mw_configs=None, dataset_dir=None):
    """
    Resolve MW configuration channel indices from CLI/training parameters.

    Args:
        synthetic: Legacy fallback when dataset_dir is unavailable
        mw_configs: Explicit list of config indices (default: all configs in dataset)
        dataset_dir: Path to dataset — used to auto-detect the number of MW channels

    Returns:
        List of MW config indices
    """
    max_configs = detect_num_mw_configs(dataset_dir, synthetic=synthetic)
    if mw_configs is not None:
        indices = list(mw_configs)
    else:
        indices = list(range(max_configs))

    for idx in indices:
        if idx < 0 or idx >= max_configs:
            raise ValueError(f"MW config index {idx} out of range [0, {max_configs - 1}]")
    return indices


class MWConfigSubset(Dataset):
    """Wrapper to select a subset of MW configuration channels."""

    def __init__(self, base_dataset, mw_indices):
        self.base_dataset = base_dataset
        self.mw_indices = list(mw_indices)
        self.dataset_dir = getattr(base_dataset, "dataset_dir", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        signals, labels = self.base_dataset[idx]
        return signals[self.mw_indices, :], labels


def train_val_test_split(dataset_dir, val_size=0.15, test_size=0.15, random_state=1, synthetic=False, mw_indices=None):
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

    if synthetic:
        full_dataset = ODMRDatasetSynthetic(dataset_dir)
    else:
        full_dataset = ODMRDatasetMultiConfig(dataset_dir)

    if mw_indices is None:
        mw_indices = resolve_mw_indices(synthetic=synthetic, dataset_dir=dataset_dir)
    max_configs = detect_num_mw_configs(dataset_dir, synthetic=synthetic)
    if mw_indices != list(range(max_configs)):
        full_dataset = MWConfigSubset(full_dataset, mw_indices)

    train_idx = [i for i, row in metadata.iterrows() if row["experiment_id"] in train_set]
    val_idx   = [i for i, row in metadata.iterrows() if row["experiment_id"] in val_set]
    test_idx  = [i for i, row in metadata.iterrows() if row["experiment_id"] in test_set]

    return (
        Subset(full_dataset, train_idx),
        Subset(full_dataset, val_idx),
        Subset(full_dataset, test_idx),
    )


def _build_full_dataset(dataset_dir, synthetic=False, mw_indices=None):
    if synthetic:
        full_dataset = ODMRDatasetSynthetic(dataset_dir)
    else:
        full_dataset = ODMRDatasetMultiConfig(dataset_dir)
    if mw_indices is None:
        mw_indices = resolve_mw_indices(synthetic=synthetic, dataset_dir=dataset_dir)
    max_configs = detect_num_mw_configs(dataset_dir, synthetic=synthetic)
    if mw_indices != list(range(max_configs)):
        full_dataset = MWConfigSubset(full_dataset, mw_indices)
    return full_dataset


def _print_zone_split_counts(zones, train_idx, val_idx, test_idx):
    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if not split_idx:
            print(f"  {split_name}: 0 samples")
            continue
        split_zones = zones[split_idx]
        counts = np.bincount(split_zones, minlength=int(zones.max()) + 1)
        active = counts[counts > 0]
        print(
            f"  {split_name}: {len(split_idx)} samples | "
            f"zones {len(active)}/{len(counts)} | "
            f"per-zone min/mean/max = {active.min()}/{active.mean():.1f}/{active.max()}"
        )


def _val_target_count(n_total, n_zones, val_size, val_samples_per_zone):
    """Target number of validation samples (global ~val_size fraction)."""
    if val_samples_per_zone is not None:
        return int(val_samples_per_zone) * n_zones
    return max(1, int(round(val_size * n_total)))


def _homogeneous_val_per_zone(n_total, n_zones, val_size, val_samples_per_zone):
    """Target val count per zone before per-zone feasibility cap."""
    if val_samples_per_zone is not None:
        return int(val_samples_per_zone)
    return max(1, int(round(val_size * n_total / n_zones)))


def stratified_zone_split(
    dataset_dir,
    val_size=0.10,
    test_size=0.10,
    random_state=1,
    synthetic=False,
    mw_indices=None,
    balanced_val=True,
    val_samples_per_zone=None,
    balanced_test=False,
    test_samples_per_zone=1,
):
    """
    Split the dataset by zone indices into train, val, and test sets.

    Default (~80/10/10): validation with the same target count per zone (where
    feasible), topped up to ~val_size globally; test ~test_size per zone; rest train.

    Small zones keep fewer val samples; extra val is taken from large zones so
    the global val fraction reaches ~10%.

    Args:
        val_size: Target validation fraction (used to auto-set val_samples_per_zone).
        test_size: Test fraction within each zone (after val is removed).
        balanced_val: If True, assign the same number of val samples to every zone.
        val_samples_per_zone: Explicit val count per zone (overrides val_size auto).
        balanced_test: Legacy mode — fixed test count per zone, then split train/val.
        test_samples_per_zone: Test samples per zone when balanced_test=True.
    """
    full_dataset = _build_full_dataset(dataset_dir, synthetic=synthetic, mw_indices=mw_indices)

    zones, _, _ = compute_zones_for_dataset(dataset_dir)
    zones = np.array(zones)
    print("Unique zones in full dataset:", np.unique(zones))

    zone_indices = defaultdict(list)
    for idx, zone in enumerate(zones):
        zone_indices[int(zone)].append(idx)

    all_zones = sorted(zone_indices.keys())
    rng = np.random.RandomState(random_state)
    train_idx, val_idx, test_idx = [], [], []

    n_total = len(zones)
    min_zone_count = min(len(zone_indices[z]) for z in all_zones)

    if balanced_test:
        k = int(test_samples_per_zone)
        if k < 1:
            raise ValueError("test_samples_per_zone must be >= 1")
        if k >= min_zone_count:
            raise ValueError(
                f"test_samples_per_zone={k} is too large: the smallest zone has "
                f"only {min_zone_count} samples (need at least 1 left for train/val)."
            )
        print(f"Balanced test split: {k} sample(s) per zone x {len(all_zones)} zones "
              f"= {k * len(all_zones)} test samples (disjoint from train/val)")

        for zone in all_zones:
            indices = shuffle(zone_indices[zone], random_state=rng)
            test_idx.extend(indices[:k])
            remaining = indices[k:]
            n_rem = len(remaining)
            if n_rem == 0:
                continue
            if n_rem == 1:
                train_idx.extend(remaining)
                continue
            n_val = max(1, int(np.round(val_size * n_rem)))
            n_val = min(n_val, n_rem - 1)
            val_idx.extend(remaining[:n_val])
            train_idx.extend(remaining[n_val:])
    elif balanced_val:
        n_zones = len(all_zones)
        k_target = _homogeneous_val_per_zone(n_total, n_zones, val_size, val_samples_per_zone)
        if val_samples_per_zone is not None:
            target_val = k_target * n_zones
        else:
            target_val = _val_target_count(n_total, n_zones, val_size, val_samples_per_zone)
        zone_splits = {}

        for zone in all_zones:
            indices = shuffle(zone_indices[zone], random_state=rng)
            n = len(indices)
            n_test = max(1, int(np.round(test_size * n)))
            n_test = min(n_test, max(1, n - 2))
            k_val = min(k_target, n - n_test - 1)
            if k_val < 1:
                raise ValueError(
                    f"Zone {zone} has only {n} samples; cannot allocate test, val, and train."
                )
            zone_splits[zone] = {
                "val": list(indices[:k_val]),
                "test": list(indices[k_val:k_val + n_test]),
                "train": list(indices[k_val + n_test:]),
            }

        cur_val = sum(len(s["val"]) for s in zone_splits.values())
        if cur_val < target_val:
            shortfall = target_val - cur_val
            print(
                f"Homogeneous val: {k_target}/zone where possible ({cur_val} samples); "
                f"adding {shortfall} from large zones to reach ~{100 * val_size:.0f}% val."
            )
            while cur_val < target_val:
                zone = max(zone_splits, key=lambda z: len(zone_splits[z]["train"]))
                if len(zone_splits[zone]["train"]) <= 1:
                    print(
                        f"Warning: could only reach {cur_val} val samples "
                        f"({100 * cur_val / n_total:.1f}%), target was {target_val}."
                    )
                    break
                moved = zone_splits[zone]["train"].pop()
                zone_splits[zone]["val"].append(moved)
                cur_val += 1
        else:
            print(
                f"Homogeneous val split: {k_target} sample(s)/zone where feasible; "
                f"test ~{100 * test_size:.0f}%/zone; remainder -> train"
            )

        for split in zone_splits.values():
            val_idx.extend(split["val"])
            test_idx.extend(split["test"])
            train_idx.extend(split["train"])
    else:
        for zone, indices in zone_indices.items():
            indices = shuffle(indices, random_state=rng)
            n = len(indices)
            n_test = max(1, int(np.round(test_size * n)))
            n_val = max(1, int(np.round(val_size * n)))
            n_train = n - n_test - n_val
            if n_train < 0:
                n_train = 0
            if n_train + n_val + n_test > n:
                n_test = n - n_train - n_val
            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
            test_idx.extend(indices[n_train + n_val:])

        # Legacy: ensure every split contains at least one sample from every zone
        for split_idx in [train_idx, val_idx, test_idx]:
            split_zones = set(zones[split_idx])
            missing_zones = set(all_zones) - split_zones
            if missing_zones:
                for mz in missing_zones:
                    candidates = [i for i in zone_indices[mz] if i not in split_idx]
                    if candidates:
                        split_idx.append(candidates[0])
                    else:
                        split_idx.append(zone_indices[mz][0])

    train_idx = shuffle(train_idx, random_state=rng)
    val_idx = shuffle(val_idx, random_state=rng)
    test_idx = shuffle(test_idx, random_state=rng)

    print("Zone split summary:")
    _print_zone_split_counts(zones, train_idx, val_idx, test_idx)
    n_all = len(train_idx) + len(val_idx) + len(test_idx)
    print(
        f"Global split: train {len(train_idx)} ({100 * len(train_idx) / n_all:.1f}%) | "
        f"val {len(val_idx)} ({100 * len(val_idx) / n_all:.1f}%) | "
        f"test {len(test_idx)} ({100 * len(test_idx) / n_all:.1f}%)"
    )
    print("Unique zones in train split:", np.unique(zones[train_idx]))
    print("Unique zones in val split:", np.unique(zones[val_idx]))
    print("Unique zones in test split:", np.unique(zones[test_idx]))

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


def print_dataset_statistics(train_set, val_set, test_set, label_names, labels_mean, labels_std):
    """
    Print statistics for the dataset: label stats (normalized and denormalized), signal stats, and frequency axis.
    Args:
        full_dataset: Dataset object (should support __getitem__ returning (signal, label))
        label_names: List of label names (e.g., ['Ax', 'Ay', 'Az'])
        labels_mean: Mean used for normalization
        labels_std: Std used for normalization
        train_set, val_set, test_set: Subsets for printing sizes
    """
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Dataset name: {train_set.dataset.dataset_dir.split(os.sep)[-1]}")
    print(f"Dataset sizes:  Total: {len(train_set) + len(val_set) + len(test_set)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
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


if __name__ == "__main__":
    stratified_zone_split("datasets_pytorch/dataset_multi_mw_2", random_state=1)