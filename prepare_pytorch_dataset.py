"""Preprocess multi-MW ODMR raw data into a PyTorch dataset."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def load_currents(currents_file):
    data = pd.read_csv(currents_file, header=None)
    Ax = data.iloc[0].values
    Ay = data.iloc[1].values
    Az = data.iloc[2].values
    return Ax, Ay, Az


def resolve_mw_layout(n_total, n_mw_configs, n_repetitions):
    """
    Resolve (n_mw_configs, n_repetitions) from the number of measurement lines.

    If n_mw_configs is None, infer it as n_total // n_repetitions.
    Otherwise verify that n_total == n_mw_configs * n_repetitions.
    """
    if n_mw_configs is None:
        if n_total % n_repetitions != 0:
            raise ValueError(
                f"{n_total} measurement lines is not divisible by "
                f"{n_repetitions} repetitions per MW config. "
                f"Pass --n_mw_configs explicitly or fix --n_repetitions."
            )
        return n_total // n_repetitions, n_repetitions

    if n_total % n_mw_configs != 0:
        raise ValueError(
            f"{n_total} measurement lines is not divisible by "
            f"{n_mw_configs} MW configs"
        )
    reps = n_total // n_mw_configs
    if reps != n_repetitions:
        print(
            f"Note: using {reps} repetitions per MW config "
            f"(from {n_total} lines / {n_mw_configs} configs), "
            f"not --n_repetitions={n_repetitions}"
        )
    return n_mw_configs, reps


def resolve_n_freq(measurements, n_freq=None):
    """
    Resolve the number of frequency points from measurement columns.
    SPLIT files store signal and background blocks of equal length:
        [signal_0..n-1 | background_0..n-1]
    """
    n_cols = measurements.shape[1]
    if n_freq is None:
        if n_cols % 2 != 0:
            raise ValueError(
                f"Expected an even number of columns (signal + background), got {n_cols}. "
                "Pass --n_freq explicitly."
            )
        return n_cols // 2

    if n_cols != 2 * n_freq:
        raise ValueError(
            f"File has {n_cols} measurement columns, expected {2 * n_freq} "
            f"for --n_freq={n_freq} (signal + background blocks)."
        )
    return n_freq


def load_esr_multi_mw(esr_file, n_mw_configs=None, n_repetitions=10, n_freq=None):
    """
    Load raw ESR data from SPLIT file with MW configurations.
    File structure:
        - Line 1: frequency axis (first n_freq values used; repeated block ignored)
        - Lines 2+: n_mw_configs × n_repetitions measurement lines
        - Columns 0..n_freq-1: signals
        - Columns n_freq..2*n_freq-1: backgrounds
    Returns:
        frequencies : (n_freq,)
        signals : (n_mw_configs, n_freq) - normalized and averaged signals
    """
    data = np.loadtxt(esr_file, delimiter='\t')
    measurements = data[1:, :]
    n_total = measurements.shape[0]

    n_mw_configs, n_repetitions = resolve_mw_layout(n_total, n_mw_configs, n_repetitions)
    n_freq = resolve_n_freq(measurements, n_freq)

    frequencies = data[0, :n_freq]
    signals = measurements[:, :n_freq]
    backgrounds = measurements[:, n_freq:2 * n_freq]

    normalized = signals / backgrounds - 1.0

    mw_configs = []
    for i in range(n_mw_configs):
        start_idx = i * n_repetitions
        end_idx = start_idx + n_repetitions
        config_avg = normalized[start_idx:end_idx, :].mean(axis=0)
        mw_configs.append(config_avg)

    signals_array = np.stack(mw_configs, axis=0)
    return frequencies, signals_array


def normalize_global(signals):
    """
    Global z-score over all signals (shared mean/std).
    - Preserves amplitude differences between configurations
    - Puts values in range suitable for neural networks (roughly -3 to +3)
    - All signals use the same normalization parameters
    """
    global_mean = signals.mean()
    global_std = signals.std()
    return (signals - global_mean) / (global_std + 1e-8)


def create_pytorch_dataset(dataset_dir, output_dir, n_mw_configs=None, n_repetitions=10, n_freq=None):
    """
    Process raw ODMR data into PyTorch-compatible format.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Create output directories
    output_dir.mkdir(exist_ok=True)
    signals_dir = output_dir / 'signals'
    signals_dir.mkdir(exist_ok=True)

    # Find currents CSV
    currents_csv = list(dataset_dir.glob('3Dcurrents_sweep_*.csv'))
    if not currents_csv:
        raise FileNotFoundError(f"No currents CSV found in {dataset_dir}")
    currents_csv = max(currents_csv, key=lambda p: p.stat().st_mtime)  # Most recent file

    # Extract timestamp from CSV filename to match SPLIT files
    # CSV: 3Dcurrents_sweep_2026-02-06_11h47m27s.csv
    # SPLIT: ESR_2026-02-06_11h47m27s_SPLIT_0000_Raw.txt
    csv_timestamp = currents_csv.stem.split('_', 2)[2]

    # Find all SPLIT files matching this timestamp
    split_pattern = f'ESR_{csv_timestamp}_SPLIT_*_Raw.txt'
    split_files = sorted(dataset_dir.glob(split_pattern))

    # Fallback: if no files found with timestamp, try generic pattern (for original datasets)
    if len(split_files) == 0:
        split_files = sorted(dataset_dir.glob('*SPLIT*Raw.txt'))
        print(f"Found {len(split_files)} SPLIT files (generic pattern)")
    else:
        print(f"Found {len(split_files)} SPLIT files matching timestamp {csv_timestamp}")
    if len(split_files) == 0:
        raise FileNotFoundError(f"No SPLIT files found in {dataset_dir}")

    Ax, Ay, Az = load_currents(currents_csv)
    label_names = ['Ax', 'Ay', 'Az']

    n_total = np.loadtxt(split_files[0], delimiter='\t').shape[0] - 1
    resolved_n_mw, resolved_rep = resolve_mw_layout(n_total, n_mw_configs, n_repetitions)
    probe_measurements = np.loadtxt(split_files[0], delimiter='\t')[1:, :]
    resolved_n_freq = resolve_n_freq(probe_measurements, n_freq)
    print(
        f"MW layout: {resolved_n_mw} configs x {resolved_rep} repetitions "
        f"({n_total} measurement lines per SPLIT file)"
    )
    print(f"Frequency points: {resolved_n_freq}" + (" (auto)" if n_freq is None else " (explicit)"))

    frequencies, _ = load_esr_multi_mw(
        split_files[0],
        n_mw_configs=resolved_n_mw,
        n_repetitions=resolved_rep,
        n_freq=resolved_n_freq,
    )
    freq_path = output_dir / 'frequencies.npy'
    np.save(freq_path, frequencies.astype(np.float32))
    all_signals = []
    for split_path in tqdm(split_files, desc="Loading signals"):
        _, signals = load_esr_multi_mw(
            split_path,
            n_mw_configs=resolved_n_mw,
            n_repetitions=resolved_rep,
            n_freq=resolved_n_freq,
        )
        all_signals.append(signals)
    all_signals = np.stack(all_signals, axis=0)

    # Apply global normalization to signals before saving (preserves relative differences between spectra)
    all_signals = normalize_global(all_signals)

    # Compute normalization stats
    labels_mean = np.array([Ax[:len(split_files)].mean(),
                            Ay[:len(split_files)].mean(),
                            Az[:len(split_files)].mean()], dtype=np.float32)
    ax_std = Ax[:len(split_files)].std()
    ay_std = Ay[:len(split_files)].std()
    az_std = Az[:len(split_files)].std()
    global_std = max(ax_std, ay_std, az_std)
    labels_std = np.array([global_std, global_std, global_std], dtype=np.float32)
    normalization_stats = {
        'labels_mean': labels_mean,
        'labels_std': labels_std,
    }
    np.save(output_dir / 'normalization_stats.npy', normalization_stats)

    # Normalize labels
    Ax_norm = (Ax[:len(split_files)] - labels_mean[0]) / (labels_std[0] + 1e-8)
    Ay_norm = (Ay[:len(split_files)] - labels_mean[1]) / (labels_std[1] + 1e-8)
    Az_norm = (Az[:len(split_files)] - labels_mean[2]) / (labels_std[2] + 1e-8)

    # Create metadata DataFrame with normalized labels
    metadata = pd.DataFrame({
        'experiment_id': range(len(split_files)),
        label_names[0]: Ax_norm.astype(np.float32),
        label_names[1]: Ay_norm.astype(np.float32),
        label_names[2]: Az_norm.astype(np.float32)
    })

    metadata_path = output_dir / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)

    for exp_id in tqdm(range(len(split_files)), desc="Saving signals"):
        signal_path = signals_dir / f'config_{exp_id:04d}.npy'
        np.save(signal_path, all_signals[exp_id].astype(np.float32))

    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - metadata.csv: {len(metadata)} experiments (NORMALIZED labels)")
    print(f"  - frequencies.npy: {len(frequencies)} frequencies")
    print(f"  - normalization_stats.npy: label normalization parameters")
    print(f"  - signals/: {len(split_files)} files, each with shape {tuple(all_signals.shape[1:])}")
    print(f"  - MW configs: {resolved_n_mw}, repetitions per config: {resolved_rep}")
    print(f"\nSignal statistics:")
    print(f"  Min: {all_signals.min():.4f}")
    print(f"  Max: {all_signals.max():.4f}")
    print(f"  Mean: {all_signals.mean():.4f}")
    print(f"  Std: {all_signals.std():.4f}")
    print(f"\nNormalized labels statistics:")
    print(f"  {label_names[0]} - Min: {metadata[label_names[0]].min():.4f}, Max: {metadata[label_names[0]].max():.4f}, Mean: {metadata[label_names[0]].mean():.4f}, Std: {metadata[label_names[0]].std():.4f}")
    print(f"  {label_names[1]} - Min: {metadata[label_names[1]].min():.4f}, Max: {metadata[label_names[1]].max():.4f}, Mean: {metadata[label_names[1]].mean():.4f}, Std: {metadata[label_names[1]].std():.4f}")
    print(f"  {label_names[2]} - Min: {metadata[label_names[2]].min():.4f}, Max: {metadata[label_names[2]].max():.4f}, Mean: {metadata[label_names[2]].mean():.4f}, Std: {metadata[label_names[2]].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Prepare ODMR dataset for training')
    parser.add_argument('--dataset_dir', type=str, default='dataset_10ElliptConf_V2', help='Input directory with raw data (default: dataset_10ElliptConf_V2 in datasets_raw/)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated from input name)')
    parser.add_argument('--n_mw_configs', type=int, default=None, help='Number of MW configurations (default: infer from lines / --n_repetitions)')
    parser.add_argument('--n_repetitions', type=int, default=10, help='Repetitions per MW config when inferring --n_mw_configs (default: 10)')
    parser.add_argument('--n_freq', type=int, default=None, help='Number of frequency points per spectrum (default: auto from file columns / 2)')
    args = parser.parse_args()

    input_dir = os.path.join("datasets_raw", args.dataset_dir)
    if args.output_dir:
        output_dir = os.path.join("datasets_pytorch", args.output_dir)
    else:
        base_name = Path(input_dir).name
        output_dir = f'datasets_pytorch/{base_name}'
        Path('datasets_pytorch').mkdir(exist_ok=True)

    mw_label = (f"{args.n_mw_configs} (explicit)" if args.n_mw_configs is not None else f"auto (lines / {args.n_repetitions} reps)")

    print(f"\n{'='*60}")
    print(f"DATASET PREPARATION")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"MW configurations: {mw_label}")
    if args.n_freq is not None:
        print(f"Frequency points: {args.n_freq} (explicit)")
    else:
        print("Frequency points: auto (columns / 2)")
    print(f"{'='*60}\n")

    create_pytorch_dataset(input_dir, output_dir, n_mw_configs=args.n_mw_configs, n_repetitions=args.n_repetitions, n_freq=args.n_freq)

if __name__ == "__main__":
    main()
