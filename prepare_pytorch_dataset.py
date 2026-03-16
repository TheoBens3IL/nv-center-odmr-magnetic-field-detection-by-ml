"""
Preprocessing script for multi-MW configuration ODMR dataset.

Output structure:
    dataset_multi_mw/
    ├── currents_config.csv  # (nb_experiments, 4): experiment_id, Ax, Ay, Az
    ├── frequencies.npy      # (201,): MW sweep frequencies (2.77-2.97 GHz)
    └── signals/
        ├── config_0000.npy  # (10, 201): 10 MW configs × 201 frequencies
        ├── config_0001.npy  
        └── ...
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def load_currents(currents_file):
    """
    Load electric currents from CSV file.

    Parameters:
        currents_file : str or Path
            Path to currents file (3Dcurrents_sweep_*.csv)
    Returns:
        tuple : (Ax, Ay, Az)
            Numpy arrays containing currents for each experiment
    """
    data = pd.read_csv(currents_file, header=None)
    
    # First 3 rows contain Ax, Ay, Az
    Ax = data.iloc[0].values
    Ay = data.iloc[1].values
    Az = data.iloc[2].values

    return Ax, Ay, Az


def load_esr_multi_mw(esr_file, n_mw_configs=10):
    """
    Load raw ESR data from SPLIT file with MW configurations.
    File structure:
        - Line 1: 402 frequencies (first 201 + repeated 201)
        - Lines 2+: N measurements (n_mw_configs × n_repetitions lines)
        - Columns 0-200: signals for 201 frequencies  
        - Columns 201-401: backgrounds for 201 frequencies
    Parameters:
        esr_file : str or Path
            Path to ESR raw file (*SPLIT*Raw.txt)
        n_mw_configs : int
            Number of MW configurations (default: 10)
    Returns:
        frequencies : (201,)
        signals : (n_mw_configs, 201) - normalized and averaged signals
    """
    # Read the entire file
    data = np.loadtxt(esr_file, delimiter='\t')
    
    # Extract frequencies from first line (first 201 values)
    freq_line = data[0, :]
    frequencies = freq_line[:201]  # Take first 201 frequencies
    
    # Extract measurements (all lines after first)
    measurements = data[1:, :]  # Shape: (n_total, 402)
    n_total = measurements.shape[0]
    
    # Determine number of repetitions per MW config
    n_repetitions = n_total // n_mw_configs
    
    # Split into signals and backgrounds
    signals = measurements[:, :201]      # Shape: (n_total, 201)
    backgrounds = measurements[:, 201:]  # Shape: (n_total, 201)
    
    # Normalize: signal / background
    normalized = signals / backgrounds   # Shape: (n_total, 201)
    
    # Apply ODMR contrast
    normalized = normalized - 1.0
    
    # Average over repetitions for each MW config
    mw_configs = []
    for i in range(n_mw_configs):
        start_idx = i * n_repetitions
        end_idx = start_idx + n_repetitions
        
        # Average repetitions for this MW config
        config_avg = normalized[start_idx:end_idx, :].mean(axis=0)  # Shape: (201,)
        mw_configs.append(config_avg)
    
    # Stack to shape (n_mw_configs, 201)
    signals_array = np.stack(mw_configs, axis=0)
    
    return frequencies, signals_array


def normalize_global(signals):
    """
    Global normalization: standardize all signals using global mean and std.
    This preserves relative differences between spectra while putting values in a reasonable range.
    Benefits:
    - Preserves amplitude differences between configurations
    - Puts values in range suitable for neural networks (roughly -3 to +3)
    - All signals use the same normalization parameters
    """
    global_mean = signals.mean()
    global_std = signals.std()
    return (signals - global_mean) / (global_std + 1e-8)


def create_pytorch_dataset(dataset_dir, output_dir):
    """
    Process raw ODMR data into PyTorch-compatible format.
    Parameters:
        dataset_dir : str or Path
            Directory containing raw data (currents CSV + ESR SPLIT files)
        output_dir : str or Path
            Output directory to save processed dataset
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
    csv_timestamp = currents_csv.stem.split('_', 2)[2]  # e.g., "2026-02-06_11h47m27s"
    
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
    
    # Load currents
    Ax, Ay, Az = load_currents(currents_csv)
    label_names = ['Ax', 'Ay', 'Az']
    frequencies, _ = load_esr_multi_mw(split_files[0])
    freq_path = output_dir / 'frequencies.npy'
    np.save(freq_path, frequencies.astype(np.float32))
    all_signals = []
    for split_path in tqdm(split_files, desc="Loading signals"):
        _, signals = load_esr_multi_mw(split_path)
        all_signals.append(signals)
    all_signals = np.stack(all_signals, axis=0)

    # Apply global normalization to signals before saving (preserves relative differences between spectra)
    all_signals = normalize_global(all_signals)
    
    # Stack all signals: (2109, 10, 201)
    all_signals = np.stack(all_signals, axis=0)

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
    
    # Save metadata
    metadata_path = output_dir / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    
    # Save individual signal files
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
    print(f"  - signals/: {len(split_files)} files, each with shape (10, 201)")
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
    parser.add_argument('--dataset_dir', type=str, default='dataset_10ElliptConf_V2',
                       help='Input directory with raw data (default: dataset_10ElliptConf_V2 in datasets_raw/)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated from input name)')
    args = parser.parse_args()
    input_dir = os.path.join("datasets_raw", args.dataset_dir)
    if args.output_dir:
        output_dir = os.path.join("datasets_pytorch", args.output_dir)
    else:
        base_name = Path(input_dir).name
        output_dir = f'datasets_pytorch/{base_name}_prepared'
        Path('datasets_pytorch').mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DATASET PREPARATION")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    create_pytorch_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main()
