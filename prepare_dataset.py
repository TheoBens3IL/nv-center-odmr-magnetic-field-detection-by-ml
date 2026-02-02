"""
Preprocessing script for multi-MW configuration ODMR dataset.
Processes 2109 experiments with 10 elliptical MW configurations.

Output structure:
    dataset_multi_mw/
    ├── currents_config.csv  # (2109, 4): experiment_id, Ax, Ay, Az
    ├── frequencies.npy      # (201,): MW sweep frequencies (2.77-2.97 GHz)
    └── signals/
        ├── config_0000.npy  # (10, 201): 10 MW configs × 201 frequencies
        ├── config_0001.npy  
        └── ... (2109 files total)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


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


def load_esr_raw_multi_mw(esr_file):
    """
    Load raw ESR data from SPLIT file with 10 MW configurations.
    
    File structure:
        - Line 1: 402 frequencies (first 201 + repeated 201)
        - Lines 2-101: 100 measurements (10 MW configs × 10 signals per config)
        - Columns 0-200: signals for 201 frequencies  
        - Columns 201-401: backgrounds for 201 frequencies
    
    Parameters:
        esr_file : str or Path
            Path to ESR raw file (*SPLIT*Raw.txt)
    
    Returns:
        frequencies : (201,) - unique frequencies
        signals : (10, 201) - normalized and averaged signals for 10 MW configs
    """
    # Read the entire file
    data = np.loadtxt(esr_file, delimiter='\t')
    
    # Extract frequencies from first line (first 201 values)
    # Line has 402 values: [freq1...freq201, freq1...freq201]
    freq_line = data[0, :]
    frequencies = freq_line[:201]  # Take first 201 frequencies
    
    # Extract measurements from lines 2-101 (100 lines)
    measurements = data[1:, :]  # Shape: (100, 402)
    
    # Split into signals and backgrounds
    # First 201 columns: signals, last 201 columns: backgrounds
    signals = measurements[:, :201]      # Shape: (100, 201) - all signals
    backgrounds = measurements[:, 201:]  # Shape: (100, 201) - all backgrounds
    
    # Normalize: signal / background
    normalized = signals / backgrounds  # Shape: (100, 201)
    
    # Apply ODMR contrast
    normalized = normalized - 1.0
    
    # Average over 10 signals per MW config
    # 100 measurements = 10 MW configs × 10 signals per config
    mw_configs = []
    for i in range(10):  # 10 MW configurations
        start_idx = i * 10
        end_idx = start_idx + 10
        
        # Average 10 signals for this MW config
        config_avg = normalized[start_idx:end_idx, :].mean(axis=0)  # Shape: (201,)
        mw_configs.append(config_avg)
    
    # Stack to shape (10, 201)
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
    print(f"Global normalization: mean={global_mean:.6f}, std={global_std:.6f}")
    return (signals - global_mean) / (global_std + 1e-8)


def create_pytorch_dataset(dataset_dir, output_dir):
    """
    Process entire multi-MW dataset and save in PyTorch-compatible format.
    
    Parameters:
        dataset_dir : str or Path
            Directory containing SPLIT files and currents CSV
        output_dir : str or Path
            Directory to save processed dataset
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    signals_dir = output_dir / 'signals'
    signals_dir.mkdir(exist_ok=True)
    
    # Find all SPLIT files
    split_files = sorted(dataset_dir.glob('*SPLIT*Raw.txt'))
    print(f"Found {len(split_files)} SPLIT files")
    
    # Find currents CSV
    currents_csv = list(dataset_dir.glob('3Dcurrents_sweep_*.csv'))
    if not currents_csv:
        raise FileNotFoundError(f"No currents CSV found in {dataset_dir}")
    currents_csv = currents_csv[0]
    print(f"Found currents CSV: {currents_csv.name}")
    
    # Load currents
    Ax, Ay, Az = load_currents(currents_csv)
    print(f"Loaded currents for {len(Ax)} experiments")
    
    # Process first file to get frequencies
    print("\nProcessing first file to extract frequencies...")
    frequencies, _ = load_esr_raw_multi_mw(split_files[0])
    
    # Save frequencies
    freq_path = output_dir / 'frequencies.npy'
    np.save(freq_path, frequencies.astype(np.float32))
    print(f"Saved frequencies to {freq_path}")
    print(f"  Frequency range: {frequencies[0]:.2e} - {frequencies[-1]:.2e} Hz")
    print(f"  Number of frequencies: {len(frequencies)}")
    
    # Collect all signals first for global normalization
    print(f"\nLoading all signals for normalization...")
    all_signals = []
    
    for split_path in tqdm(split_files, desc="Loading signals"):
        _, signals = load_esr_raw_multi_mw(split_path)  # Shape: (10, 201)
        all_signals.append(signals)
    
    # Stack all signals: (2109, 10, 201)
    all_signals = np.stack(all_signals, axis=0)
    print(f"Loaded signals shape: {all_signals.shape}")
    
    # Apply global normalization
    all_signals = normalize_global(all_signals)
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'experiment_id': range(len(split_files)),
        'Ax': Ax[:len(split_files)].astype(np.float32),
        'Ay': Ay[:len(split_files)].astype(np.float32),
        'Az': Az[:len(split_files)].astype(np.float32)
    })
    
    # Save metadata
    metadata_path = output_dir / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    print(f"\nSaved metadata to {metadata_path}")
    
    # Save individual signal files
    print(f"\nSaving signal files...")
    for exp_id in tqdm(range(len(split_files)), desc="Saving signals"):
        signal_path = signals_dir / f'config_{exp_id:04d}.npy'
        np.save(signal_path, all_signals[exp_id].astype(np.float32))
    
    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - metadata.csv: {len(metadata)} experiments")
    print(f"  - frequencies.npy: {len(frequencies)} frequencies")
    print(f"  - signals/: {len(split_files)} files, each with shape (10, 201)")
    print(f"\nSignal statistics:")
    print(f"  Min: {all_signals.min():.4f}")
    print(f"  Max: {all_signals.max():.4f}")
    print(f"  Mean: {all_signals.mean():.4f}")
    print(f"  Std: {all_signals.std():.4f}")


def verify_dataset(output_dir):
    """
    Verify the created dataset by loading and checking shapes.
    
    Parameters:
        output_dir : str or Path
            Directory containing processed dataset
    """
    output_dir = Path(output_dir)
    
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    
    # Check metadata
    metadata_path = output_dir / 'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    print(f"\nMetadata CSV:")
    print(f"  Shape: {metadata.shape}")
    print(f"  Columns: {list(metadata.columns)}")
    print(f"  Sample:\n{metadata.head()}")
    
    # Check frequencies
    freq_path = output_dir / 'frequencies.npy'
    frequencies = np.load(freq_path)
    print(f"\nFrequencies:")
    print(f"  Shape: {frequencies.shape}")
    print(f"  Range: {frequencies[0]:.6e} - {frequencies[-1]:.6e} Hz")
    print(f"  Step: {(frequencies[1] - frequencies[0]):.2e} Hz")
    
    # Check signals
    signals_dir = output_dir / 'signals'
    signal_files = sorted(signals_dir.glob('config_*.npy'))
    print(f"\nSignals:")
    print(f"  Number of files: {len(signal_files)}")
    
    # Load first signal to check shape
    first_signal = np.load(signal_files[0])
    print(f"  Signal shape: {first_signal.shape} (10 MW configs × {len(frequencies)} frequencies)")
    print(f"  Sample values (first config, first 5 freqs):")
    print(f"    {first_signal[0, :5]}")
    
    print("\n" + "="*60)


def main():
    """Main function to process the dataset."""
    
    # Set paths
    DATASET_DIR = "dataset_10ElliptConf"
    OUTPUT_DIR = "dataset_multi_mw"
    
    # Create dataset
    create_pytorch_dataset(DATASET_DIR, OUTPUT_DIR)
    
    # Verify
    verify_dataset(OUTPUT_DIR)


if __name__ == "__main__":
    main()
