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


def cartesian_to_spherical(Ax, Ay, Az):
    """
    Convert cartesian current coordinates (Ax, Ay, Az) to spherical (Ar, theta, phi).
    
    Parameters:
        Ax, Ay, Az : np.ndarray
            Cartesian current components
    
    Returns:
        tuple : (Ar, theta, phi)
            Ar : radial magnitude (0 to infinity)
            theta : elevation angle from z-axis (0 to π)
            phi : azimuthal angle in xy-plane (-π to π)
    
    Notes:
        - Ar = sqrt(Ax² + Ay² + Az²)
        - theta = arccos(Az/Ar)  [0, π]
        - phi = arctan2(Ay, Ax)  [-π, π]
        - Singularity handling: when Ar~0, set theta=phi=0
    """
    Ar = np.sqrt(Ax**2 + Ay**2 + Az**2)
    
    # Handle singularity when Ar is very small
    theta = np.zeros_like(Ar)
    phi = np.zeros_like(Ar)
    
    # Only compute angles where magnitude is significant
    mask = Ar > 1e-8
    theta[mask] = np.arccos(np.clip(Az[mask] / Ar[mask], -1.0, 1.0))
    phi[mask] = np.arctan2(Ay[mask], Ax[mask])
    
    return Ar, theta, phi


def spherical_to_cartesian(Ar, theta, phi):
    """
    Convert spherical coordinates back to cartesian (for verification/denormalization).
    
    Parameters:
        Ar, theta, phi : np.ndarray
            Spherical coordinates
    
    Returns:
        tuple : (Ax, Ay, Az)
    """
    Ax = Ar * np.sin(theta) * np.cos(phi)
    Ay = Ar * np.sin(theta) * np.sin(phi)
    Az = Ar * np.cos(theta)
    
    return Ax, Ay, Az


def load_esr_multi_mw(esr_file):
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


def create_pytorch_dataset(dataset_dir, output_dir, coordinate_system='cartesian'):
    """
    Process raw ODMR data into PyTorch-compatible format.
    
    Parameters:
        dataset_dir : str or Path
            Directory containing raw data (currents CSV + ESR SPLIT files)
        output_dir : str or Path
            Output directory for processed dataset
        coordinate_system : str
            'cartesian' for (Ax, Ay, Az) or 'spherical' for (Ar, theta, phi)
    
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
    
    # Convert to spherical if requested
    if coordinate_system == 'spherical':
        print(f"\nConverting to spherical coordinates...")
        Ar, theta, phi = cartesian_to_spherical(Ax, Ay, Az)
        print(f"  Ar range: [{Ar.min():.4f}, {Ar.max():.4f}]")
        print(f"  theta range: [{theta.min():.4f}, {theta.max():.4f}] (should be [0, π])")
        print(f"  phi range: [{phi.min():.4f}, {phi.max():.4f}] (should be [-π, π])")
        # Replace Ax, Ay, Az with spherical coordinates
        Ax, Ay, Az = Ar, theta, phi
        label_names = ['Ar', 'theta', 'phi']
    else:
        label_names = ['Ax', 'Ay', 'Az']
    
    # Process first file to get frequencies
    print("\nProcessing first file to extract frequencies...")
    frequencies, _ = load_esr_multi_mw(split_files[0])
    
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
        _, signals = load_esr_multi_mw(split_path)  # Shape: (10, 201)
        all_signals.append(signals)
    
    # Stack all signals: (2109, 10, 201)
    all_signals = np.stack(all_signals, axis=0)
    print(f"Loaded signals shape: {all_signals.shape}")
    
    # Apply global normalization
    all_signals = normalize_global(all_signals)
    
    # Create metadata DataFrame with NORMALIZED labels
    # Compute normalization stats
    labels_mean = np.array([Ax[:len(split_files)].mean(), 
                           Ay[:len(split_files)].mean(), 
                           Az[:len(split_files)].mean()], dtype=np.float32)
    
    if coordinate_system == 'spherical':
        # Spherical normalization: use actual std for all components
        # This ensures proper metric calculation during training
        ar_std = Ax[:len(split_files)].std()      # Ax now contains Ar
        theta_std = Ay[:len(split_files)].std()   # Ay now contains theta
        phi_std = Az[:len(split_files)].std()     # Az now contains phi
        
        labels_std = np.array([ar_std, theta_std, phi_std], dtype=np.float32)
        print(f"\nLabels normalization stats (SPHERICAL):")
        print(f"  Mean: Ar={labels_mean[0]:.4f}, theta={labels_mean[1]:.4f}, phi={labels_mean[2]:.4f}")
        print(f"  Std: Ar={ar_std:.4f}, theta={theta_std:.4f}, phi={phi_std:.4f}")
        print(f"  Note: theta range [0, π], phi range [-π, π]")
    else:
        # Use GLOBAL std (max of all components) to preserve physical proportions
        # This prevents artificially amplifying Ax (which varies less physically)
        ax_std = Ax[:len(split_files)].std()
        ay_std = Ay[:len(split_files)].std()
        az_std = Az[:len(split_files)].std()
        global_std = max(ax_std, ay_std, az_std)
        
        labels_std = np.array([global_std, global_std, global_std], dtype=np.float32)
        
        print(f"\nLabels normalization stats (GLOBAL NORMALIZATION):")
        print(f"  Mean: Ax={labels_mean[0]:.4f}, Ay={labels_mean[1]:.4f}, Az={labels_mean[2]:.4f}")
        print(f"  Individual std: Ax={ax_std:.4f}, Ay={ay_std:.4f}, Az={az_std:.4f}")
        print(f"  Global std used: {global_std:.4f} (preserves physical proportions)")
        print(f"  Result: Normalized std will be Ax≈{ax_std/global_std:.2f}, Ay≈{ay_std/global_std:.2f}, Az≈{az_std/global_std:.2f}")
    
    # Save normalization stats for later denormalization
    normalization_stats = {
        'labels_mean': labels_mean,
        'labels_std': labels_std,
        'coordinate_system': coordinate_system
    }
    np.save(output_dir / 'normalization_stats.npy', normalization_stats)
    
    # Normalize labels
    Ax_norm = (Ax[:len(split_files)] - labels_mean[0]) / (labels_std[0] + 1e-8)
    Ay_norm = (Ay[:len(split_files)] - labels_mean[1]) / (labels_std[1] + 1e-8)
    Az_norm = (Az[:len(split_files)] - labels_mean[2]) / (labels_std[2] + 1e-8)
    
    metadata = pd.DataFrame({
        'experiment_id': range(len(split_files)),
        label_names[0]: Ax_norm.astype(np.float32),
        label_names[1]: Ay_norm.astype(np.float32),
        label_names[2]: Az_norm.astype(np.float32)
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
    print(f"  Sample (NORMALIZED labels):\n{metadata.head()}")
    
    # Check normalization stats
    stats_path = output_dir / 'normalization_stats.npy'
    if stats_path.exists():
        stats = np.load(stats_path, allow_pickle=True).item()
        print(f"\nNormalization stats:")
        print(f"  Labels mean: {stats['labels_mean']}")
        print(f"  Labels std:  {stats['labels_std']}")
    
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
    
    parser = argparse.ArgumentParser(description='Prepare ODMR dataset for training')
    parser.add_argument('--coordinate_system', type=str, default='cartesian',
                       choices=['cartesian', 'spherical'],
                       help='Coordinate system for current labels: cartesian (Ax,Ay,Az) or spherical (Ar,theta,phi)')
    parser.add_argument('--dataset_dir', type=str, default='dataset_10ElliptConf',
                       help='Input directory with raw data')
    
    args = parser.parse_args()
    
    # Set output directory based on coordinate system
    if args.coordinate_system == 'spherical':
        output_dir = 'dataset_multi_mw_spherical'
    else:
        output_dir = 'dataset_multi_mw'
    
    print(f"\n{'='*60}")
    print(f"DATASET PREPARATION")
    print(f"{'='*60}")
    print(f"Coordinate system: {args.coordinate_system.upper()}")
    print(f"Input directory: {args.dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Create dataset
    create_pytorch_dataset(args.dataset_dir, output_dir, args.coordinate_system)
    
    # Verify
    verify_dataset(output_dir)


if __name__ == "__main__":
    main()
