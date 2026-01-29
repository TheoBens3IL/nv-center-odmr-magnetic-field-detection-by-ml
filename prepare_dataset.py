import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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


def load_esr_raw(esr_file, num_experiments=257, num_mw_freq_points=201, num_signal_measurement=500) :
    """
    Load raw ESR data : frequencies, signals, backgrounds.

    Parameters:
        esr_file : str or Path
            Path to ESR raw file (*Raw.txt)
        num_experiments : int
            Number of different current configurations
        num_mw_freq_points : int
            Number of lines per experiment (swept frequencies)
        num_signal_measurement : int
            Number of signal measurements per frequency
    Returns:
        tuple : (frequencies, signals, backgrounds)
            - frequencies: array (num_mw_freq_points,)
            - signals: array (num_experiments, num_mw_freq_points, num_signal_measurement)
            - backgrounds: array (num_experiments, num_mw_freq_points, num_signal_measurement)
    """

    data = pd.read_csv(esr_file, sep='\t', header=None)

    # Extract frequencies : extract first column of the first 201 rows
    frequencies = data.iloc[0:num_mw_freq_points, 0].values

    # Extract signals and backgrounds
    signals_raw = data.iloc[0:num_experiments * num_mw_freq_points, 1:(1 + num_signal_measurement)].values
    backgrounds_raw = data.iloc[0:num_experiments * num_mw_freq_points, (1 + num_signal_measurement):(1 + 2 * num_signal_measurement)].values

    # Reshape to tensors (num_experiments, num_mw_freq_points, num_signal_measurement)
    signals = signals_raw.reshape((num_experiments, num_mw_freq_points, num_signal_measurement))
    backgrounds = backgrounds_raw.reshape((num_experiments, num_mw_freq_points, num_signal_measurement))

    return frequencies, signals, backgrounds


def normalize_by_background(signals, backgrounds):
    '''
    Normalize signals by backgrounds.
    '''
    return signals / backgrounds

def odmr_contrast(signals):
    '''
    Compute ODMR contrast from signals for having the signal vary around 0 instead of 1,
    so having large dips compare to the rest of the signal. And gradient will be no longer dominated by the offset.
    '''
    return signals - 1.0

def normalize_per_spectrum(signals):
    '''
    Normalize each spectrum by its mean and standard deviation, for having larger dips variations on each spectrum.
    '''
    return (signals - signals.mean(axis=1, keepdims=True)) / (signals.std(axis=1, keepdims=True) + 1e-8)


def create_pytorch_dataset(frequencies, normalized_signals, Ax, Ay, Az, output_dir):
    """
    Create a compatible PyTorch dataset.

    Structure created:
    output_dir/
    ├── frequencies.npy     # (201,)
    ├── metadata.csv        # config_id, Ax, Ay, Az
    └── signals/
        ├── config_000.npy  # (500, 201)
        ├── config_001.npy
        └── ...

    Parameters:
        frequencies : array (num_mw_freq_points,)
            Microwave frequency axis
        normalized_signals : array (num_experiments, num_freq, num_signals)
            Normalized ODMR signals
        Ax, Ay, Az : arrays (num_experiments,)
            Current configurations
        output_dir : str or Path
            Root directory where dataset will be written
    """

    output_dir = os.path.abspath(output_dir)          # Ensure absolute path for output directory
    signals_dir = os.path.join(output_dir, "signals") # Define signals directory
    os.makedirs(output_dir, exist_ok=True)            # Create output directory
    os.makedirs(signals_dir, exist_ok=True)           # Create signals directory

    num_experiments = normalized_signals.shape[0]     # Number of configurations

    # ===== Save frequencies array as numpy file ===== #
    np.save(
        os.path.join(output_dir, "frequencies.npy"),
        frequencies.astype(np.float32)
    )

    # ===== Save metadata as CSV file ===== #
    metadata = pd.DataFrame({
        "config_id": np.arange(num_experiments),
        "Ax": Ax.astype(np.float32),
        "Ay": Ay.astype(np.float32),
        "Az": Az.astype(np.float32),
    })

    metadata.to_csv(
        os.path.join(output_dir, "metadata.csv"),
        index=False
    )

    # ===== Save signals per configuration ===== #
    for config_id in range(num_experiments):
        # shape: (num_freq, num_signal_measurement)
        signals_cfg = normalized_signals[config_id]

        # Transpose → (num_signal_measurement, num_freq)
        signals_cfg = signals_cfg.T.astype(np.float32)

        np.save(
            os.path.join(signals_dir, f"config_{config_id:03d}.npy"),
            signals_cfg
        )


def main() :
    
    CURRENTS_FILE = "dataset_example/3Dcurrents_sweep_2026-01-26_23h00m53s.csv"
    ESR_FILE = "dataset_example/ESR_2026-01-26_23h00m57sRaw.txt"
    OUTPUT_DIR = "pytorch_dataset_example"

    # Load currents
    Ax, Ay, Az = load_currents(CURRENTS_FILE)

    # Load ESR raw data
    frequencies, signals, backgrounds = load_esr_raw(ESR_FILE)

    # Normalize signals by backgrounds
    normalized_signals = normalize_by_background(signals, backgrounds)

    # Then subtract 1.0 so that the signal varies around 0 instead of 1
    normalized_signals = odmr_contrast(normalized_signals)

    # Further normalize per spectrum for MLP/CNN training (optional but allows having larger amplitudes variations)
    normalized_signals = normalize_per_spectrum(normalized_signals)

    # Create PyTorch dataset
    create_pytorch_dataset(
        frequencies,
        normalized_signals,
        Ax,
        Ay,
        Az,
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()