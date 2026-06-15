"""
ODMR experimental noise from raw SPLIT files.
noise_ratio = mean(std inter-repetitions) / amplitude contraste
"""

import argparse
import numpy as np
from pathlib import Path


def load_contrast_repetitions(esr_file, mw_index=0, n_mw_configs=10):
    """Return (frequencies, spectra) with shape (n_repetitions, 201) for the chosen MW config."""
    data = np.loadtxt(esr_file, delimiter="\t")
    measurements = data[1:, :]
    n_repetitions = measurements.shape[0] // n_mw_configs
    if measurements.shape[0] != n_mw_configs * n_repetitions:
        raise ValueError(f"{Path(esr_file).name}: {measurements.shape[0]} raws, expected multiple of {n_mw_configs}")
    contrast = measurements[:, :201] / measurements[:, 201:402] - 1.0  # contrast = signal / background - 1.0
    spectra = contrast.reshape(n_mw_configs, n_repetitions, -1)[mw_index]
    return data[0, :201], spectra


def experimental_noise_details(esr_file, mw_index=0):
    spectra = load_contrast_repetitions(esr_file, mw_index)[1]
    mean_spectrum = spectra.mean(axis=0)
    noise_std = float(spectra.std(axis=0, ddof=1).mean())
    amplitude = float(np.ptp(mean_spectrum))
    return {
        "noise_ratio": noise_std / (amplitude + 1e-12),
        "noise_std": noise_std,
        "signal_amplitude": amplitude,
        "n_repetitions": spectra.shape[0],
        "mw_index": mw_index,
    }


def main():
    DEFAULT_ESR_FILE = "datasets_raw/dataset_10ElliptConf_V2/3Dcurrents_sweep_SPLIT_2026-02-07_15h18m23sRaw.txt"

    p = argparse.ArgumentParser(description="Experimental ODMR noise ratio from raw SPLIT files.")
    p.add_argument("esr_file", nargs="?", default=DEFAULT_ESR_FILE, help="Chemin vers un fichier *SPLIT*Raw.txt")
    p.add_argument("--mw_index", type=int, default=0, help="MW config (0-9)")
    args = p.parse_args()

    esr_file = Path(args.esr_file)
    if not esr_file.is_file():
        raise FileNotFoundError(f"File not found: {esr_file}")

    results = experimental_noise_details(esr_file, mw_index=args.mw_index)
    print(f"File             : {esr_file.name}")
    print(f"MW config        : {results['mw_index']}")
    print(f"Repetitions      : {results['n_repetitions']}")
    print(f"Noise ratio      : {results['noise_ratio']:.2%}")
    print(f"Noise std        : {results['noise_std']:.6f}")
    print(f"Signal amplitude : {results['signal_amplitude']:.6f}")

if __name__ == "__main__":
    main()