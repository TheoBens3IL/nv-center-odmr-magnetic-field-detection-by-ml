"""Estimate ODMR experimental noise from raw SPLIT files."""

import argparse
from pathlib import Path
import numpy as np
import tqdm

def infer_n_mw_configs(esr_file, n_mw_configs=None):
    if n_mw_configs is not None:
        return n_mw_configs
    data = np.loadtxt(esr_file, delimiter="\t")
    n_rows = data.shape[0] - 1
    for candidate in (6, 10, 5, 4, 8):
        if n_rows % candidate == 0:
            return candidate
    raise ValueError(
        f"Cannot infer n_mw_configs from {Path(esr_file).name} ({n_rows} measurement rows). "
        "Pass --n_mw_configs explicitly."
    )


def load_all_contrast_repetitions(esr_file, n_mw_configs=10):
    """Return contrast spectra shaped (n_mw_configs, n_repetitions, n_freq)."""
    data = np.loadtxt(esr_file, delimiter="\t")
    measurements = data[1:, :]
    n_repetitions = measurements.shape[0] // n_mw_configs
    if measurements.shape[0] != n_mw_configs * n_repetitions:
        raise ValueError(
            f"{Path(esr_file).name}: {measurements.shape[0]} raws, "
            f"expected multiple of {n_mw_configs}"
        )
    n_freq = measurements.shape[1] // 2
    contrast = measurements[:, :n_freq] / measurements[:, n_freq:2 * n_freq] - 1.0
    return contrast.reshape(n_mw_configs, n_repetitions, n_freq)


def load_contrast_repetitions(esr_file, mw_index=0, n_mw_configs=10):
    """Return (frequencies, spectra) with shape (n_repetitions, n_freq) for one MW config."""
    data = np.loadtxt(esr_file, delimiter="\t")
    measurements = data[1:, :]
    n_repetitions = measurements.shape[0] // n_mw_configs
    n_freq = measurements.shape[1] // 2
    contrast = measurements[:, :n_freq] / measurements[:, n_freq:2 * n_freq] - 1.0
    spectra = contrast.reshape(n_mw_configs, n_repetitions, n_freq)[mw_index]
    return data[0, :n_freq], spectra


def noise_details_from_spectra(spectra):
    """spectra: (n_repetitions, n_freq)"""
    mean_spectrum = spectra.mean(axis=0)
    noise_std = float(spectra.std(axis=0, ddof=1).mean())
    amplitude = float(np.ptp(mean_spectrum))
    return {
        "noise_ratio": noise_std / (amplitude + 1e-12),
        "noise_std": noise_std,
        "signal_amplitude": amplitude,
        "n_repetitions": spectra.shape[0],
    }


def experimental_noise_details(esr_file, mw_index=0, n_mw_configs=None):
    if n_mw_configs is None:
        n_mw_configs = infer_n_mw_configs(esr_file)
    all_spectra = load_all_contrast_repetitions(esr_file, n_mw_configs)
    details = noise_details_from_spectra(all_spectra[mw_index])
    details["mw_index"] = mw_index
    details["n_mw_configs"] = n_mw_configs
    return details


def find_raw_split_files(dataset_dir):
    """List SPLIT raw files in a datasets_raw experiment folder."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    split_files = []
    currents_csv = list(dataset_dir.glob("3Dcurrents_sweep_*.csv"))
    if currents_csv:
        csv_path = max(currents_csv, key=lambda p: p.stat().st_mtime)
        parts = csv_path.stem.split("_", 2)
        if len(parts) >= 3:
            timestamp = parts[2]
            split_files = sorted(dataset_dir.glob(f"ESR_{timestamp}_SPLIT_*_Raw.txt"))
            if not split_files:
                split_files = sorted(dataset_dir.glob(f"*{timestamp}*SPLIT*Raw.txt"))

    if not split_files:
        split_files = sorted(dataset_dir.glob("*SPLIT*Raw.txt"))

    return split_files


def analyze_single_file(esr_file, mw_index=0, n_mw_configs=None):
    esr_file = Path(esr_file)
    if not esr_file.is_file():
        raise FileNotFoundError(f"File not found: {esr_file}")

    results = experimental_noise_details(esr_file, mw_index=mw_index, n_mw_configs=n_mw_configs)
    print(f"File             : {esr_file.name}")
    print(f"MW config        : {results['mw_index']}")
    print(f"Repetitions      : {results['n_repetitions']}")
    print(f"Noise ratio      : {results['noise_ratio']:.2%}")
    print(f"Noise std        : {results['noise_std']:.6f}")
    print(f"Signal amplitude : {results['signal_amplitude']:.6f}")
    return results


def analyze_raw_dataset(
    dataset_dir,
    n_mw_configs=None,
    mw_indices=None,
    save_csv=None,
):
    dataset_dir = Path(dataset_dir)
    split_files = find_raw_split_files(dataset_dir)
    if not split_files:
        raise FileNotFoundError(f"No *SPLIT*Raw.txt files in {dataset_dir}")

    if n_mw_configs is None:
        n_mw_configs = infer_n_mw_configs(split_files[0])
    if mw_indices is None:
        mw_indices = list(range(n_mw_configs))

    rows = []
    errors = []
    for exp_id, esr_file in enumerate(tqdm(split_files, desc="Noise analysis")):
        try:
            all_spectra = load_all_contrast_repetitions(esr_file, n_mw_configs)
            for mw_index in mw_indices:
                details = noise_details_from_spectra(all_spectra[mw_index])
                details["mw_index"] = mw_index
                details["n_mw_configs"] = n_mw_configs
                rows.append({
                    "experiment_id": exp_id,
                    "file": esr_file.name,
                    **details,
                })
        except Exception as exc:
            errors.append((esr_file.name, str(exc)))

    if not rows:
        raise RuntimeError(f"No successful noise estimates in {dataset_dir}")

    ratios = np.array([r["noise_ratio"] for r in rows], dtype=np.float64)
    per_mw = {}
    for mw in mw_indices:
        mw_ratios = [r["noise_ratio"] for r in rows if r["mw_index"] == mw]
        if mw_ratios:
            per_mw[mw] = np.array(mw_ratios, dtype=np.float64)

    per_experiment = []
    for exp_id in range(len(split_files)):
        exp_ratios = [r["noise_ratio"] for r in rows if r["experiment_id"] == exp_id]
        if exp_ratios:
            per_experiment.append(float(np.mean(exp_ratios)))
    per_experiment = np.array(per_experiment, dtype=np.float64)

    n_rep = rows[0]["n_repetitions"]
    print("=" * 60)
    print(f"Raw dataset noise analysis: {dataset_dir.name}")
    print("=" * 60)
    print(f"SPLIT files      : {len(split_files)}")
    print(f"MW configs       : {n_mw_configs} (analyzed: {mw_indices})")
    print(f"Repetitions/MW   : {n_rep}")
    print(f"Successful rows  : {len(rows)} ({len(errors)} file errors)")
    print()
    print("Noise ratio (mean over MW configs, per experiment):")
    print(f"  mean   : {per_experiment.mean():.2%}")
    print(f"  median : {np.median(per_experiment):.2%}")
    print(f"  std    : {per_experiment.std():.2%}")
    print(f"  min    : {per_experiment.min():.2%}")
    print(f"  max    : {per_experiment.max():.2%}")
    print(f"  p10/p90: {np.percentile(per_experiment, 10):.2%} / {np.percentile(per_experiment, 90):.2%}")
    print()
    print("Noise ratio by MW config (over all experiments):")
    for mw in mw_indices:
        if mw not in per_mw:
            continue
        arr = per_mw[mw]
        print(
            f"  MW {mw}: mean {arr.mean():.2%} | median {np.median(arr):.2%} | "
            f"std {arr.std():.2%} | min {arr.min():.2%} | max {arr.max():.2%}"
        )
    print()
    print("Global (all experiments × MW configs):")
    print(f"  mean   : {ratios.mean():.2%}")
    print(f"  median : {np.median(ratios):.2%}")
    print(f"  std    : {ratios.std():.2%}")

    if errors:
        print(f"\nWarnings: {len(errors)} failed file(s)")
        for name, msg in errors[:5]:
            print(f"  {name}: {msg}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    if save_csv:
        import csv
        save_path = Path(save_csv)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-file details saved to {save_path.resolve()}")

    return {
        "dataset_dir": str(dataset_dir),
        "n_files": len(split_files),
        "n_mw_configs": n_mw_configs,
        "n_repetitions": n_rep,
        "per_experiment": per_experiment,
        "per_mw": per_mw,
        "rows": rows,
        "errors": errors,
    }


def resolve_dataset_path(dataset_dir):
    path = Path(dataset_dir)
    if path.is_dir():
        return path
    candidate = Path("datasets_raw") / dataset_dir
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Raw dataset not found: {dataset_dir}")


def main():
    default_esr = "datasets_raw/dataset_10ElliptConf_V2/3Dcurrents_sweep_SPLIT_2026-02-07_15h18m23sRaw.txt"

    p = argparse.ArgumentParser(description="Experimental ODMR noise ratio from raw SPLIT files.")
    p.add_argument("--dataset_dir", type=str, default=None,
                   help="Raw dataset folder in datasets_raw/ (analyzes all SPLIT files)")
    p.add_argument("--esr_file", default=None, help="Single *SPLIT*Raw.txt file")
    p.add_argument("--mw_index", type=int, default=None,
                   help="MW config index (single-file mode default: 0)")
    p.add_argument("--mw_configs", type=int, nargs="+", default=None,
                   help="MW indices to analyze in dataset mode (default: all)")
    p.add_argument("--n_mw_configs", type=int, default=None,
                   help="Number of MW configs per SPLIT file (default: auto, usually 6 or 10)")
    p.add_argument("--save_csv", type=str, default=None,
                   help="Optional CSV path for per-file / per-MW noise details")
    args = p.parse_args()

    if args.dataset_dir:
        dataset_path = resolve_dataset_path(args.dataset_dir)
        analyze_raw_dataset(
            dataset_path,
            n_mw_configs=args.n_mw_configs,
            mw_indices=args.mw_configs,
            save_csv=args.save_csv,
        )
        return

    esr_file = Path(args.esr_file or default_esr)
    mw_index = 0 if args.mw_index is None else args.mw_index
    analyze_single_file(esr_file, mw_index=mw_index, n_mw_configs=args.n_mw_configs)


if __name__ == "__main__":
    main()
