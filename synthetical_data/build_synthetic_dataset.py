"""Build synthetic ODMR spectra from real current labels."""


import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from simulate_esr_spectrum import ensemble_spectrum, Wstart, Wend, Wnr


MW_CONFIGS = [
    {"MW_field": [0.5, 0.0, 0.0], "MW_phase": np.deg2rad(90),  "name": "mw1"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(30),  "name": "mw2"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(150), "name": "mw3"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(210), "name": "mw4"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(330), "name": "mw5"},
]


def load_raw_labels(real_meta_dir: str) -> pd.DataFrame:
    """
    Denormalize and return (Ax, Ay, Az) from a real PyTorch dataset directory.
    Requires both metadata.csv and normalization_stats.npy to be present.
    Returns a DataFrame with columns ['Ax', 'Ay', 'Az'] in physical units.
    """
    real_meta_dir = Path(real_meta_dir)
    meta  = pd.read_csv(real_meta_dir / "metadata.csv")
    stats = np.load(real_meta_dir / "normalization_stats.npy", allow_pickle=True).item()
    mean  = stats["labels_mean"]  # (3,)
    std   = stats["labels_std"]   # (3,)
    return pd.DataFrame({
        "Ax": meta["Ax"].values * std[0] + mean[0],
        "Ay": meta["Ay"].values * std[1] + mean[1],
        "Az": meta["Az"].values * std[2] + mean[2],
    })


def build_dataset(
    output_dir: str,
    raw_labels: pd.DataFrame,
    mw_configs: list = MW_CONFIGS,
    noise_std: float = 0.0,
    seed: int = 42,
) -> None:
    """
    Simulate all spectra and write the raw (un-normalized) dataset to disk.

    Parameters
    ----------
    output_dir  : destination directory (will be created)
    raw_labels  : DataFrame with columns Ax, Ay, Az in physical units
    mw_configs  : list of MW-configuration dicts
    noise_std   : Gaussian noise std added to each spectrum (0 = no noise)
    seed        : RNG seed (only used when noise_std > 0)
    """
    output_dir = Path(output_dir)
    signals_dir = output_dir / "signals"
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(exist_ok=True)

    freq_list = np.linspace(Wstart, Wend, Wnr)
    n_samples = len(raw_labels)
    n_mw      = len(mw_configs)
    rng       = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print("SYNTHETIC DATASET GENERATION  (raw, un-normalized)")
    print(f"{'='*60}")
    print(f"Output     : {output_dir}")
    print(f"Samples    : {n_samples}")
    print(f"MW configs : {n_mw}  {[c['name'] for c in mw_configs]}")
    print(f"Freq pts   : {Wnr}  ({Wstart:.0f}–{Wend:.0f} MHz)")
    print(f"Noise std  : {noise_std}")
    print(f"{'='*60}\n")

    all_signals = np.empty((n_samples, n_mw, Wnr), dtype=np.float32)

    for mw_idx, mw in enumerate(mw_configs):
        print(f"[{mw_idx+1}/{n_mw}] '{mw['name']}'  "
              f"field={mw['MW_field']}  phase={np.rad2deg(mw['MW_phase']):.0f}°")
        for idx, row in tqdm(raw_labels.iterrows(), total=n_samples, desc=f"  {mw['name']}"):
            spectrum = ensemble_spectrum(
                [row["Ax"], row["Ay"], row["Az"]],
                mw["MW_field"], mw["MW_phase"], freq_list,
            )
            if noise_std > 0:
                spectrum = spectrum + rng.normal(0.0, noise_std, size=spectrum.shape)
            all_signals[idx, mw_idx, :] = spectrum.astype(np.float32)

    np.save(output_dir / "frequencies.npy", freq_list.astype(np.float32))

    metadata = pd.DataFrame({
        "experiment_id": range(n_samples),
        "Ax": raw_labels["Ax"].values.astype(np.float32),
        "Ay": raw_labels["Ay"].values.astype(np.float32),
        "Az": raw_labels["Az"].values.astype(np.float32),
    })
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    print("\nSaving signal files …")
    for exp_id in tqdm(range(n_samples)):
        np.save(signals_dir / f"config_{exp_id:04d}.npy", all_signals[exp_id])

    print(f"\n{'='*60}")
    print("Raw dataset saved.")
    print(f"  signals/config_XXXX.npy : shape ({n_mw}, {Wnr})")
    print(f"  metadata.csv            : raw labels (physical units)")
    print(f"  Raw signal range        : [{all_signals.min():.4f}, {all_signals.max():.4f}]")
    print(f"\nRun normalize_synthetic_dataset.py next to apply normalization.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate synthetic ODMR dataset from real current labels (raw output)."
    )
    parser.add_argument(
        "--real_meta_dir",
        type=str,
        required=True,
        help="Path to a real PyTorch dataset directory (must contain metadata.csv and normalization_stats.npy).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../datasets_synthetic/synthetic_multi_mw_raw",
        help="Output directory for the raw dataset (default: ../datasets_synthetic/synthetic_multi_mw_raw).",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to each spectrum (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    real_meta_dir = (script_dir / args.real_meta_dir).resolve()
    output_dir    = (script_dir / args.output_dir).resolve()

    print(f"Loading labels from: {real_meta_dir}")
    raw_labels = load_raw_labels(str(real_meta_dir))
    print(f"  → {len(raw_labels)} samples  "
          f"Ax=[{raw_labels['Ax'].min():.3f}, {raw_labels['Ax'].max():.3f}]")

    build_dataset(
        output_dir=str(output_dir),
        raw_labels=raw_labels,
        mw_configs=MW_CONFIGS,
        noise_std=args.noise_std,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
