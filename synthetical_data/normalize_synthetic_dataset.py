"""Normalize a raw synthetic ODMR dataset for training."""


import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_dataset(input_dir: str, output_dir: str) -> None:
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    signals_in  = input_dir  / "signals"
    signals_out = output_dir / "signals"

    output_dir.mkdir(parents=True, exist_ok=True)
    signals_out.mkdir(exist_ok=True)

    signal_files = sorted(signals_in.glob("config_*.npy"))
    n_samples    = len(signal_files)
    if n_samples == 0:
        raise FileNotFoundError(f"No config_*.npy files found in {signals_in}")

    meta = pd.read_csv(input_dir / "metadata.csv")

    print(f"Loading {n_samples} signal files for global normalization …")
    sample = np.load(signal_files[0])          # (n_mw, Wnr)
    n_mw, Wnr = sample.shape
    all_signals = np.empty((n_samples, n_mw, Wnr), dtype=np.float32)
    for i, fpath in enumerate(tqdm(signal_files, desc="  loading")):
        all_signals[i] = np.load(fpath)

    sig_mean = float(all_signals.mean())
    sig_std  = float(all_signals.std())
    all_signals = ((all_signals - sig_mean) / (sig_std + 1e-8)).astype(np.float32)

    ax_vals = meta["Ax"].values.astype(np.float64)
    ay_vals = meta["Ay"].values.astype(np.float64)
    az_vals = meta["Az"].values.astype(np.float64)

    labels_mean = np.array([ax_vals.mean(), ay_vals.mean(), az_vals.mean()],
                           dtype=np.float32)
    global_std  = float(max(ax_vals.std(), ay_vals.std(), az_vals.std()))
    labels_std  = np.array([global_std, global_std, global_std], dtype=np.float32)

    normalization_stats = {"labels_mean": labels_mean, "labels_std": labels_std}
    np.save(output_dir / "normalization_stats.npy", normalization_stats)

    ax_norm = (ax_vals - labels_mean[0]) / (labels_std[0] + 1e-8)
    ay_norm = (ay_vals - labels_mean[1]) / (labels_std[1] + 1e-8)
    az_norm = (az_vals - labels_mean[2]) / (labels_std[2] + 1e-8)

    norm_meta = pd.DataFrame({
        "experiment_id": meta["experiment_id"].values,
        "Ax": ax_norm.astype(np.float32),
        "Ay": ay_norm.astype(np.float32),
        "Az": az_norm.astype(np.float32),
    })
    norm_meta.to_csv(output_dir / "metadata.csv", index=False)

    shutil.copy2(input_dir / "frequencies.npy", output_dir / "frequencies.npy")

    print("Saving normalized signal files …")
    for exp_id in tqdm(range(n_samples)):
        np.save(signals_out / f"config_{exp_id:04d}.npy", all_signals[exp_id])

    print(f"\n{'='*60}")
    print("Normalization complete.")
    print(f"{'='*60}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Samples: {n_samples}  |  Shape per file: ({n_mw}, {Wnr})")
    print(f"\nSignal stats (normalized):")
    print(f"  mean={all_signals.mean():.4f}  std={all_signals.std():.4f}"
          f"  min={all_signals.min():.4f}  max={all_signals.max():.4f}")
    print(f"\nLabel normalization:")
    print(f"  mean : Ax={labels_mean[0]:.4f}  Ay={labels_mean[1]:.4f}  Az={labels_mean[2]:.4f}")
    print(f"  std  : {global_std:.6f}  (max of individual stds, same for all axes)")
    for name, arr in [("Ax", ax_norm), ("Ay", ay_norm), ("Az", az_norm)]:
        print(f"  {name} normalized : min={arr.min():.3f}  max={arr.max():.3f}"
              f"  mean={arr.mean():.3f}  std={arr.std():.3f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize a raw synthetic ODMR dataset for training."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the raw dataset produced by build_synthetic_dataset.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory for the normalized (training-ready) dataset.",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    normalize_dataset(str(input_dir), str(output_dir))


if __name__ == "__main__":
    main()
