"""Prepare a label-free MW sweep PyTorch dataset from SPLIT files."""


import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def resolve_input_files(path_str):
    """Return the list of SPLIT files."""
    path = Path(path_str)
    if path.is_file():
        return [path]
    if path.is_dir():
        candidates = sorted(path.glob("*SPLIT*Raw*.txt"))
        if not candidates:
            candidates = sorted(path.glob("*SPLIT*Raw*"))
        if not candidates:
            candidates = sorted(path.glob("*SPLIT*"))
        if not candidates:
            raise FileNotFoundError(f"No SPLIT file in {path}")
        return candidates
    for candidate in (path, Path(str(path) + ".txt")):
        if candidate.is_file():
            return [candidate]
    raise FileNotFoundError(f"File or directory not found: {path_str}")


def load_esr_split(esr_file, n_mw_angles=None):
    data = np.loadtxt(esr_file, delimiter="\t")
    frequencies = data[0, :201].astype(np.float32)
    measurements = data[1:, :]
    n_total = measurements.shape[0]

    signals = measurements[:, :201]
    backgrounds = measurements[:, 201:402]
    normalized = signals / backgrounds - 1.0

    if n_mw_angles is not None and n_total != n_mw_angles:
        raise ValueError(
            f"{esr_file.name}: expected {n_mw_angles} rows, found {n_total}."
        )
    signals_array = normalized.astype(np.float32)

    return frequencies, signals_array


def normalize_global(signals):
    mean = signals.mean()
    std = signals.std()
    return (signals - mean) / (std + 1e-8), float(mean), float(std)


def create_mw_sweep_dataset(input_path, output_dir, n_mw_angles=None, channel_IQ_ratio=None):
    input_files = resolve_input_files(input_path)
    output_dir = Path(output_dir)
    signals_dir = output_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    if channel_IQ_ratio is not None:
        if len(channel_IQ_ratio) != len(input_files):
            raise ValueError(f"--IQ_ratio: {len(channel_IQ_ratio)} values for {len(input_files)} files")
    else:
        channel_IQ_ratio = [1.0] * len(input_files)

    all_signals = []
    frequencies = None
    rows = []

    for exp_id, (esr_file, scale) in enumerate(zip(input_files, channel_IQ_ratio)):
        freq, signals = load_esr_split(esr_file, n_mw_angles=n_mw_angles)
        if frequencies is None:
            frequencies = freq
        elif not np.allclose(freq, frequencies, rtol=1e-5, atol=1e-3):
            raise ValueError(f"Frequency axis mismatch in {esr_file.name}")
        all_signals.append(signals)
        rows.append(
            {
                "experiment_id": exp_id,
                "channel_scale": float(scale),
                "source_file": esr_file.name,
                "n_angles": signals.shape[0],
                "Ax": 0.0,
                "Ay": 0.0,
                "Az": 0.0,
            }
        )

    all_signals = np.stack(all_signals, axis=0)  # (n_exp, n_angles, n_freq)
    all_signals, sig_mean, sig_std = normalize_global(all_signals)

    np.save(output_dir / "frequencies.npy", frequencies)
    np.save(output_dir / "normalization_stats.npy",
        {
            "labels_mean": np.zeros(3, dtype=np.float32),
            "labels_std": np.ones(3, dtype=np.float32),
        },
    )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    for exp_id in tqdm(range(len(input_files)), desc="Saving signals"):
        np.save(
            signals_dir / f"config_{exp_id:04d}.npy",
            all_signals[exp_id].astype(np.float32),
        )

    n_mw = all_signals.shape[1]
    n_freq = all_signals.shape[2]

    print(f"\n{'=' * 60}")
    print("Label-free dataset ready")
    print(f"{'=' * 60}")
    print(f"Input        : {input_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Experiments  : {len(input_files)} (1 per MW amplitude condition)")
    print(f"Angles / exp : {n_mw} MW configs")
    print(f"Frequencies  : {n_freq}")
    for row in rows:
        print(
            f"  exp {row['experiment_id']:02d} | scale={row['channel_scale']:.2f} | "
            f"{row['source_file']}"
        )
    print(f"Signal norm  : global z-score (mean={sig_mean:.4f}, std={sig_std:.4f})")
    print(f"Signal range : [{all_signals.min():.4f}, {all_signals.max():.4f}]")
    print(f"\nVisualize:")
    print(f"  py visualize_dataset.py --mode pytorch --dataset_dir {output_dir.name}")
    print("  Slider 'Experiment' = scale condition (0.7 / 1.0 / 1.3)")
    print("  Slider 'MW config'  = angle (0..119)")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="SPLIT file or directory with multiple files (e.g. 3-scale sweep)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Name under datasets_pytorch/",
    )
    parser.add_argument(
        "--n_mw_angles",
        type=int,
        default=120,
        help="Number of MW angles per file (default: 120)",
    )
    parser.add_argument("--IQ_ratio", type=float, nargs="+", default=None, help="MW amplitude factor per file, sorted order (e.g. 0.7 1.0 1.3)")
    parser.add_argument("--visualize", action="store_true", help="Open interactive visualization after preparation")
    args = parser.parse_args()

    input_files = resolve_input_files(args.input_file)
    if args.IQ_ratio is None and len(input_files) == 3:
        args.IQ_ratio = [0.7, 1.0, 1.3]
        print("Auto IQ_ratio (3 files): 0.7, 1.0, 1.3")

    if args.output_dir:
        output_dir = Path("datasets_pytorch") / args.output_dir
    else:
        base = Path(args.input_file).name if Path(args.input_file).is_dir() else Path(args.input_file).stem
        output_dir = Path("datasets_pytorch") / f"{base}_preview"
    Path("datasets_pytorch").mkdir(exist_ok=True)

    out = create_mw_sweep_dataset(args.input_file, output_dir, n_mw_angles=args.n_mw_angles, pre_averaged=not args.no_pre_averaged, normalize_signals=not args.no_normalize_signals, channel_IQ_ratio=args.IQ_ratio)

    if args.visualize:
        from visualize_dataset import plot_pytorch_dataset
        plot_pytorch_dataset(str(out))


if __name__ == "__main__":
    main()
