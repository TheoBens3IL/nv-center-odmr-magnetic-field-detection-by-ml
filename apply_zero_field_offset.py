"""Apply zero-field current offset correction to a PyTorch dataset."""

import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import denormalize_labels, load_normalization_stats, split_zones

# DAC limits from MaGNiFi3D_ESR_3D_magnetometry_uniform.py (new_2 / new_3 era)
DEFAULT_DAC_LIMITS = np.array(
    [
        [0x6000, 0xA000],  # Ax / channel c2
        [0x4800, 0xB800],  # Ay / channel c3
        [0x4800, 0xB800],  # Az / channel c4
    ],
    dtype=np.int64,
)

# Zero-field calibration DAC (dataset_new_3, post magnetic-offset removal)
DEFAULT_ZERO_FIELD_DAC = np.array([0x7F66, 0x7EA0, 0x8175], dtype=np.int64)


def current_offset_from_dac(
    zero_field_dac=None,
    dac_limits=None,
    full_scale_amp=(1.1, 1.1, 1.1),
):
    """Estimate offset currents (A) from zero-field DAC codes."""
    if zero_field_dac is None:
        zero_field_dac = DEFAULT_ZERO_FIELD_DAC
    if dac_limits is None:
        dac_limits = DEFAULT_DAC_LIMITS

    zero_field_dac = np.asarray(zero_field_dac, dtype=np.float64).reshape(3)
    dac_limits = np.asarray(dac_limits, dtype=np.float64).reshape(3, 2)
    full_scale = np.asarray(full_scale_amp, dtype=np.float64).reshape(3)

    mid = dac_limits.mean(axis=1)
    half = (dac_limits[:, 1] - dac_limits[:, 0]) / 2.0
    delta_dac = zero_field_dac - mid
    return (delta_dac / half) * full_scale


def correct_measured_currents(currents_amp, offset_amp):
    """Subtract zero-field current offset from measured currents."""
    currents = np.asarray(currents_amp, dtype=np.float64)
    offset = np.asarray(offset_amp, dtype=np.float64).reshape(3)
    return currents - offset


def describe_offset(offset_amp, label="Current offset"):
    offset = np.asarray(offset_amp, dtype=np.float64).reshape(3)
    print(f"{label} (A): Ax={offset[0]:+.4f}, Ay={offset[1]:+.4f}, Az={offset[2]:+.4f}")
    print(f"  |offset| = {np.linalg.norm(offset):.4f} A")


def load_labels_amp(dataset_dir):
    stats = load_normalization_stats(dataset_dir)
    meta = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    labels_norm = meta[["Ax", "Ay", "Az"]].values.astype(np.float64)
    labels_amp = denormalize_labels(labels_norm, stats["labels_mean"], stats["labels_std"])
    if hasattr(labels_amp, "numpy"):
        labels_amp = labels_amp.numpy()
    return labels_amp, meta, stats


def compare_zones(labels_ref, labels_src, offset_amp, name=""):
    zones_ref = split_zones(labels_ref)
    zones_raw = split_zones(labels_src)
    labels_corr = correct_measured_currents(labels_src, offset_amp)
    zones_corr = split_zones(labels_corr)
    agree_raw = (zones_ref == zones_raw).mean() * 100
    agree_corr = (zones_ref == zones_corr).mean() * 100
    print(f"\nZone agreement vs reference{name}:")
    print(f"  Before correction : {agree_raw:.2f}% ({(zones_ref != zones_raw).sum()} mismatches)")
    print(f"  After correction  : {agree_corr:.2f}% ({(zones_ref != zones_corr).sum()} mismatches)")
    return labels_corr, zones_corr


def write_corrected_dataset(source_dir, output_dir, labels_corr):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_mean = labels_corr.mean(axis=0).astype(np.float32)
    global_std = float(max(labels_corr[:, 0].std(), labels_corr[:, 1].std(), labels_corr[:, 2].std()))
    labels_std = np.array([global_std, global_std, global_std], dtype=np.float32)

    labels_norm = ((labels_corr - labels_mean) / (labels_std + 1e-8)).astype(np.float32)

    meta = pd.read_csv(source_dir / "metadata.csv")
    meta["Ax"] = labels_norm[:, 0]
    meta["Ay"] = labels_norm[:, 1]
    meta["Az"] = labels_norm[:, 2]
    meta.to_csv(output_dir / "metadata.csv", index=False)

    np.save(
        output_dir / "normalization_stats.npy",
        {"labels_mean": labels_mean, "labels_std": labels_std},
    )

    for fname in ("frequencies.npy",):
        src = source_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    signals_src = source_dir / "signals"
    signals_dst = output_dir / "signals"
    if signals_dst.exists():
        shutil.rmtree(signals_dst)
    shutil.copytree(signals_src, signals_dst)

    print(f"\nCorrected dataset written to: {output_dir}")
    print(f"  labels_mean: {labels_mean}")
    print(f"  labels_std : {labels_std}")


def main():
    parser = argparse.ArgumentParser(description="Correct labels for zero-field current offset")
    parser.add_argument("--source_dir", required=True, help="PyTorch dataset (e.g. dataset_new_3)")
    parser.add_argument("--output_dir", default=None, help="Output under datasets_pytorch/")
    parser.add_argument("--reference_dir", default="dataset_new_2", help="Reference for zone comparison")
    parser.add_argument(
        "--offset_amp",
        type=float,
        nargs=3,
        default=None,
        metavar=("Ax", "Ay", "Az"),
        help="Measured offset in A (Keithley at zero-field DAC). Overrides DAC estimate.",
    )
    parser.add_argument(
        "--zero_dac",
        type=lambda s: int(s, 0),
        nargs=3,
        default=list(DEFAULT_ZERO_FIELD_DAC),
        metavar=("Ax", "Ay", "Az"),
        help="Zero-field DAC codes (default: 0x7f66 0x7ea0 0x8175)",
    )
    parser.add_argument(
        "--full_scale_amp",
        type=float,
        nargs=3,
        default=[1.1, 1.1, 1.1],
        help="|I| at fv=±1 per channel for DAC linear estimate",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print zone agreement, do not write")
    args = parser.parse_args()

    def resolve(name):
        p = Path("datasets_pytorch") / name
        if not p.exists():
            p = Path(name)
        return p

    source = resolve(args.source_dir)
    reference = resolve(args.reference_dir)

    labels_src, _, _ = load_labels_amp(source)
    labels_ref, _, _ = load_labels_amp(reference)

    if args.offset_amp is not None:
        offset = np.array(args.offset_amp, dtype=np.float64)
        describe_offset(offset, "Using measured offset")
    else:
        offset = current_offset_from_dac(
            zero_field_dac=args.zero_dac,
            full_scale_amp=args.full_scale_amp,
        )
        describe_offset(offset, "Estimated offset from DAC codes")
        print(f"  DAC zero: {[hex(x) for x in args.zero_dac]}")

    labels_corr, _ = compare_zones(labels_ref, labels_src, offset, name=f" ({source.name} vs {reference.name})")

    if args.dry_run:
        return

    out_name = args.output_dir or f"{source.name}_zcorr"
    out_path = Path("datasets_pytorch") / out_name
    write_corrected_dataset(source, out_path, labels_corr)


if __name__ == "__main__":
    main()
