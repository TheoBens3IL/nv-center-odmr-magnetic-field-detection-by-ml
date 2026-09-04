"""Find MW configs that best attenuate each ODMR peak."""


import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from physics_informed import extract_odmr_peak_frequencies

NUM_PEAKS = 8
PEAK_HALF_WIDTH_MHZ = 12.0
PROMINENCE_FACTOR = 0.25
SWEEP_DIR = Path("datasets_raw/mw_sweep_3_third")
OUTPUT_DIR = Path("results/mw_sweep_3_best_configs")


def ensure_gui_backend():
    """Agg (default in many scripts) cannot open windows; switch to TkAgg on Windows."""
    import matplotlib

    if matplotlib.get_backend().lower().endswith("agg"):
        matplotlib.use("TkAgg", force=True)


def resolve_sweep_files(sweep_dir):
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {sweep_dir}")
    files = sorted(sweep_dir.glob("*SPLIT*Raw*.txt"))
    if not files:
        files = sorted(sweep_dir.glob("*SPLIT*"))
    if not files:
        raise FileNotFoundError(f"No SPLIT file in {sweep_dir}")
    return files


def load_esr_split(esr_file):
    data = np.loadtxt(esr_file, delimiter="\t")
    frequencies = data[0, :201].astype(np.float64)
    measurements = data[1:, :]
    contrast = measurements[:, :201] / measurements[:, 201:402] - 1.0
    return frequencies, contrast.astype(np.float64)


def load_sweep(sweep_dir):
    frequencies = None
    stacks = []
    names = []
    for esr_file in resolve_sweep_files(sweep_dir):
        freq, contrast = load_esr_split(esr_file)
        if frequencies is None:
            frequencies = freq
        elif not np.allclose(freq, frequencies, rtol=1e-5, atol=1e-3):
            raise ValueError(f"Frequency axis mismatch in {esr_file.name}")
        stacks.append(contrast)
        names.append(esr_file.name)
    return frequencies, np.stack(stacks, axis=0), names


def default_qi_ratios(n_files):
    if n_files == 9:
        return np.linspace(0.8, 1.2, 9, dtype=np.float64)
    if n_files == 3:
        return np.array([0.7, 1.0, 1.3], dtype=np.float64)
    return np.linspace(0.8, 1.2, n_files, dtype=np.float64)


def detect_reference_peaks(mean_spectrum, frequencies, num_peaks, prominence_factor):
    fitted = extract_odmr_peak_frequencies(
        mean_spectrum,
        frequencies,
        num_peaks=num_peaks,
        prominence_factor=prominence_factor,
    )
    valid = fitted[~np.isnan(fitted)]
    if len(valid) == 0:
        inverted = np.max(mean_spectrum) - mean_spectrum
        prom = max(np.std(inverted) * prominence_factor, 1e-8)
        peaks_idx, _ = find_peaks(inverted, distance=5, prominence=prom)
        if len(peaks_idx) == 0:
            raise RuntimeError("No ODMR peak detected on the mean spectrum.")
        valid = frequencies[peaks_idx]

    order = np.argsort(valid)
    peak_freqs = valid[order]
    peak_indices = np.array([int(np.argmin(np.abs(frequencies - f))) for f in peak_freqs])
    return peak_freqs, peak_indices


def dip_depth_near_peak(spectrum, frequencies, peak_freq_hz, half_width_hz):
    """
    Dip depth at the ODMR resonance closest to the reference peak frequency.

    Within a frequency window, detect local dip minima and pick the one nearest
    to peak_freq_hz (avoids grabbing a neighbouring peak when the window is wide).
    Higher contrast = shallower dip = better attenuation.
    """
    mask = np.abs(frequencies - peak_freq_hz) <= half_width_hz
    local_idx = np.where(mask)[0]
    if len(local_idx) == 0:
        idx = int(np.argmin(np.abs(frequencies - peak_freq_hz)))
        return float(spectrum[idx]), idx

    local_spec = spectrum[local_idx]
    inverted = np.max(local_spec) - local_spec
    prom = max(float(np.std(inverted)) * 0.1, 1e-8)
    dip_rel, _ = find_peaks(inverted, distance=2, prominence=prom)

    if len(dip_rel) == 0:
        rel = int(np.argmin(local_spec))
        idx = int(local_idx[rel])
        return float(spectrum[idx]), idx

    dip_indices = local_idx[dip_rel]
    idx = int(dip_indices[np.argmin(np.abs(frequencies[dip_indices] - peak_freq_hz))])
    return float(spectrum[idx]), idx


def score_all_configs(spectra, frequencies, peak_freqs_hz, half_width_hz):
    n_files, n_angles, _ = spectra.shape
    rows = []
    for peak_id, f_ref in enumerate(peak_freqs_hz):
        best_val = -np.inf
        best = None
        worst_val = np.inf
        worst = None
        all_vals = []

        for file_idx in range(n_files):
            for angle_idx in range(n_angles):
                val, _ = dip_depth_near_peak(
                    spectra[file_idx, angle_idx], frequencies, f_ref, half_width_hz
                )
                all_vals.append(val)
                if val > best_val:
                    best_val = val
                    best = (file_idx, angle_idx, val)
                if val < worst_val:
                    worst_val = val
                    worst = (file_idx, angle_idx, val)

        mean_val = float(np.mean(all_vals))
        file_idx, angle_idx, val = best
        worst_file_idx, worst_angle_idx, _ = worst
        rows.append(
            {
                "peak_id": peak_id + 1,
                "peak_freq_GHz": f_ref / 1e9,
                "best_file_idx": file_idx,
                "best_angle_idx": angle_idx,
                "worst_file_idx": worst_file_idx,
                "worst_angle_idx": worst_angle_idx,
                "best_contrast": val,
                "mean_contrast": mean_val,
                "worst_contrast": worst_val,
                "reduction": val - worst_val,
                "improvement_vs_mean": val - mean_val,
            }
        )
    return pd.DataFrame(rows)


def infer_angle_degrees(angle_idx, n_angles):
    if n_angles <= 1:
        arr = np.asarray(angle_idx, dtype=float)
        if arr.ndim == 0:
            return 0.0
        return np.zeros_like(arr, dtype=float)
    step = 360.0 / n_angles
    out = np.asarray(angle_idx, dtype=float) * step
    if out.ndim == 0:
        return float(out)
    return out


def dip_depth_at_peaks(spectra, frequencies, peak_freqs_hz, half_width_hz):
    """Dip depth per (peak, file, angle). Higher = shallower dip."""
    n_peaks = len(peak_freqs_hz)
    n_files, n_angles, _ = spectra.shape
    out = np.zeros((n_peaks, n_files, n_angles), dtype=np.float64)
    for pi, f_ref in enumerate(peak_freqs_hz):
        for fi in range(n_files):
            for ai in range(n_angles):
                val, _ = dip_depth_near_peak(
                    spectra[fi, ai], frequencies, f_ref, half_width_hz
                )
                out[pi, fi, ai] = val
    return out


def _freq_axis_ghz(frequencies):
    if frequencies.max() > 1e6:
        return frequencies / 1e9
    return frequencies


def _draw_attenuated_peak_on_ax(ax, spectra, frequencies, row, n_angles, half_width_hz):
    peak_hz = float(row["peak_freq_GHz"]) * 1e9
    peak_id = int(row["peak_id"])
    fi_best = int(row["best_file_idx"])
    ai_best = int(row["best_angle_idx"])
    angle_deg = infer_angle_degrees(ai_best, n_angles)
    spec = spectra[fi_best, ai_best]
    dip_val, dip_idx = dip_depth_near_peak(spec, frequencies, peak_hz, half_width_hz)
    freq_ghz = _freq_axis_ghz(frequencies)

    ax.plot(freq_ghz, spec, color="0.15", lw=1.4)
    ax.axvline(peak_hz / 1e9, color="tab:red", ls="--", lw=1.2, alpha=0.9, label="ref peak")
    ax.scatter(
        [freq_ghz[dip_idx]],
        [dip_val],
        color="tab:green",
        s=45,
        zorder=5,
        label=f"dip depth={dip_val:.4f}",
    )
    ax.set_title(
        f"Peak {peak_id} - {row['peak_freq_GHz']:.3f} GHz\n"
        f"MW config {fi_best}, angle {angle_deg:.1f} deg",
        fontsize=9,
    )
    ax.grid(True, alpha=0.25)


def show_attenuated_peak_spectra(spectra, frequencies, results, n_angles, half_width_hz):
    n_peaks = len(results)
    n_cols = min(4, n_peaks)
    n_rows = int(np.ceil(n_peaks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)

    for ax_idx, (_, row) in enumerate(results.iterrows()):
        r, c = divmod(ax_idx, n_cols)
        _draw_attenuated_peak_on_ax(
            axes[r, c], spectra, frequencies, row, n_angles, half_width_hz
        )

    for ax_idx in range(n_peaks, n_rows * n_cols):
        r, c = divmod(ax_idx, n_cols)
        axes[r, c].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (GHz)")
    fig.supylabel("ODMR contrast", fontsize=10)
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.88])
    fig.suptitle("Spectra at best MW configs", fontsize=12, y=0.95)


def _concatenated_angle_coordinates(n_files, n_angles):
    step = 360.0 / n_angles
    angles = np.arange(n_angles, dtype=np.float64) * step
    xs = [fi * 360.0 + angles for fi in range(n_files)]
    return np.concatenate(xs)


def show_fluorescence_vs_concatenated_angles(fluo, results, qi_ratios, n_angles):
    n_peaks = fluo.shape[0]
    n_files = fluo.shape[1]
    n_cols = min(4, n_peaks)
    n_rows = int(np.ceil(n_peaks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.2 * n_rows), squeeze=False)
    x_concat = _concatenated_angle_coordinates(n_files, n_angles)

    for pi, (_, row) in enumerate(results.iterrows()):
        r, c = divmod(pi, n_cols)
        ax = axes[r, c]
        ax.plot(x_concat, fluo[pi].reshape(-1), color="0.2", lw=0.8)
        for fi in range(1, n_files):
            ax.axvline(fi * 360.0, color="0.75", ls="--", lw=0.8)
        fi_best = int(row["best_file_idx"])
        ai_best = int(row["best_angle_idx"])
        x_best = fi_best * 360.0 + infer_angle_degrees(ai_best, n_angles)
        ax.axvline(x_best, color="tab:red", ls="--", lw=1.2, alpha=0.9)
        ax.set_title(
            f"Peak {int(row['peak_id'])} - {row['peak_freq_GHz']:.3f} GHz\n"
            f"best: file {fi_best} (Q/I={qi_ratios[fi_best]:.3g}), "
            f"{row['best_angle_deg']:.0f} deg",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)

    for pi in range(n_peaks, n_rows * n_cols):
        r, c = divmod(pi, n_cols)
        axes[r, c].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Angle (deg): 0-360 per file, concatenated")
    fig.supylabel("ODMR contrast at dip (S/B - 1)", fontsize=10)
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.90])
    fig.suptitle(
        "Dip depth at peak vs MW phase",
        fontsize=11,
        y=0.96,
    )


def show_qi_angle_heatmaps(fluo, results, qi_ratios, n_angles):
    n_peaks = fluo.shape[0]
    n_files = fluo.shape[1]
    n_cols = min(4, n_peaks)
    n_rows = int(np.ceil(n_peaks / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)
    angle_deg = infer_angle_degrees(np.arange(n_angles), n_angles)

    for pi, (_, row) in enumerate(results.iterrows()):
        r, c = divmod(pi, n_cols)
        ax = axes[r, c]
        qi_sorted_idx = np.argsort(qi_ratios)
        grid = fluo[pi][qi_sorted_idx]
        qi_y = qi_ratios[qi_sorted_idx]
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            extent=[angle_deg[0], angle_deg[-1], -0.5, n_files - 0.5],
            cmap="viridis",
        )
        ax.set_yticks(range(n_files))
        ax.set_yticklabels([f"{q:.3g}" for q in qi_y], fontsize=7)
        fi_best = int(row["best_file_idx"])
        fi_pos = int(np.where(qi_sorted_idx == fi_best)[0][0])
        ax.scatter(
            [row["best_angle_deg"]],
            [fi_pos],
            marker="x",
            color="white",
            s=40,
            linewidths=1.5,
            zorder=5,
        )
        ax.set_title(f"Peak {int(row['peak_id'])} - {row['peak_freq_GHz']:.3f} GHz", fontsize=9)
        plt.colorbar(im, ax=ax, label="Dip depth", pad=0.02)

    for pi in range(n_peaks, n_rows * n_cols):
        r, c = divmod(pi, n_cols)
        axes[r, c].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("MW phase (deg)")
    fig.supylabel("Q / I (MW ch1 / ch2)", fontsize=10)
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.90])
    fig.suptitle("Dip depth at peak vs MW phase and Q/I ratio", fontsize=11, y=0.96)


def show_sweep_analysis(spectra, frequencies, results, peak_freqs_hz, qi_ratios, half_width_hz, n_angles):
    fluo = dip_depth_at_peaks(spectra, frequencies, peak_freqs_hz, half_width_hz)
    show_fluorescence_vs_concatenated_angles(fluo, results, qi_ratios, n_angles)
    show_qi_angle_heatmaps(fluo, results, qi_ratios, n_angles)


def main():
    parser = argparse.ArgumentParser(description="Best MW config per ODMR peak (mw_sweep_3)")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show 2x4 grid of spectra at best MW config per peak",
    )
    parser.add_argument(
        "--show_sweep",
        action="store_true",
        help="Show fluorescence vs angle and Q/I heatmap figures",
    )
    args = parser.parse_args()

    if args.show or args.show_sweep:
        ensure_gui_backend()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies, spectra, file_names = load_sweep(SWEEP_DIR)
    n_files, n_angles, n_freq = spectra.shape
    half_width_hz = PEAK_HALF_WIDTH_MHZ * 1e6
    qi_ratios = default_qi_ratios(n_files)

    mean_spectrum = spectra.reshape(-1, n_freq).mean(axis=0)
    peak_freqs, _ = detect_reference_peaks(mean_spectrum, frequencies, NUM_PEAKS, PROMINENCE_FACTOR)

    print("=" * 60)
    print("MW config selection — ODMR peak attenuation")
    print("=" * 60)
    print(f"Sweep          : {SWEEP_DIR}")
    print(f"MW files       : {n_files}")
    print(f"Angles/file    : {n_angles}  (step ~ {360.0 / n_angles:.2f} deg)")
    print(f"Frequencies    : {n_freq} pts, {frequencies[0]/1e9:.3f}-{frequencies[-1]/1e9:.3f} GHz")
    print(f"Peaks detected : {len(peak_freqs)}")
    for i, f in enumerate(peak_freqs):
        print(f"  Peak {i+1}: {f/1e9:.4f} GHz")
    for fi, (name, qi) in enumerate(zip(file_names, qi_ratios)):
        print(f"  file {fi}: Q/I={qi:.4g}  |  {name[:55]}")

    results = score_all_configs(spectra, frequencies, peak_freqs, half_width_hz)
    results["best_source_file"] = results["best_file_idx"].map(
        {i: name for i, name in enumerate(file_names)}
    )
    results["best_angle_deg"] = results["best_angle_idx"].apply(
        lambda idx: infer_angle_degrees(int(idx), n_angles)
    )

    csv_path = OUTPUT_DIR / "best_mw_config_per_peak.csv"
    results.to_csv(csv_path, index=False)

    summary = {
        "sweep_dir": str(SWEEP_DIR),
        "n_mw_files": n_files,
        "n_angles": n_angles,
        "angle_step_deg": 360.0 / n_angles,
        "file_names": file_names,
        "qi_ratios": qi_ratios.tolist(),
        "peaks_GHz": (peak_freqs / 1e9).tolist(),
        "best_per_peak": results.to_dict(orient="records"),
    }
    json_path = OUTPUT_DIR / "best_mw_configs.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n--- Best MW config per peak (shallowest dip) ---")
    for _, row in results.iterrows():
        print(
            f"Peak {int(row['peak_id']):d} - {row['peak_freq_GHz']:.3f} GHz : "
            f"file={int(row['best_file_idx'])} angle={int(row['best_angle_idx'])} "
            f"({row['best_angle_deg']:.1f} deg) | contrast={row['best_contrast']:.5f} | "
            f"reduction={row['reduction']:.5f} | {row['best_source_file']}"
        )

    if args.show or args.show_sweep:
        if args.show_sweep:
            show_sweep_analysis(spectra, frequencies, results, peak_freqs, qi_ratios, half_width_hz, n_angles)
        if args.show:
            show_attenuated_peak_spectra(spectra, frequencies, results, n_angles, half_width_hz)
        plt.show()

    print(f"\nResults : {csv_path}")
    print(f"            {json_path}")


if __name__ == "__main__":
    main()
