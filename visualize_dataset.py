from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import argparse
from physics_informed import extract_odmr_peak_frequencies


def extract_signals_and_backgrounds(data):
    """
    Extract frequencies, signals, and backgrounds from a SPLIT file.
    Assumes signals and backgrounds are always of size 201 each.
    Returns:
        frequencies (GHz), signals_raw, backgrounds_raw
    """
    freq_line = data[0, :]
    measurements = data[1:, :]
    frequencies = freq_line[:201] / 1e9
    signals_raw = measurements[:, :201]
    backgrounds_raw = measurements[:, 201:]
    return frequencies, signals_raw, backgrounds_raw


def plot_raw_dataset(dataset_dir, filter_spectra=False, filter_type='gaussian', filter_kwargs=None):
    """
    Interactive visualization of raw SPLIT files from multi-MW dataset.
    Allows exploring raw signal/background data before preprocessing.

    Parameters:
        dataset_dir : str
            Path to dataset_10ElliptConf directory
    """
    from pathlib import Path

    dataset_dir = Path(dataset_dir)

    currents_csv_list = list(dataset_dir.glob('3Dcurrents_sweep_*.csv'))
    if currents_csv_list:
        currents_data = pd.read_csv(currents_csv_list[0], header=None)
        Ax = currents_data.iloc[0].values
        Ay = currents_data.iloc[1].values
        Az = currents_data.iloc[2].values
    else:
        meta_path = dataset_dir / 'metadata.csv'
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No '3Dcurrents_sweep_*.csv' nor 'metadata.csv' found in {dataset_dir}"
            )
        currents_data = pd.read_csv(meta_path)
        Ax = currents_data['Ax'].values
        Ay = currents_data['Ay'].values
        Az = currents_data['Az'].values

    # Get all SPLIT files — if none exist, this is a numpy-format dataset
    split_files = sorted(dataset_dir.glob('*SPLIT*Raw.txt'))
    if len(split_files) == 0:
        npy_signals = sorted((dataset_dir / 'signals').glob('config_*.npy'))
        if npy_signals and (dataset_dir / 'frequencies.npy').exists():
            print(f"No SPLIT files found in {dataset_dir}. "
                  f"Detected numpy-format dataset — switching to pytorch visualization.")
            plot_pytorch_dataset(str(dataset_dir))
            return
        raise FileNotFoundError(
            f"No SPLIT files nor numpy signals found in {dataset_dir}"
        )
    num_experiments = len(split_files)

    data_first = np.loadtxt(split_files[0], delimiter='\t')
    frequencies, signals_raw, backgrounds_raw = extract_signals_and_backgrounds(data_first)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25, hspace=0.3)

    # Initial experiment
    exp_id = 0

    # Plot first measurement of each type
    line1, = ax1.plot(frequencies, signals_raw[0, :], 'b-', label='Signal (meas 0)')
    line1b, = ax1.plot(frequencies, backgrounds_raw[0, :], 'r-', alpha=0.5, label='Background (meas 0)')
    ax1.set_ylabel('Raw Signal (a.u.)')
    ax1.set_title(f'Exp {exp_id} — Raw Signal & Background — Ax={Ax[exp_id]:.2f} A, Ay={Ay[exp_id]:.2f} A, Az={Az[exp_id]:.2f} A')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot normalized (signal/background)
    normalized = signals_raw[0, :] / backgrounds_raw[0, :]
    line2, = ax2.plot(frequencies, normalized, 'g-')
    ax2.set_ylabel('Signal / Background')
    ax2.set_title('Normalized (S/B)')
    ax2.grid(True, alpha=0.3)

    # Create sliders and checkbox
    ax_exp = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_meas = plt.axes([0.15, 0.06, 0.7, 0.03])
    ax_filter = plt.axes([0.15, 0.18, 0.15, 0.04])
    ax_sigma = plt.axes([0.45, 0.18, 0.2, 0.03])
    ax_window = plt.axes([0.45, 0.22, 0.2, 0.03])
    ax_poly = plt.axes([0.45, 0.26, 0.2, 0.03])

    slider_exp = Slider(ax_exp, 'Experiment', 0, num_experiments - 1, valinit=0, valstep=1)
    slider_meas = Slider(ax_meas, 'Measurement', 0, signals_raw.shape[0] - 1, valinit=0, valstep=1)

    from matplotlib.widgets import CheckButtons
    filter_checkbox = CheckButtons(ax_filter, ['Apply Filter'], [filter_spectra])

    # Filter parameter sliders
    slider_sigma = Slider(ax_sigma, 'Sigma (Gaussian)', 0.1, 10.0, valinit=filter_kwargs['sigma'] if filter_kwargs and 'sigma' in filter_kwargs else 2.0)
    slider_window = Slider(ax_window, 'Window (Savgol)', 3, 51, valinit=filter_kwargs['window_length'] if filter_kwargs and 'window_length' in filter_kwargs else 15, valstep=2)
    slider_poly = Slider(ax_poly, 'Polyorder (Savgol)', 1, 10, valinit=filter_kwargs['polyorder'] if filter_kwargs and 'polyorder' in filter_kwargs else 3, valstep=1)

    def apply_filter(normed):
        if filter_checkbox.get_status()[0]:
            if filter_type == 'savgol':
                from scipy.signal import savgol_filter
                window_length = int(slider_window.val)
                polyorder = int(slider_poly.val)
                # Ensure window_length > polyorder and odd
                if window_length <= polyorder:
                    window_length = polyorder + 1
                if window_length % 2 == 0:
                    window_length += 1
                return savgol_filter(normed, window_length=window_length, polyorder=polyorder, axis=0)
            elif filter_type == 'gaussian':
                from scipy.ndimage import gaussian_filter1d
                sigma = slider_sigma.val
                return gaussian_filter1d(normed, sigma=sigma, axis=0)
            else:
                raise ValueError(f"Unknown filter_type: {filter_type}")
        else:
            return normed


    # Update function
    def update(val=None):
        exp_id = int(slider_exp.val)
        meas_id = int(slider_meas.val)
        data = np.loadtxt(split_files[exp_id], delimiter='\t')
        frequencies, signals_raw, backgrounds_raw = extract_signals_and_backgrounds(data)
        line1.set_ydata(signals_raw[meas_id, :])
        line1b.set_ydata(backgrounds_raw[meas_id, :])
        normalized = signals_raw[meas_id, :] / backgrounds_raw[meas_id, :]
        filtered_normed = apply_filter(normalized)
        line2.set_ydata(filtered_normed)
        ax1.set_title(f'Exp {exp_id} — Raw Signal & Background (meas {meas_id}) — Ax={Ax[exp_id]:.2f} A, Ay={Ay[exp_id]:.2f} A, Az={Az[exp_id]:.2f} A')
        for ax in [ax1, ax2]:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()

    # Checkbox callback
    def on_filter_checkbox(label):
        update()

    filter_checkbox.on_clicked(on_filter_checkbox)
    slider_exp.on_changed(update)
    slider_meas.on_changed(update)
    slider_sigma.on_changed(update)
    slider_window.on_changed(update)
    slider_poly.on_changed(update)
    update()
    plt.show()


def plot_pytorch_dataset(dataset_dir, peak_extraction=False):
    """
    Interactive visualization of PyTorch dataset.

    Parameters:
        dataset_dir : str
            Path to the PyTorch dataset directory containing:
            - frequencies.npy
            - metadata.csv
            - signals/ folder with config_XXX.npy files
    """
    frequencies = np.load(os.path.join(dataset_dir, "frequencies.npy"))
    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))

    num_configs = len(metadata)
    Ax = metadata["Ax"].values
    Ay = metadata["Ay"].values
    Az = metadata["Az"].values
    has_scale = "channel_scale" in metadata.columns
    channel_scales = metadata["channel_scale"].values if has_scale else None

    def _exp_subtitle(exp_id, mw_id):
        parts = [f"Experiment {exp_id}, MW angle {mw_id}"]
        if channel_scales is not None:
            parts.append(f"channel scale={channel_scales[exp_id]:.2f}")
        if not has_scale or (Ax[exp_id] == 0 and Ay[exp_id] == 0 and Az[exp_id] == 0):
            return " — ".join(parts)
        parts.append(f"Ax={Ax[exp_id]:.2f} A, Ay={Ay[exp_id]:.2f} A, Az={Az[exp_id]:.2f} A")
        return " — ".join(parts)

    first_signal = np.load(os.path.join(dataset_dir, "signals", "config_0000.npy"))
    num_mw_configs = first_signal.shape[0]  # (num_mw_configs, num_freq) - should be 10

    # Determine units for plotting (assume Hz if max is large)
    if np.max(frequencies) > 1e6:
        freq_plot = frequencies / 1e9  # Hz → GHz
    else:
        freq_plot = frequencies        # Already in GHz

    # Create figure with sliders
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.25)

    # Initial plot (experiment=0, mw_config=0)
    spectrum0 = first_signal[0, :]
    line, = ax.plot(freq_plot, spectrum0, 'b-', label='Spectrum')

    # Optional initial peak extraction and scatter overlay
    peak_scatter = None
    if peak_extraction:
        peak_freqs0 = extract_odmr_peak_frequencies(spectrum0, frequencies)

        if np.max(frequencies) > 1e6:
            peak_x0 = np.array(peak_freqs0) / 1e9
        else:
            peak_x0 = np.array(peak_freqs0)
        # Interpolate spectrum value at peak positions for y-coordinates
        peak_y0 = np.interp(peak_x0, freq_plot, spectrum0)
        valid0 = ~np.isnan(peak_x0) & ~np.isnan(peak_y0)
        peak_scatter = ax.scatter(peak_x0[valid0], peak_y0[valid0], color='r', marker='x', s=40, label='Detected peaks')

    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Normalized Signal (a.u.)', fontsize=12)
    ax.set_title('PyTorch Dataset — ' + _exp_subtitle(0, 0))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Create sliders
    ax_exp = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_mw = plt.axes([0.15, 0.06, 0.7, 0.03])

    slider_exp = Slider(ax_exp, 'Experiment', 0, num_configs - 1, valinit=0, valstep=1)
    slider_mw = Slider(ax_mw, 'MW config', 0, num_mw_configs - 1, valinit=0, valstep=1)

    # Update function
    def update(val=None):
        exp_id = int(slider_exp.val)
        mw_id = int(slider_mw.val)

        signal_file = os.path.join(dataset_dir, "signals", f"config_{exp_id:04d}.npy")
        signals = np.load(signal_file)  # (num_mw_configs, num_freq) - (10, 201)
        spectrum = signals[mw_id, :]

        # Update main spectrum line
        line.set_ydata(spectrum)

        # Recompute and update peak scatter overlay (if enabled)
        if peak_extraction and peak_scatter is not None:
            peak_freqs = extract_odmr_peak_frequencies(spectrum, frequencies)
            if np.max(frequencies) > 1e6:
                peak_x = np.array(peak_freqs) / 1e9
            else:
                peak_x = np.array(peak_freqs)
            peak_y = np.interp(peak_x, freq_plot, spectrum)
            # Handle possible NaNs from peak extraction
            valid = ~np.isnan(peak_x) & ~np.isnan(peak_y)
            peak_scatter.set_offsets(np.c_[peak_x[valid], peak_y[valid]])

        ax.set_title('PyTorch Dataset — ' + _exp_subtitle(exp_id, mw_id))

        # Auto-scale y-axis
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()

    # Connect sliders
    slider_exp.on_changed(update)
    slider_mw.on_changed(update)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Pytorch dataset or raw SPLIT files with interactive sliders")
    parser.add_argument('--mode', type=str, choices=['raw', 'pytorch'], default='pytorch', help='Visualization mode: raw for SPLIT files, pytorch for processed dataset')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--peak_extraction', action='store_true', help='If set in pytorch mode, run extract_odmr_peak_frequencies and overlay detected peaks as scatter')
    args = parser.parse_args()

    def resolve_dir(user_path, default_prefix):
        if os.path.isabs(user_path) or os.path.exists(user_path):
            return user_path
        prefixed = os.path.join(default_prefix, user_path)
        return prefixed

    if args.mode == 'raw':
        dataset_dir = DATASET_DIR = resolve_dir(args.dataset_dir, "datasets_raw")
        plot_raw_dataset(dataset_dir)
    else:
        dataset_dir = resolve_dir(args.dataset_dir, "datasets_pytorch")
        plot_pytorch_dataset(dataset_dir, peak_extraction=args.peak_extraction)