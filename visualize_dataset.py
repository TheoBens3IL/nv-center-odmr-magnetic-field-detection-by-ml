from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider, CheckButtons


def plot_spectrum_multi_mw(dataset_dir):
    """
    Interactive visualization of raw SPLIT files from multi-MW dataset.
    Allows exploring raw signal/background data before preprocessing.
    
    Parameters:
        dataset_dir : str
            Path to dataset_10ElliptConf directory
    """
    from pathlib import Path
    
    dataset_dir = Path(dataset_dir)
    
    # Load currents
    currents_csv = list(dataset_dir.glob('3Dcurrents_sweep_*.csv'))[0]
    currents_data = pd.read_csv(currents_csv, header=None)
    Ax = currents_data.iloc[0].values
    Ay = currents_data.iloc[1].values
    Az = currents_data.iloc[2].values
    
    # Get all SPLIT files
    split_files = sorted(dataset_dir.glob('*SPLIT*Raw.txt'))
    num_experiments = len(split_files)
    
    print(f"Found {num_experiments} SPLIT files")
    
    # Load first file to get structure
    data_first = np.loadtxt(split_files[0], delimiter='\t')
    freq_line = data_first[0, :]
    
    # Test different extraction hypotheses
    print(f"Data shape: {data_first.shape}")
    print(f"First line length: {len(freq_line)}")
    print(f"First 10 values of line 1: {freq_line[:10]}")
    print(f"Values at positions 200-210: {freq_line[200:210]}")
    
    # Hypothesis 1: alternating signal/bg for each freq (current)
    # Hypothesis 2: all signals first, then all backgrounds
    # Let's test if first half and second half are different
    
    # Try: first 201 columns = signals, last 201 columns = backgrounds
    measurements = data_first[1:, :]  # Lines 2-101
    
    if len(freq_line) == 402:
        # Check if first half == second half for frequencies
        first_half_freq = freq_line[:201]
        second_half_freq = freq_line[201:]
        
        if np.allclose(first_half_freq, second_half_freq):
            print("Frequencies are repeated: first 201 = last 201")
            frequencies = first_half_freq / 1e9
            # Structure: columns 0-200 = signals, columns 201-401 = backgrounds
            signals_raw = measurements[:, :201]
            backgrounds_raw = measurements[:, 201:]
        else:
            print("Frequencies alternate")
            frequencies = freq_line[::2] / 1e9
            # Structure: alternating signal/background
            signals_raw = measurements[:, ::2]
            backgrounds_raw = measurements[:, 1::2]
    else:
        frequencies = freq_line / 1e9
        signals_raw = measurements
        backgrounds_raw = None
    
    print(f"Extracted frequencies: {len(frequencies)} points")
    print(f"Signals shape: {signals_raw.shape}")
    if backgrounds_raw is not None:
        print(f"Backgrounds shape: {backgrounds_raw.shape}")
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
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
    
    # Plot ODMR contrast (normalized - 1)
    contrast = normalized - 1.0
    line3, = ax3.plot(frequencies, contrast, 'm-')
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('ODMR Contrast')
    ax3.set_title('ODMR Contrast (S/B - 1)')
    ax3.grid(True, alpha=0.3)
    
    # Create sliders
    ax_exp = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_meas = plt.axes([0.15, 0.06, 0.7, 0.03])
    
    slider_exp = Slider(ax_exp, 'Experiment', 0, num_experiments - 1, valinit=0, valstep=1)
    slider_meas = Slider(ax_meas, 'Measurement', 0, 99, valinit=0, valstep=1)  # 100 measurements per file
    
    # Update function
    def update(val=None):
        exp_id = int(slider_exp.val)
        meas_id = int(slider_meas.val)
        
        # Load data for this experiment
        data = np.loadtxt(split_files[exp_id], delimiter='\t')
        freq_line_update = data[0, :]
        measurements = data[1:, :]
        
        # Use same extraction logic
        if len(freq_line_update) == 402:
            first_half = freq_line_update[:201]
            second_half = freq_line_update[201:]
            if np.allclose(first_half, second_half):
                signals_raw = measurements[:, :201]
                backgrounds_raw = measurements[:, 201:]
            else:
                signals_raw = measurements[:, ::2]
                backgrounds_raw = measurements[:, 1::2]
        else:
            signals_raw = measurements
            backgrounds_raw = measurements
        
        # Update plots
        line1.set_ydata(signals_raw[meas_id, :])
        line1b.set_ydata(backgrounds_raw[meas_id, :])
        
        normalized = signals_raw[meas_id, :] / backgrounds_raw[meas_id, :]
        line2.set_ydata(normalized)
        
        contrast = normalized - 1.0
        line3.set_ydata(contrast)
        
        # Update titles
        ax1.set_title(f'Exp {exp_id} — Raw Signal & Background (meas {meas_id}) — Ax={Ax[exp_id]:.2f} A, Ay={Ay[exp_id]:.2f} A, Az={Az[exp_id]:.2f} A')
        
        # Auto-scale y-axis
        for ax in [ax1, ax2, ax3]:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        
        fig.canvas.draw_idle()
    
    # Connect sliders
    slider_exp.on_changed(update)
    slider_meas.on_changed(update)
    
    plt.show()


# def plot_spectrum(frequencies, raw_signals, raw_backgrounds, currents):
#     """
#     Interactive visualization of ODMR spectra with sliders and processing options.
    
#     Parameters:
#         frequencies : array (num_freq,)
#             Microwave frequency axis in GHz
#         raw_signals : array (num_experiments, num_freq, num_measurements)
#             Raw ODMR signal data
#         raw_backgrounds : array (num_experiments, num_freq, num_measurements)
#             Background signal data
#         currents : array (num_experiments, 3)
#             Currents applied in each experiment
#     """
#     num_experiments, num_freq, num_measurements = raw_signals.shape
#     Ax, Ay, Az = currents[:, 0], currents[:, 1], currents[:, 2]
    
#     # Create figure with sliders and checkboxes
#     fig, ax = plt.subplots(figsize=(12, 7))
#     plt.subplots_adjust(bottom=0.25, left=0.3)
    
#     # Initial plot (config=0, measurement=0)
#     line, = ax.plot(frequencies, raw_signals[0, :, 0], 'b-')
#     ax.set_xlabel('Frequency (GHz)', fontsize=12)
#     ax.set_ylabel('Signal (a.u.)', fontsize=12)
#     ax.set_title('ODMR Spectrum — Config 0, Measurement 0 — Currents: Ax=%.2f A, Ay=%.2f A, Az=%.2f A' % (Ax[0], Ay[0], Az[0]))
#     ax.grid(True, alpha=0.3)
    
#     # Create sliders
#     ax_config = plt.axes([0.35, 0.12, 0.55, 0.03])
#     ax_measurement = plt.axes([0.35, 0.06, 0.55, 0.03])
#     ax_mw = plt.axes([0.35, 0.00, 0.55, 0.03])  # MW config slider at different position
    
#     slider_config = Slider(ax_config, 'Current config', 0, num_experiments - 1, valinit=0, valstep=1)
#     slider_measurement = Slider(ax_measurement, 'Measurement', 0, num_measurements - 1, valinit=0, valstep=1)
#     slider_mw = Slider(ax_mw, 'MW config', 0, 4, valinit=0, valstep=1)
#     # Hide MW slider initially
#     ax_mw.set_visible(False)
    
#     # Create checkboxes for processing options
#     ax_check = plt.axes([0.015, 0.2, 0.22, 0.25])
#     labels = ['Normalize by background', 'ODMR contrast', 'Normalize per spectrum', 'Average per MW config', 'Normalize global', 'Normalize global percentile']
#     visibility = [False, False, False, False, False, False]
#     check = CheckButtons(ax_check, labels, visibility)
    
#     # Update function
#     def update(val=None):
#         config_id = int(slider_config.val)
#         measurement_id = int(slider_measurement.val)
#         mw_id = int(slider_mw.val)
        
#         # Get checkbox states
#         normalize_bg = check.get_status()[0]
#         apply_contrast = check.get_status()[1]
#         normalize_spec = check.get_status()[2]
#         apply_average = check.get_status()[3]
#         normalize_global_flag = check.get_status()[4]
#         normalize_global_percentile_flag = check.get_status()[5]

#         # Start with raw signals
#         processed = raw_signals.copy()
        
#         # Apply transformations in order
#         if normalize_bg:
#             processed = normalize_by_background(processed, raw_backgrounds)
        
#         if apply_contrast:
#             processed = odmr_contrast(processed)
        
#         if normalize_spec:
#             processed = normalize_per_spectrum(processed)

#         if normalize_global_flag:
#             processed = normalize_global(processed)

#         if normalize_global_percentile_flag:
#             processed = normalize_global_percentile(processed)
        
#         if apply_average:
#             processed = average_per_mw_config(processed, n_repeat_per_mw=100)
#             # After averaging, shape changes to (num_experiments, n_mw, num_freq)
            
#             # Show MW slider, hide measurement slider
#             slider_measurement.ax.set_visible(False)
#             ax_mw.set_visible(True)
            
#             # Update config slider to show original experiment configs
#             slider_config.valmax = num_experiments - 1
#             slider_config.ax.set_xlim(0, num_experiments - 1)
            
#             # Plot: direct indexing with config_id and mw_id
#             config_id = min(int(slider_config.val), num_experiments - 1)
#             mw_id = int(slider_mw.val)
            
#             line.set_ydata(processed[config_id, mw_id, :])
            
#             # Update title
#             ax.set_title(f'ODMR Spectrum — Config {config_id}, MW block {mw_id} — '
#                         f'Currents: Ax={Ax[config_id]:.2f} A, Ay={Ay[config_id]:.2f} A, Az={Az[config_id]:.2f} A')
#         else:
#             # Reset slider ranges to original
#             slider_config.valmax = num_experiments - 1
#             slider_config.ax.set_xlim(0, num_experiments - 1)
#             # Re-enable measurement slider, hide MW slider
#             slider_measurement.ax.set_visible(True)
#             ax_mw.set_visible(False)
            
#             # Plot with measurement dimension
#             config_id = min(int(slider_config.val), num_experiments - 1)
#             measurement_id = int(slider_measurement.val)
#             line.set_ydata(processed[config_id, :, measurement_id])
            
#             # Update title
#             ax.set_title(f'ODMR Spectrum — Config {config_id}, Measurement {measurement_id} — '
#                         f'Currents: Ax={Ax[config_id]:.2f} A, Ay={Ay[config_id]:.2f} A, Az={Az[config_id]:.2f} A')
        
#         # Auto-scale y-axis
#         ax.relim()
#         ax.autoscale_view(scalex=False, scaley=True)
#         fig.canvas.draw_idle()
    
#     # Connect sliders and checkboxes
#     slider_config.on_changed(update)
#     slider_measurement.on_changed(update)
#     slider_mw.on_changed(update)
#     check.on_clicked(lambda label: update())
    
#     plt.show()


def plot_pytorch_dataset(dataset_dir):
    """
    Interactive visualization of PyTorch dataset.
    
    Parameters:
        dataset_dir : str
            Path to the PyTorch dataset directory containing:
            - frequencies.npy
            - metadata.csv
            - signals/ folder with config_XXX.npy files
    """
    # Load dataset components
    frequencies = np.load(os.path.join(dataset_dir, "frequencies.npy"))
    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
    
    num_configs = len(metadata)
    Ax = metadata["Ax"].values
    Ay = metadata["Ay"].values
    Az = metadata["Az"].values
    
    # Load first config to get shape
    first_signal = np.load(os.path.join(dataset_dir, "signals", "config_0000.npy"))
    num_mw_configs = first_signal.shape[0]  # (num_mw_configs, num_freq) - should be 10
    
    # Create figure with sliders
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial plot (experiment=0, mw_config=0)
    line, = ax.plot(frequencies / 1e9, first_signal[0, :], 'b-')  # Convert to GHz
    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Normalized Signal (a.u.)', fontsize=12)
    ax.set_title('PyTorch Dataset — Experiment 0, MW config 0 — Currents: Ax=%.2f A, Ay=%.2f A, Az=%.2f A' % (Ax[0], Ay[0], Az[0]))
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
        
        # Load the signal file for this experiment
        signal_file = os.path.join(dataset_dir, "signals", f"config_{exp_id:04d}.npy")
        signals = np.load(signal_file)  # (num_mw_configs, num_freq) - (10, 201)
        
        # Update plot
        line.set_ydata(signals[mw_id, :])
        ax.set_title(f'PyTorch Dataset — Experiment {exp_id}, MW config {mw_id} — '
                    f'Currents: Ax={Ax[exp_id]:.2f} A, Ay={Ay[exp_id]:.2f} A, Az={Az[exp_id]:.2f} A')
        
        # Auto-scale y-axis
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()
    
    # Connect sliders
    slider_exp.on_changed(update)
    slider_mw.on_changed(update)
    
    plt.show()


if __name__ == "__main__":
    # === Option 1: Visualize raw SPLIT files (multi-MW) ===
    # DATASET_DIR_RAW = "dataset_10ElliptConf"
    # plot_spectrum_multi_mw(DATASET_DIR_RAW)
    
    # === Option 2: Visualize processed PyTorch dataset ===
    DATASET_DIR = "dataset_multi_mw"
    plot_pytorch_dataset(DATASET_DIR)