from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider, CheckButtons
from prepare_dataset import load_currents, load_esr_raw, normalize_by_background, odmr_contrast, normalize_per_spectrum, average_per_mw_config, normalize_global, normalize_global_percentile

def plot_spectrum(frequencies, raw_signals, raw_backgrounds, currents):
    """
    Interactive visualization of ODMR spectra with sliders and processing options.
    
    Parameters:
        frequencies : array (num_freq,)
            Microwave frequency axis in GHz
        raw_signals : array (num_experiments, num_freq, num_measurements)
            Raw ODMR signal data
        raw_backgrounds : array (num_experiments, num_freq, num_measurements)
            Background signal data
        currents : array (num_experiments, 3)
            Currents applied in each experiment
    """
    num_experiments, num_freq, num_measurements = raw_signals.shape
    Ax, Ay, Az = currents[:, 0], currents[:, 1], currents[:, 2]
    
    # Create figure with sliders and checkboxes
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.25, left=0.3)
    
    # Initial plot (config=0, measurement=0)
    line, = ax.plot(frequencies, raw_signals[0, :, 0], 'b-')
    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Signal (a.u.)', fontsize=12)
    ax.set_title('ODMR Spectrum — Config 0, Measurement 0 — Currents: Ax=%.2f A, Ay=%.2f A, Az=%.2f A' % (Ax[0], Ay[0], Az[0]))
    ax.grid(True, alpha=0.3)
    
    # Create sliders
    ax_config = plt.axes([0.35, 0.12, 0.55, 0.03])
    ax_measurement = plt.axes([0.35, 0.06, 0.55, 0.03])
    ax_mw = plt.axes([0.35, 0.00, 0.55, 0.03])  # MW config slider at different position
    
    slider_config = Slider(ax_config, 'Current config', 0, num_experiments - 1, valinit=0, valstep=1)
    slider_measurement = Slider(ax_measurement, 'Measurement', 0, num_measurements - 1, valinit=0, valstep=1)
    slider_mw = Slider(ax_mw, 'MW config', 0, 4, valinit=0, valstep=1)
    # Hide MW slider initially
    ax_mw.set_visible(False)
    
    # Create checkboxes for processing options
    ax_check = plt.axes([0.015, 0.2, 0.22, 0.25])
    labels = ['Normalize by background', 'ODMR contrast', 'Normalize per spectrum', 'Average per MW config', 'Normalize global', 'Normalize global percentile']
    visibility = [False, False, False, False, False, False]
    check = CheckButtons(ax_check, labels, visibility)
    
    # Update function
    def update(val=None):
        config_id = int(slider_config.val)
        measurement_id = int(slider_measurement.val)
        mw_id = int(slider_mw.val)
        
        # Get checkbox states
        normalize_bg = check.get_status()[0]
        apply_contrast = check.get_status()[1]
        normalize_spec = check.get_status()[2]
        apply_average = check.get_status()[3]
        normalize_global_flag = check.get_status()[4]
        normalize_global_percentile_flag = check.get_status()[5]

        # Start with raw signals
        processed = raw_signals.copy()
        
        # Apply transformations in order
        if normalize_bg:
            processed = normalize_by_background(processed, raw_backgrounds)
        
        if apply_contrast:
            processed = odmr_contrast(processed)
        
        if normalize_spec:
            processed = normalize_per_spectrum(processed)

        if normalize_global_flag:
            processed = normalize_global(processed)

        if normalize_global_percentile_flag:
            processed = normalize_global_percentile(processed)
        
        if apply_average:
            processed = average_per_mw_config(processed, n_repeat_per_mw=100)
            # After averaging, shape changes to (num_experiments, n_mw, num_freq)
            
            # Show MW slider, hide measurement slider
            slider_measurement.ax.set_visible(False)
            ax_mw.set_visible(True)
            
            # Update config slider to show original experiment configs
            slider_config.valmax = num_experiments - 1
            slider_config.ax.set_xlim(0, num_experiments - 1)
            
            # Plot: direct indexing with config_id and mw_id
            config_id = min(int(slider_config.val), num_experiments - 1)
            mw_id = int(slider_mw.val)
            
            line.set_ydata(processed[config_id, mw_id, :])
            
            # Update title
            ax.set_title(f'ODMR Spectrum — Config {config_id}, MW block {mw_id} — '
                        f'Currents: Ax={Ax[config_id]:.2f} A, Ay={Ay[config_id]:.2f} A, Az={Az[config_id]:.2f} A')
        else:
            # Reset slider ranges to original
            slider_config.valmax = num_experiments - 1
            slider_config.ax.set_xlim(0, num_experiments - 1)
            # Re-enable measurement slider, hide MW slider
            slider_measurement.ax.set_visible(True)
            ax_mw.set_visible(False)
            
            # Plot with measurement dimension
            config_id = min(int(slider_config.val), num_experiments - 1)
            measurement_id = int(slider_measurement.val)
            line.set_ydata(processed[config_id, :, measurement_id])
            
            # Update title
            ax.set_title(f'ODMR Spectrum — Config {config_id}, Measurement {measurement_id} — '
                        f'Currents: Ax={Ax[config_id]:.2f} A, Ay={Ay[config_id]:.2f} A, Az={Az[config_id]:.2f} A')
        
        # Auto-scale y-axis
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()
    
    # Connect sliders and checkboxes
    slider_config.on_changed(update)
    slider_measurement.on_changed(update)
    slider_mw.on_changed(update)
    check.on_clicked(lambda label: update())
    
    plt.show()


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
    first_signal = np.load(os.path.join(dataset_dir, "signals", "config_000.npy"))
    num_measurements = first_signal.shape[0]  # (num_measurements, num_freq)
    
    # Create figure with sliders
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.25)
    
    # Initial plot (config=0, measurement=0)
    line, = ax.plot(frequencies, first_signal[0, :], 'b-')
    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Signal (a.u.)', fontsize=12)
    ax.set_title('PyTorch Dataset — Config 0, MW config 0 — Currents: Ax=%.2f A, Ay=%.2f A, Az=%.2f A' % (Ax[0], Ay[0], Az[0]))
    ax.grid(True, alpha=0.3)
    
    # Create sliders
    ax_config = plt.axes([0.15, 0.12, 0.7, 0.03])
    ax_measurement = plt.axes([0.15, 0.06, 0.7, 0.03])
    
    slider_config = Slider(ax_config, 'Config', 0, num_configs - 1, valinit=0, valstep=1)
    slider_measurement = Slider(ax_measurement, 'MW config', 0, num_measurements - 1, valinit=0, valstep=1)
    
    # Update function
    def update(val=None):
        config_id = int(slider_config.val)
        measurement_id = int(slider_measurement.val)
        
        # Load the signal file for this config
        signal_file = os.path.join(dataset_dir, "signals", f"config_{config_id:03d}.npy")
        signals = np.load(signal_file)  # (num_measurements, num_freq)
        
        # Update plot
        line.set_ydata(signals[measurement_id, :])
        ax.set_title(f'PyTorch Dataset — Config {config_id}, MW config {measurement_id} — '
                    f'Currents: Ax={Ax[config_id]:.2f} A, Ay={Ay[config_id]:.2f} A, Az={Az[config_id]:.2f} A')
        
        # Auto-scale y-axis
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()
    
    # Connect sliders
    slider_config.on_changed(update)
    slider_measurement.on_changed(update)
    
    plt.show()


if __name__ == "__main__":
    # === Option 1: Visualize raw dataset with processing options ===
    CURRENTS_FILE = "dataset_example/3Dcurrents_sweep_2026-01-26_23h00m53s.csv"
    ESR_FILE = "dataset_example/ESR_2026-01-26_23h00m57sRaw.txt"
    Ax, Ay, Az = load_currents(CURRENTS_FILE)
    frequencies, signals, backgrounds = load_esr_raw(ESR_FILE)
    plot_spectrum(frequencies, signals, backgrounds, np.column_stack((Ax, Ay, Az)))
    
    # === Option 2: Visualize PyTorch dataset ===
    # DATASET_DIR = "pytorch_dataset_example"
    # plot_pytorch_dataset(DATASET_DIR)