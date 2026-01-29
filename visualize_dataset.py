from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from prepare_dataset import load_currents, load_esr_raw, normalize_by_background, odmr_contrast, normalize_per_spectrum

def plot_spectrum(frequencies, signals, currents):
    """
    Interactive visualization of ODMR spectra with sliders.
    
    Parameters:
        frequencies : array (num_freq,)
            Microwave frequency axis in GHz
        signals : array (num_experiments, num_freq, num_measurements)
            ODMR signal data
        currents : array (num_experiments, 3), optional
            Currents applied in each experiment
    """
    num_experiments, num_freq, num_measurements = signals.shape
    Ax, Ay, Az = currents[:, 0], currents[:, 1], currents[:, 2]
    
    # Create figure with sliders
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.22)
    
    # Initial plot (config=0, measurement=0)
    line, = ax.plot(frequencies, signals[0, :, 0], 'b-')
    ax.set_xlabel('Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Signal (a.u.)', fontsize=12)
    ax.set_title('ODMR Spectrum - Config 0, Measurement 0, Currents: Ax=%.2f mA, Ay=%.2f mA, Az=%.2f mA' % (Ax[0], Ay[0], Az[0]))
    ax.grid(True, alpha=0.3)
    
    # Create sliders
    ax_config = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_measurement = plt.axes([0.15, 0.04, 0.7, 0.03])
    
    slider_config = Slider(ax_config, 'Current config', 0, num_experiments - 1, valinit=0, valstep=1)
    slider_measurement = Slider(ax_measurement, 'Measurement', 0, num_measurements - 1, valinit=0, valstep=1)
    
    # Update function
    def update(val):
        config_id = int(slider_config.val)
        measurement_id = int(slider_measurement.val)
        
        # Update plot
        line.set_ydata(signals[config_id, :, measurement_id])
        ax.set_title(f'ODMR Spectrum - Current config {config_id}, Measurement {measurement_id}, Currents: Ax={Ax[config_id]:.2f} mA, Ay={Ay[config_id]:.2f} mA, Az={Az[config_id]:.2f} mA')
        
        # Auto-scale y-axis
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle() # Redraw the canvas
    
    # Connect sliders
    slider_config.on_changed(update)
    slider_measurement.on_changed(update)
    
    plt.show()


if __name__ == "__main__":

    CURRENTS_FILE = "dataset_example/3Dcurrents_sweep_2026-01-26_23h00m53s.csv"
    ESR_FILE = "dataset_example/ESR_2026-01-26_23h00m57sRaw.txt"
    Ax, Ay, Az = load_currents(CURRENTS_FILE)
    frequencies, signals, backgrounds = load_esr_raw(ESR_FILE)

    normalized_signals = normalize_by_background(signals, backgrounds)
    normalized_signals = odmr_contrast(normalized_signals)
    normalized_signals = normalize_per_spectrum(normalized_signals)

    # plot_spectrum(frequencies, signals, np.column_stack((Ax, Ay, Az)))
    plot_spectrum(frequencies, normalized_signals, np.column_stack((Ax, Ay, Az)))