import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from simulate_esr_spectrum import ensemble_spectrum, Wstart, Wend, Wnr
from tqdm import tqdm

# Fixed MW parameters
MW_field = [1.0, 1.0]
MW_phase_deg = np.arange(0, 361, 1)
MW_phases = np.deg2rad(MW_phase_deg)

# Fixed magnetic field
B_vec = [0.005, 0.002, 0.0028]

# ODMR frequencies
freqList = np.linspace(Wstart, Wend, Wnr)

# Storage list for peaks
all_peaks_freqs = []  # List of peak frequencies for each phase
all_peaks_vals = []   # List of peak intensities for each phase

# Scan through all phases and compute the spectrum, then find peaks
for phase in tqdm(MW_phases, desc="Scanning phases"):
    spectrum = ensemble_spectrum(B_vec, MW_field, phase, freqList, 0, 0)
    # Invert the spectrum for find_peaks (since ODMR = dips)
    peaks, props = find_peaks(-spectrum, prominence=0.01)
    peak_freqs = freqList[peaks]
    peak_vals = spectrum[peaks]
    all_peaks_freqs.append(peak_freqs)
    all_peaks_vals.append(peak_vals)

# Normalize the number of peaks (max 8)
max_peaks = 8
Nphases = len(MW_phases)
peaks_matrix = np.full((Nphases, max_peaks), np.nan)
vals_matrix = np.full((Nphases, max_peaks), np.nan)
for i, (freqs, vals) in enumerate(zip(all_peaks_freqs, all_peaks_vals)):
    n = min(len(freqs), max_peaks)
    peaks_matrix[i, :n] = freqs[:n]
    vals_matrix[i, :n] = vals[:n]

# Plot each peak fluorescence over phases sweep
for k in range(max_peaks):
    plt.figure(figsize=(8, 5))
    plt.plot(MW_phase_deg, vals_matrix[:, k], label=f"Peak {k+1}")
    plt.xlabel("MW Phase (degrees)")
    plt.ylabel("Fluorescence (contrast)")
    plt.title(f"Fluorescence of ODMR Peak {k+1} vs MW Phase (Ox=Oy=1.0)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Find the phase where each peak is at maximum (peak cancelled)
max_indices = np.nanargmax(vals_matrix, axis=0)
max_phases = MW_phase_deg[max_indices]
for k in range(max_peaks):
    print(f"Peak {k+1}: maximum à phase = {max_phases[k]}° (valeur = {vals_matrix[max_indices[k], k]:.3f})")

# Save the data
np.savez('odmr_peaks_vs_phase.npz', MW_phase_deg=MW_phase_deg, peaks_matrix=peaks_matrix, vals_matrix=vals_matrix)
