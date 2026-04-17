import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from simulate_esr_spectrum_optimized import ensemble_spectrum, Wstart, Wend, Wnr
from tqdm import tqdm

# Paramètres MW fixes
MW_field = [1.0, 1.0]  # Ox = Oy = 1.0
MW_phase_deg = np.arange(0, 361, 1)  # 0 à 360 degrés
MW_phases = np.deg2rad(MW_phase_deg)

# Paramètres du champ magnétique (exemple)
B_vec = [0.005, 0.002, 0.0028]

# Fréquences ODMR
freqList = np.linspace(Wstart, Wend, Wnr)

# Stockage des pics
all_peaks_freqs = []  # Liste des fréquences des pics pour chaque phase
all_peaks_vals = []   # Liste des intensités des pics pour chaque phase

for phase in tqdm(MW_phases, desc="Scanning phases"):
    spectrum = ensemble_spectrum(B_vec, MW_field, phase, freqList, 0, 0)
    # Inverser le spectre pour find_peaks (car ODMR = creux)
    peaks, props = find_peaks(-spectrum, prominence=0.01)
    peak_freqs = freqList[peaks]
    peak_vals = spectrum[peaks]
    all_peaks_freqs.append(peak_freqs)
    all_peaks_vals.append(peak_vals)

# Normalisation du nombre de pics (max 8)
max_peaks = 8
Nphases = len(MW_phases)
peaks_matrix = np.full((Nphases, max_peaks), np.nan)
vals_matrix = np.full((Nphases, max_peaks), np.nan)
for i, (freqs, vals) in enumerate(zip(all_peaks_freqs, all_peaks_vals)):
    n = min(len(freqs), max_peaks)
    peaks_matrix[i, :n] = freqs[:n]
    vals_matrix[i, :n] = vals[:n]


# Un plot séparé pour chaque pic
for k in range(max_peaks):
    plt.figure(figsize=(8, 5))
    plt.plot(MW_phase_deg, vals_matrix[:, k], label=f"Peak {k+1}")
    plt.xlabel("Phase MW (degrés)")
    plt.ylabel("Fluorescence (contraste)")
    plt.title(f"Fluorescence du pic ODMR {k+1} en fonction de la phase MW (Ox=Oy=1.0)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Trouver la phase où chaque pic est au maximum (pic annulé)
max_indices = np.nanargmax(vals_matrix, axis=0)
max_phases = MW_phase_deg[max_indices]
for k in range(max_peaks):
    print(f"Peak {k+1}: maximum à phase = {max_phases[k]}° (valeur = {vals_matrix[max_indices[k], k]:.3f})")

# Sauvegarde des données
np.savez('odmr_peaks_vs_phase.npz', MW_phase_deg=MW_phase_deg, peaks_matrix=peaks_matrix, vals_matrix=vals_matrix)
