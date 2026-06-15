import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
from simulate_esr_spectrum import ensemble_spectrum, Wstart, Wend, Wnr

# MW configs
mw_configs = [
    {"MW_field": [1.0, 0.0, 0.0], "MW_phase": np.deg2rad(90), "name": "mw1"},
    {"MW_field": [1.0, 1.0, 0.0], "MW_phase": np.deg2rad(30), "name": "mw2"},
    {"MW_field": [1.0, 1.0, 0.0], "MW_phase": np.deg2rad(150), "name": "mw3"},
    {"MW_field": [1.0, 1.0, 0.0], "MW_phase": np.deg2rad(210), "name": "mw4"},
    {"MW_field": [1.0, 1.0, 0.0], "MW_phase": np.deg2rad(330), "name": "mw5"},
]

# Tilt angles to explore (in degrees)
tilt_x_deg = np.arange(5, 21, 1)
tilt_y_deg = np.arange(5, 21, 1)

# Fixed magnetic field
B_vec = [0.005, 0.002, 0.0028]
freqList = np.linspace(Wstart, Wend, Wnr)

for mw in mw_configs:
    print(f"Exploration tilt pour {mw['name']}...")
    # Store results : (tilt_x, tilt_y, n_peaks, max_val, idx_peak_max)
    results = []
    for tx in tqdm(tilt_x_deg, desc=f"tilt_x {mw['name']}"):
        for ty in tilt_y_deg:
            tilt_x = np.deg2rad(tx)
            tilt_y = np.deg2rad(ty)
            spectrum = ensemble_spectrum(B_vec, mw['MW_field'], mw['MW_phase'], freqList, tilt_x, tilt_y)
            peaks, _ = find_peaks(-spectrum, prominence=0.01)
            peak_vals = spectrum[peaks]
            if len(peak_vals) > 0:
                idx_max = np.argmax(peak_vals)
                max_val = peak_vals[idx_max]
                results.append({
                    'tilt_x_deg': tx,
                    'tilt_y_deg': ty,
                    'n_peaks': len(peak_vals),
                    'max_val': max_val,
                    'idx_peak_max': idx_max,
                    'peak_vals': peak_vals
                })
    # Analyse : for each config, search if max_val > 0.95 and n_peaks >= 6 (indicating a good spectrum with many peaks and a strong main peak)
    # Visualisation : heatmap of max_val vs tilt_x and tilt_y
    heatmap = np.zeros((len(tilt_x_deg), len(tilt_y_deg)))
    for r in results:
        ix = r['tilt_x_deg'] - tilt_x_deg[0]
        iy = r['tilt_y_deg'] - tilt_y_deg[0]
        heatmap[ix, iy] = r['max_val']
    plt.figure(figsize=(8,6))
    plt.imshow(heatmap, origin='lower', extent=[tilt_y_deg[0], tilt_y_deg[-1], tilt_x_deg[0], tilt_x_deg[-1]], aspect='auto')
    plt.colorbar(label='Max fluorescence d\'un pic')
    plt.xlabel('tilt_y (deg)')
    plt.ylabel('tilt_x (deg)')
    plt.title(f"Max fluorescence d'un pic vs tilt (MW: {mw['name']})")
    plt.tight_layout()
    plt.savefig(f"tilt_scan_heatmap_{mw['name']}.png")
    plt.show()
    # Optionnal : print configurations with good spectra
    for r in results:
        if r['max_val'] > 0.95 and r['n_peaks'] >= 6:
            print(f"{mw['name']} : tilt_x={r['tilt_x_deg']}°, tilt_y={r['tilt_y_deg']}° -> max_val={r['max_val']:.2f}, n_peaks={r['n_peaks']}")
