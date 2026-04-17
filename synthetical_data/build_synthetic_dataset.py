import numpy as np
import pandas as pd
import os
from simulate_esr_spectrum_optimized import ensemble_spectrum, Wstart, Wend, Wnr


# --- MW configurations demandées ---
mw_configs = [
    {"MW_field": [1.0, 0.0, 0.0], "MW_phase": np.deg2rad(90), "name": "mw1"},
    {"MW_field": [1.0, 1.0, 0.0], "MW_phase": np.deg2rad(30), "name": "mw2"},
    {"MW_field": [1.0, 1.0, 0.0], "MW_phase": np.deg2rad(150), "name": "mw3"},
    {"MW_field": [1.0, 1.0, 0.0], "MW_phase": np.deg2rad(210), "name": "mw4"},
    {"MW_field": [1.0, 1.0, 0.0], "MW_phase": np.deg2rad(330), "name": "mw5"},
]
freq_list = np.linspace(Wstart, Wend, Wnr)
output_dir = "datasets_synthetic/synthetic_multi_mw_2"
os.makedirs(output_dir, exist_ok=True)

# --- Load labels (Ax, Ay, Az) from real dataset ---
meta_path = "datasets_pytorch/dataset_multi_mw_2/metadata.csv"
meta = pd.read_csv(meta_path)


# --- Build synthetic dataset for each MW config ---
for mw in mw_configs:
    signals_dir = os.path.join(output_dir, f"signals_{mw['name']}")
    os.makedirs(signals_dir, exist_ok=True)
    print(f"Génération pour MW_field={mw['MW_field']}, phase={mw['MW_phase']:.2f} rad ({mw['name']})...")
    all_signals = []
    for idx, row in meta.iterrows():
        B_vec = [row['Ax'], row['Ay'], row['Az']]
        spectrum = ensemble_spectrum(B_vec, mw['MW_field'], mw['MW_phase'], freq_list)
        np.save(os.path.join(signals_dir, f"config_{idx:04d}.npy"), spectrum.astype(np.float32))
        all_signals.append(spectrum)
    print(f"  -> {len(meta)} spectres générés dans {signals_dir}")

# Save frequencies and metadata (communs à toutes les configs)
np.save(os.path.join(output_dir, "frequencies.npy"), freq_list.astype(np.float32))
meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

print(f"Synthetic dataset created in {output_dir} (5 configs MW)")
