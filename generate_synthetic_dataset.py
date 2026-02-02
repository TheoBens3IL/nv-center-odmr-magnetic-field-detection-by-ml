import numpy as np
import pandas as pd
from pathlib import Path

# =======================
# Paramètres
# =======================
N_CONFIGS = 5000
N_REPEAT = 25         # 25 signaux par config (comme dataset original)
N_FREQ = 201
FREQ_RANGE = (2.86e9, 2.89e9)
NOISE_STD = 0.01

DATASET_DIR = Path("synthetic_dataset")
SIGNALS_DIR = DATASET_DIR / "signals"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# Fréquences MW communes
# =======================
frequencies = np.linspace(*FREQ_RANGE, N_FREQ).astype(np.float64)
np.save(DATASET_DIR / "frequencies.npy", frequencies)

# =======================
# Physique NV simplifiée
# =======================
def step3_Blab_to_fi_Simu_3state(B):
    D = 2.87e9          # Hz
    gamma = 28e6        # Hz/T (effectif)

    nv_axes = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1]
    ], dtype=float)
    nv_axes /= np.linalg.norm(nv_axes, axis=1)[:, None]

    nu = []  # fréquences de résonance
    amp = [] # amplitudes relatives

    for u in nv_axes:
        Bp = np.dot(B, u)
        nu.append(D + gamma * Bp)
        nu.append(D - gamma * Bp)
        amp.append(1.0)
        amp.append(1.0)

    return np.array(nu), np.array(amp)

def lorentzian(x, x0, gamma):
    return (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2)

def simulate_esr(B, linewidth, seed=None):
    if seed is not None:
        np.random.seed(seed)

    nu, amp = step3_Blab_to_fi_Simu_3state(B)

    spectrum = np.zeros_like(frequencies)

    for n, a in zip(nu, amp):
        spectrum += a * lorentzian(frequencies, n, linewidth)

    spectrum /= spectrum.max()
    fluo = 1.0 - spectrum
    fluo += np.random.normal(0.0, NOISE_STD, size=fluo.shape)

    return fluo.astype(np.float32)

# =======================
# Génération du dataset
# =======================
metadata = []

for i in range(N_CONFIGS):
    config_id = f"config_{i:06d}"

    # Champ magnétique aléatoire (unités arbitraires cohérentes)
    B = np.random.uniform(-0.3, 0.3, size=3)

    signals = []
    for k in range(N_REPEAT):
        linewidth = np.random.uniform(1e6, 5e6)
        signals.append(simulate_esr(B, linewidth, seed=i * 10 + k))

    signals = np.stack(signals)  # (N_REPEAT, 201)
    np.save(SIGNALS_DIR / f"{config_id}.npy", signals)

    metadata.append({
        "config_id": config_id,
        "Bx": B[0],
        "By": B[1],
        "Bz": B[2],
    })

df = pd.DataFrame(metadata)
df.to_csv(DATASET_DIR / "metadata.csv", index=False)

print("Dataset synthétique généré")
print("Example signals shape:", signals.shape)
