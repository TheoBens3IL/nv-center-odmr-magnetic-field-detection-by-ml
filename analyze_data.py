import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Analyser les corrélations entre labels et caractéristiques spectrales
metadata = pd.read_csv("pytorch_dataset_example/metadata.csv")

# Extraire des features simples de chaque spectre
features = []
for i in range(len(metadata)):
    sig = np.load(f"pytorch_dataset_example/signals/config_{i:03d}.npy").flatten()
    
    # Features statistiques
    mean_val = sig.mean()
    std_val = sig.std()
    min_val = sig.min()
    max_val = sig.max()
    range_val = max_val - min_val
    
    # Position et amplitude du minimum (dip principal)
    min_idx = sig.argmin()
    
    features.append([mean_val, std_val, min_val, max_val, range_val, min_idx])

features = np.array(features)

# Calculer les corrélations entre features et labels
print("=== CORRELATIONS FEATURES -> LABELS ===")
feature_names = ['mean', 'std', 'min', 'max', 'range', 'min_idx']
label_names = ['Ax', 'Ay', 'Az']

for i, fname in enumerate(feature_names):
    for lname in label_names:
        corr, pval = pearsonr(features[:, i], metadata[lname])
        if abs(corr) > 0.1:
            print(f"{fname:8s} -> {lname}: {corr:+.3f} (p={pval:.3e})")

# Vérifier la variance des labels
print(f"\n=== VARIANCE DES LABELS ===")
for lname in label_names:
    print(f"{lname}: mean={metadata[lname].mean():.4f}, std={metadata[lname].std():.4f}, range={metadata[lname].max()-metadata[lname].min():.4f}")

# Calculer l'erreur si on prédit juste la moyenne
print(f"\n=== BASELINE (prédire la moyenne) ===")
for lname in label_names:
    mean_pred = metadata[lname].mean()
    mae = np.abs(metadata[lname] - mean_pred).mean()
    mse = ((metadata[lname] - mean_pred) ** 2).mean()
    print(f"{lname}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={np.sqrt(mse):.4f}")

total_mse_baseline = sum([((metadata[lname] - metadata[lname].mean()) ** 2).mean() for lname in label_names]) / 3
print(f"\nTotal MSE baseline (predicting mean): {total_mse_baseline:.4f}")
print(f"Current model MSE: ~0.146")
print(f"Improvement over baseline: {(1 - 0.146/total_mse_baseline)*100:.1f}%")
