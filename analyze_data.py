import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

'''
Analyse linear correlations between features extracted from spectra and the current components (Ax, Ay, Az).
'''

DATASET_DIR = os.path.join("datasets_pytorch", "dataset_multi_mw_2")
metadata = pd.read_csv(f"{DATASET_DIR}/metadata.csv")

# Extraire des features simples de chaque spectre
features = []
all_signals = []  # Store all signals for frequency-wise correlation

for i in range(len(metadata)):
    sig = np.load(f"{DATASET_DIR}/signals/config_{i:04d}.npy")  # Shape: (10, 201)
    # Average over MW configs to get one spectrum per experiment
    sig_avg = sig.mean(axis=0)  # Shape: (201,)
    all_signals.append(sig_avg)
    
    # Features statistiques
    mean_val = sig_avg.mean()
    std_val = sig_avg.std()
    min_val = sig_avg.min()
    max_val = sig_avg.max()
    range_val = max_val - min_val
    
    # Position et amplitude du minimum (dip principal)
    min_idx = sig_avg.argmin()
    
    features.append([mean_val, std_val, min_val, max_val, range_val, min_idx])

features = np.array(features)
all_signals = np.array(all_signals)  # Shape: (num_experiments, 201)

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


# ========================================
# FREQUENCY-WISE CORRELATION ANALYSIS
# ========================================
print("\n" + "="*60)
print("=== FREQUENCY-WISE CORRELATION ANALYSIS ===")
print("="*60)

def feature_label_corr(all_signals, all_labels):
    """
    Compute correlation between each frequency point and each label.
    
    Parameters:
        all_signals : array (num_samples, n_freq)
        all_labels : array (num_samples, 3)
    
    Returns:
        corr_matrix : array (n_freq, 3) - correlation of each freq with Ax, Ay, Az
    """
    n_freq = all_signals.shape[1]
    corr_matrix = np.zeros((n_freq, all_labels.shape[1]))

    for i in range(n_freq):
        for j in range(all_labels.shape[1]):
            corr_matrix[i, j] = np.corrcoef(all_signals[:, i], all_labels[:, j])[0, 1]
    
    return corr_matrix  # shape (n_freq, 3)

# Extract labels
all_labels = metadata[['Ax', 'Ay', 'Az']].values

# Compute correlations
corrs = feature_label_corr(all_signals, all_labels)

print("\nCorrelation min/max per axis:")
for idx, axis in enumerate(['Ax', 'Ay', 'Az']):
    print(f"{axis}: min={corrs[:, idx].min():+.4f}, max={corrs[:, idx].max():+.4f}, mean_abs={np.abs(corrs[:, idx]).mean():.4f}")

# Find most correlated frequencies for each axis
print("\nMost correlated frequencies (top 5) for each axis:")
frequencies = np.load(f"{DATASET_DIR}/frequencies.npy") / 1e9  # Convert to GHz

for idx, axis in enumerate(['Ax', 'Ay', 'Az']):
    abs_corrs = np.abs(corrs[:, idx])
    top_indices = np.argsort(abs_corrs)[-5:][::-1]
    
    print(f"\n{axis}:")
    for i, freq_idx in enumerate(top_indices, 1):
        freq_ghz = frequencies[freq_idx]
        corr_val = corrs[freq_idx, idx]
        print(f"  {i}. Freq {freq_ghz:.3f} GHz (idx {freq_idx}): corr = {corr_val:+.4f}")

# Plot correlation heatmap
print("\nGenerating correlation heatmap...")
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Frequency-wise Correlation with Current Components', fontsize=14, fontweight='bold')

for idx, (ax, axis) in enumerate(zip(axes, ['Ax', 'Ay', 'Az'])):
    ax.plot(frequencies, corrs[:, idx], linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel(f'Correlation\nwith {axis}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)
    
    # Highlight most correlated region
    max_corr_idx = np.argmax(np.abs(corrs[:, idx]))
    ax.axvline(x=frequencies[max_corr_idx], color='r', linestyle=':', alpha=0.5, 
               label=f'Max |corr| @ {frequencies[max_corr_idx]:.3f} GHz')
    ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Frequency (GHz)', fontsize=11)
plt.tight_layout()
print("Saved plot to: frequency_correlation_analysis.png")
plt.show()
