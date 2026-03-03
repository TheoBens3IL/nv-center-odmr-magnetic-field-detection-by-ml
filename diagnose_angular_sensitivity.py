"""
Diagnose why theta and phi are not predictable.
Visualize how ODMR signals vary with angles for fixed Ar.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from scipy.stats import pearsonr

def analyze_angular_sensitivity(dataset_dir="dataset_multi_mw_spherical"):
    """
    Check if ODMR signals actually change when theta/phi change but Ar is constant.
    This tests if the elliptical MW configs are sensitive to direction.
    """
    
    dataset_dir = os.path.join("datasets_pytorch", dataset_dir)
    
    # Load data
    metadata = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    signals_dir = Path(dataset_dir) / "signals"
    
    # Load normalization stats
    stats = np.load(Path(dataset_dir) / 'normalization_stats.npy', allow_pickle=True).item()
    from utils import denormalize_labels
    
    # Denormalize labels
    labels_norm = np.stack([
        metadata['Ar'].values,
        metadata['theta'].values,
        metadata['phi'].values
    ], axis=1)
    labels_denorm = denormalize_labels(labels_norm, stats['labels_mean'], stats['labels_std'])
    
    Ar = labels_denorm[:, 0]
    theta = labels_denorm[:, 1]
    phi = labels_denorm[:, 2]
    
    print("="*60)
    print("ANGULAR SENSITIVITY DIAGNOSIS")
    print("="*60)
    
    # Find experiments with similar Ar but different angles
    # This isolates angular variation
    
    ar_target = np.median(Ar)  # Use median Ar
    ar_tolerance = 0.05  # ±0.05 A
    
    mask = np.abs(Ar - ar_target) < ar_tolerance
    print(f"\nSearching for experiments with Ar ≈ {ar_target:.3f} A (±{ar_tolerance} A)")
    print(f"Found {mask.sum()} experiments in this range")
    
    if mask.sum() < 10:
        print("WARNING: Not enough samples to analyze angular sensitivity!")
        print(f"Expanding tolerance to ±0.1 A...")
        ar_tolerance = 0.1
        mask = np.abs(Ar - ar_target) < ar_tolerance
        print(f"Found {mask.sum()} experiments")
    
    # Get indices
    indices = np.where(mask)[0]
    
    # Load signals for these experiments
    signals_subset = []
    theta_subset = []
    phi_subset = []
    
    for idx in indices:
        signal_path = signals_dir / f'config_{idx:04d}.npy'
        signal = np.load(signal_path)  # (10, 201)
        signals_subset.append(signal)
        theta_subset.append(theta[idx])
        phi_subset.append(phi[idx])
    
    signals_subset = np.array(signals_subset)  # (N, 10, 201)
    theta_subset = np.array(theta_subset)
    phi_subset = np.array(phi_subset)
    
    print(f"\nSubset stats:")
    print(f"  Ar: {ar_target:.3f} ± {ar_tolerance:.3f} A (controlled)")
    print(f"  theta: [{theta_subset.min():.3f}, {theta_subset.max():.3f}] rad (range: {theta_subset.max()-theta_subset.min():.3f})")
    print(f"  phi: [{phi_subset.min():.3f}, {phi_subset.max():.3f}] rad (range: {phi_subset.max()-phi_subset.min():.3f})")
    
    # Compute signal variance for each MW config and frequency
    # If variance is high, it means signals change with angles (good!)
    # If variance is low, angles don't affect signals (bad!)
    
    print("\n" + "="*60)
    print("SIGNAL VARIANCE ANALYSIS (for fixed Ar)")
    print("="*60)
    print("\nHigh variance = signals change with theta/phi (good for prediction)")
    print("Low variance = signals insensitive to angles (bad for prediction)")
    
    for mw_idx in range(10):
        signals_mw = signals_subset[:, mw_idx, :]  # (N, 201)
        
        # Variance across experiments (for each frequency)
        variance = signals_mw.var(axis=0)  # (201,)
        mean_variance = variance.mean()
        max_variance = variance.max()
        
        print(f"\nMW Config {mw_idx}:")
        print(f"  Mean variance: {mean_variance:.6f}")
        print(f"  Max variance: {max_variance:.6f}")
        
        # Correlation of signals with theta/phi
        # Average signal across frequencies
        avg_signal_per_exp = signals_mw.mean(axis=1)  # (N,)
        
        corr_theta, _ = pearsonr(avg_signal_per_exp, theta_subset)
        corr_phi, _ = pearsonr(avg_signal_per_exp, phi_subset)
        
        print(f"  Corr with theta: r={corr_theta:+.4f}")
        print(f"  Corr with phi: r={corr_phi:+.4f}")
    
    # Visualization: Pick a few experiments with different angles
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle(f'ODMR Signals for Fixed Ar ≈ {ar_target:.2f} A, Varying Angles', 
                 fontsize=14, fontweight='bold')
    
    # Sort by theta to show progression
    sorted_indices = np.argsort(theta_subset)
    
    # Pick 5 evenly spaced examples
    n_examples = min(5, len(sorted_indices))
    example_indices = sorted_indices[np.linspace(0, len(sorted_indices)-1, n_examples, dtype=int)]
    
    frequencies = np.load(Path(dataset_dir) / 'frequencies.npy') / 1e9  # GHz
    
    for plot_idx, exp_idx in enumerate(example_indices):
        # Top row: first 5 MW configs
        for mw_idx in range(5):
            ax = axes[0, mw_idx] if plot_idx == 0 else axes[0, mw_idx]
            signal = signals_subset[exp_idx, mw_idx, :]
            
            label = f'θ={theta_subset[exp_idx]:.2f}, φ={phi_subset[exp_idx]:.2f}'
            ax.plot(frequencies, signal, alpha=0.7, label=label)
            
            if plot_idx == 0:
                ax.set_title(f'MW Config {mw_idx}')
                ax.set_xlabel('Frequency (GHz)')
                ax.set_ylabel('ODMR Signal')
                ax.grid(True, alpha=0.3)
        
        # Bottom row: last 5 MW configs
        for mw_idx in range(5, 10):
            ax = axes[1, mw_idx-5]
            signal = signals_subset[exp_idx, mw_idx, :]
            
            label = f'θ={theta_subset[exp_idx]:.2f}, φ={phi_subset[exp_idx]:.2f}'
            ax.plot(frequencies, signal, alpha=0.7, label=label)
            
            if plot_idx == 0:
                ax.set_title(f'MW Config {mw_idx}')
                ax.set_xlabel('Frequency (GHz)')
                ax.set_ylabel('ODMR Signal')
                ax.grid(True, alpha=0.3)
    
    # Add legends
    for ax in axes.flat:
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig('diagnostic_plots/angular_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print("Visualization saved: diagnostic_plots/angular_sensitivity_analysis.png")
    print(f"{'='*60}")
    
    # Final diagnostic message
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
If the colored lines in the plots overlap completely:
  → Changing theta/phi does NOT change ODMR signals
  → This explains why theta/phi are unpredictable
  → Likely causes:
    1. MW field magnitudes too weak
    2. NV orientation makes angles insensitive
    3. Experimental setup issue (all peaks centered)

If the lines are clearly separated:
  → Signals DO change with angles
  → Problem is with ML model or normalization
  → The information IS there, just need better model

Check the variance values above:
  - Variance < 0.01: angles have almost no effect
  - Variance > 0.1: angles have measurable effect
    """)

if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset_multi_mw_spherical"
    analyze_angular_sensitivity(dataset_dir)
