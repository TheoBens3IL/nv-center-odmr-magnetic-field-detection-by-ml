"""
Analyze the raw dataset to understand if Ay and Az can be predicted from ODMR spectra.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import pearsonr

def analyze_dataset(dataset_dir="dataset_multi_mw"):
    """Analyze the multi-MW dataset for predictability of each component."""
    
    print("=" * 60)
    print("DATASET ANALYSIS: Predictability of Ax, Ay, Az")
    print("=" * 60)
    
    # Load metadata
    metadata = pd.read_csv(Path(dataset_dir) / "metadata.csv")
    signals_dir = Path(dataset_dir) / "signals"
    
    print(f"\nTotal experiments (different current configurations): {len(metadata)}")
    
    # Load labels
    Ax = metadata['Ax'].values
    Ay = metadata['Ay'].values
    Az = metadata['Az'].values
    
    # Statistics on labels
    print("\n" + "=" * 60)
    print("LABEL STATISTICS (Original scale)")
    print("=" * 60)
    for name, data in [('Ax', Ax), ('Ay', Ay), ('Az', Az)]:
        print(f"\n{name}:")
        print(f"  Mean: {data.mean():+.4f} A")
        print(f"  Std:  {data.std():.4f} A")
        print(f"  Min:  {data.min():+.4f} A")
        print(f"  Max:  {data.max():+.4f} A")
        print(f"  Range: {data.max() - data.min():.4f} A")
    
    # Label correlations
    print("\n" + "=" * 60)
    print("LABEL CORRELATIONS (are they independent?)")
    print("=" * 60)
    corr_ax_ay, _ = pearsonr(Ax, Ay)
    corr_ax_az, _ = pearsonr(Ax, Az)
    corr_ay_az, _ = pearsonr(Ay, Az)
    
    print(f"Ax vs Ay: r = {corr_ax_ay:+.4f}")
    print(f"Ax vs Az: r = {corr_ax_az:+.4f}")
    print(f"Ay vs Az: r = {corr_ay_az:+.4f}")
    
    # ==================================================================================
    # ANALYSIS 1: Individual signals (as used in ODMRDataset - single-config models)
    # ==================================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 1: INDIVIDUAL SIGNALS (ODMRDataset approach)")
    print("=" * 60)
    print("Analyzing all 21,090 individual spectra (2109 experiments × 10 MW configs)")
    print("This is how single-config models (ODMR_CNN, ODMR_CNN_Compact) see the data")
    
    n_freq = 201
    n_mw_configs = 10
    
    # Flatten all signals: treat each MW config as independent sample
    all_individual_signals = []
    all_individual_labels = {'Ax': [], 'Ay': [], 'Az': []}
    
    for idx in range(len(metadata)):
        config_id = int(metadata.iloc[idx]['experiment_id'])
        signals = np.load(signals_dir / f"config_{config_id:04d}.npy")  # (10, 201)
        
        # Each of 10 MW configs becomes a separate sample
        for mw_idx in range(10):
            all_individual_signals.append(signals[mw_idx, :])
            all_individual_labels['Ax'].append(Ax[idx])
            all_individual_labels['Ay'].append(Ay[idx])
            all_individual_labels['Az'].append(Az[idx])
    
    all_individual_signals = np.array(all_individual_signals)  # (21090, 201)
    
    print(f"\nTotal individual samples: {len(all_individual_signals)}")
    
    # Compute correlations for pooled data
    max_corr_pooled = np.zeros(3)
    for label_idx, (name, label_data) in enumerate([('Ax', all_individual_labels['Ax']), 
                                                    ('Ay', all_individual_labels['Ay']), 
                                                    ('Az', all_individual_labels['Az'])]):
        corrs = []
        for freq_idx in range(n_freq):
            corr, _ = pearsonr(all_individual_signals[:, freq_idx], label_data)
            corrs.append(abs(corr))
        max_corr_pooled[label_idx] = max(corrs)
    
    print(f"\nMaximum |correlation| when pooling all MW configs together:")
    print(f"  Ax: |r| = {max_corr_pooled[0]:.4f}")
    print(f"  Ay: |r| = {max_corr_pooled[1]:.4f}")
    print(f"  Az: |r| = {max_corr_pooled[2]:.4f}")
    
    # ==================================================================================
    # ANALYSIS 2: Per-MW-config (as used in ODMRDatasetMultiConfig - multi-config models)
    # ==================================================================================
    
    # Per-MW-config analysis (the proper way to analyze)
    print("\n" + "=" * 60)
    print("SIGNAL-LABEL CORRELATIONS PER MW CONFIGURATION")
    print("=" * 60)
    
    n_freq = 201
    n_mw_configs = 10
    
    # Store results for each MW config
    max_corr_per_config = np.zeros((n_mw_configs, 3))  # (10, 3)
    best_freq_per_config = np.zeros((n_mw_configs, 3), dtype=int)  # (10, 3)
    
    for mw_idx in range(n_mw_configs):
        # Extract signals for this MW config
        mw_signals = []
        for idx in range(len(metadata)):
            config_id = int(metadata.iloc[idx]['experiment_id'])
            signals = np.load(signals_dir / f"config_{config_id:04d}.npy")
            mw_signals.append(signals[mw_idx, :])
        
        mw_signals = np.array(mw_signals)  # (n_experiments, 201)
        
        # Compute max correlation for each label
        for label_idx, (name, label_data) in enumerate([('Ax', Ax), ('Ay', Ay), ('Az', Az)]):
            corrs = []
            for freq_idx in range(n_freq):
                corr, _ = pearsonr(mw_signals[:, freq_idx], label_data)
                corrs.append(abs(corr))
            max_corr_per_config[mw_idx, label_idx] = max(corrs)
            best_freq_per_config[mw_idx, label_idx] = np.argmax(corrs)
    
    print(f"\nMaximum |correlation| per MW configuration:")
    print(f"{'MW Config':<12} {'Ax':<10} {'Ay':<10} {'Az':<10}")
    print("-" * 42)
    for mw_idx in range(n_mw_configs):
        print(f"Config {mw_idx:<5} {max_corr_per_config[mw_idx, 0]:.4f}     "
              f"{max_corr_per_config[mw_idx, 1]:.4f}     {max_corr_per_config[mw_idx, 2]:.4f}")
    
    print("\nBest MW config for each component:")
    for label_idx, name in enumerate(['Ax', 'Ay', 'Az']):
        best_config = max_corr_per_config[:, label_idx].argmax()
        best_corr = max_corr_per_config[:, label_idx].max()
        print(f"  {name}: Config {best_config} (|r|={best_corr:.4f})")
        
        if best_corr < 0.1:
            print(f"    WARNING: Even best config has very weak correlation!")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Analysis - Predictability Assessment', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    x_pos = np.arange(n_mw_configs)
    width = 0.25
    
    ax.bar(x_pos - width, max_corr_per_config[:, 0], width, label='Ax', alpha=0.8, color='#1f77b4')
    ax.bar(x_pos, max_corr_per_config[:, 1], width, label='Ay', alpha=0.8, color='#ff7f0e')
    ax.bar(x_pos + width, max_corr_per_config[:, 2], width, label='Az', alpha=0.8, color='#2ca02c')
    
    ax.set_xlabel('MW Configuration')
    ax.set_ylabel('Max |Correlation|')
    ax.set_title('Maximum Signal-Label Correlation per MW Config')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{i}' for i in range(n_mw_configs)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5)
    ax.set_ylim([0, max(0.5, max_corr_per_config.max() * 1.1)])
    
    # Plot 2: Heatmap of correlations (MW config × component)
    ax = axes[0, 1]
    im = ax.imshow(max_corr_per_config, aspect='auto', cmap='RdBu_r', vmin=0, vmax=0.5)
    ax.set_xlabel('Component')
    ax.set_ylabel('MW Configuration')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Ax', 'Ay', 'Az'])
    ax.set_yticks(range(n_mw_configs))
    ax.set_yticklabels([f'Config {i}' for i in range(n_mw_configs)])
    ax.set_title('Correlation Heatmap: Which configs predict which component?')
    
    # Add text annotations
    for i in range(n_mw_configs):
        for j in range(3):
            text = ax.text(j, i, f'{max_corr_per_config[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='|Correlation|')
    
    # Plot 3: Label distributions (useful to see if data is balanced)
    ax = axes[1, 0]
    ax.hist(Ax, bins=30, alpha=0.7, label='Ax', edgecolor='black', density=True)
    ax.hist(Ay, bins=30, alpha=0.7, label='Ay', edgecolor='black', density=True)
    ax.hist(Az, bins=30, alpha=0.7, label='Az', edgecolor='black', density=True)
    ax.set_xlabel('Current (normalized units)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Label Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary - Overall predictability
    ax = axes[1, 1]
    overall_max = max_corr_per_config.max(axis=0)  # Best correlation across all configs
    overall_mean = max_corr_per_config.mean(axis=0)  # Average across configs
    
    x_pos = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, overall_max, width, label='Best config', 
                   alpha=0.8, edgecolor='black', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    bars2 = ax.bar(x_pos + width/2, overall_mean, width, label='Average config',
                   alpha=0.6, edgecolor='black', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_ylabel('Max |Correlation|')
    ax.set_title('Overall Predictability Summary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Ax', 'Ay', 'Az'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(2.5, 0.32, 'Good threshold', fontsize=9, color='red')
    ax.set_ylim([0, 0.6])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_dir = Path("diagnostic_plots")
    save_dir.mkdir(exist_ok=True)
    fig_path = save_dir / "dataset_predictability.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {fig_path}")
    plt.show()
    
    return max_corr_per_config


if __name__ == "__main__":
    analyze_dataset("dataset_multi_mw")
