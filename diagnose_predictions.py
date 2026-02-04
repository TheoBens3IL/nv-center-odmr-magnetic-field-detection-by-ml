"""
Diagnostic script to analyze model predictions and understand why Ay and Az are not improving.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from split_dataset import train_val_test_split
from utils import load_normalization_stats, denormalize_labels
import models

def diagnose_model(model_path, model_name, dataset_dir="dataset_multi_mw"):
    """Analyze predictions from a trained model."""
    
    # Load model
    available_models = {
        'ODMR_CNN': models.ODMR_CNN,
        'ODMR_CNN_Compact': models.ODMR_CNN_Compact,
        'ODMR_CNN_Deep': models.ODMR_CNN_Deep,
        'FrequencyAttention': models.FrequencyAttention,
        'MWConfig_CNN': models.MWConfig_CNN,
    }
    
    model_class = available_models[model_name]
    use_multi_config = model_class.requires_multi_config
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_class(n_freq=201, output_dim=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load validation set
    _, val_set, _ = train_val_test_split(dataset_dir, multi_config=use_multi_config)
    
    # Load normalization stats
    norm_stats = load_normalization_stats(dataset_dir)
    labels_mean = norm_stats['labels_mean']
    labels_std = norm_stats['labels_std']
    
    # Collect predictions and ground truth
    predictions = []
    ground_truth = []
    
    print(f"\nAnalyzing {model_name}...")
    print(f"Model path: {model_path}")
    print(f"Validation samples: {len(val_set)}\n")
    
    with torch.no_grad():
        for i in range(len(val_set)):
            x, y = val_set[i]
            x = x.unsqueeze(0).to(device)
            
            pred = model(x).cpu()
            predictions.append(pred.squeeze().numpy())
            ground_truth.append(y.numpy())
    
    predictions = np.array(predictions)  # (N, 3)
    ground_truth = np.array(ground_truth)  # (N, 3)
    
    # Denormalize
    pred_denorm = denormalize_labels(predictions, labels_mean, labels_std)
    gt_denorm = denormalize_labels(ground_truth, labels_mean, labels_std)
    
    # Statistics on NORMALIZED predictions
    print("=" * 60)
    print("NORMALIZED PREDICTIONS (should vary around 0 with std~1)")
    print("=" * 60)
    for i, name in enumerate(['Ax', 'Ay', 'Az']):
        pred_mean = predictions[:, i].mean()
        pred_std = predictions[:, i].std()
        pred_min = predictions[:, i].min()
        pred_max = predictions[:, i].max()
        
        gt_mean = ground_truth[:, i].mean()
        gt_std = ground_truth[:, i].std()
        
        print(f"\n{name}:")
        print(f"  Predictions: mean={pred_mean:+.4f}, std={pred_std:.4f}, min={pred_min:+.4f}, max={pred_max:+.4f}")
        print(f"  Ground truth: mean={gt_mean:+.4f}, std={gt_std:.4f}")
        print(f"  Variance ratio (pred/gt): {(pred_std/gt_std):.4f}")
        
        # Check if predictions are nearly constant
        if pred_std < 0.1:
            print(f"  ⚠️  WARNING: Predictions almost constant! Model is NOT learning {name}")
    
    # Statistics on DENORMALIZED predictions
    print("\n" + "=" * 60)
    print("DENORMALIZED PREDICTIONS (original scale in Amperes)")
    print("=" * 60)
    for i, name in enumerate(['Ax', 'Ay', 'Az']):
        pred_mean = pred_denorm[:, i].mean()
        pred_std = pred_denorm[:, i].std()
        pred_min = pred_denorm[:, i].min()
        pred_max = pred_denorm[:, i].max()
        
        gt_mean = gt_denorm[:, i].mean()
        gt_std = gt_denorm[:, i].std()
        gt_min = gt_denorm[:, i].min()
        gt_max = gt_denorm[:, i].max()
        
        print(f"\n{name}:")
        print(f"  Predictions: mean={pred_mean:+.4f}A, std={pred_std:.4f}A, range=[{pred_min:+.4f}, {pred_max:+.4f}]A")
        print(f"  Ground truth: mean={gt_mean:+.4f}A, std={gt_std:.4f}A, range=[{gt_min:+.4f}, {gt_max:+.4f}]A")
    
    # Correlation analysis
    print("\n" + "=" * 60)
    print("CORRELATION BETWEEN PREDICTED AND TRUE VALUES")
    print("=" * 60)
    for i, name in enumerate(['Ax', 'Ay', 'Az']):
        corr = np.corrcoef(predictions[:, i], ground_truth[:, i])[0, 1]
        print(f"{name}: r={corr:.4f}")
        if abs(corr) < 0.1:
            print(f"  ⚠️  WARNING: Almost no correlation! Model is NOT predicting {name} correctly")
    
    # Cross-correlation between components
    print("\n" + "=" * 60)
    print("CROSS-CORRELATION IN GROUND TRUTH (are Ay/Az correlated?)")
    print("=" * 60)
    names = ['Ax', 'Ay', 'Az']
    for i in range(3):
        for j in range(i+1, 3):
            corr = np.corrcoef(ground_truth[:, i], ground_truth[:, j])[0, 1]
            print(f"{names[i]} vs {names[j]}: r={corr:.4f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Prediction Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    names = ['Ax', 'Ay', 'Az']
    
    # Row 1: Scatter plots (predicted vs true)
    for i, name in enumerate(names):
        ax = axes[0, i]
        ax.scatter(gt_denorm[:, i], pred_denorm[:, i], alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(gt_denorm[:, i].min(), pred_denorm[:, i].min())
        max_val = max(gt_denorm[:, i].max(), pred_denorm[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel(f'True {name} (A)', fontsize=10)
        ax.set_ylabel(f'Predicted {name} (A)', fontsize=10)
        ax.set_title(f'{name} Predictions vs Truth', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    
    # Row 2: Error distributions
    for i, name in enumerate(names):
        ax = axes[1, i]
        errors = pred_denorm[:, i] - gt_denorm[:, i]
        
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(errors.mean(), color='g', linestyle='--', linewidth=2, label=f'Mean={errors.mean():.4f}A')
        
        ax.set_xlabel(f'Prediction Error (A)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name} Error Distribution', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    save_dir = Path("diagnostic_plots")
    save_dir.mkdir(exist_ok=True)
    fig_path = save_dir / f"diagnosis_{model_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to {fig_path}")
    plt.show()
    
    return predictions, ground_truth, pred_denorm, gt_denorm


if __name__ == "__main__":
    # Find the best models
    model_dir = Path("saved_models")
    
    # List available models
    model_files = sorted(model_dir.glob("*.pt"))
    
    if not model_files:
        print("No trained models found in saved_models/")
        exit(1)
    
    print("Available trained models:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file.name}")
    
    # Analyze the most recent models
    print("\n" + "=" * 80)
    print("ANALYZING MULTI-CONFIG MODELS")
    print("=" * 80)
    
    # Try to find and analyze multi-config models
    multi_config_models = {
        'ODMR_CNN_Deep': None,
        'FrequencyAttention': None,
        'MWConfig_CNN': None,
    }
    
    # Find the best model for each architecture (lowest loss)
    for model_file in model_files:
        for model_name in multi_config_models.keys():
            if model_name.lower() in model_file.name.lower() or \
               (model_name == 'MWConfig_CNN' and 'mwconfig' in model_file.name.lower()):
                # Extract loss from filename
                try:
                    loss_str = model_file.name.split('loss_')[1].replace('.pt', '')
                    loss = float(loss_str)
                    
                    if multi_config_models[model_name] is None or loss < multi_config_models[model_name][1]:
                        multi_config_models[model_name] = (model_file, loss)
                except:
                    pass
    
    # Analyze each multi-config model
    for model_name, model_info in multi_config_models.items():
        if model_info is not None:
            model_path, loss = model_info
            print(f"\n{'=' * 80}")
            diagnose_model(str(model_path), model_name)
        else:
            print(f"\nNo trained model found for {model_name}")
