import os
import torch
import numpy as np
import argparse
from pathlib import Path
from utils import load_normalization_stats, denormalize_labels
import models
from dataset import train_val_test_split
import json

'''
Evaluate a trained model on the test set and report metrics.
All metrics are computed on denormalized predictions and labels for interpretability (real units).
'''

def evaluate_model(model_name, model_dir=None, dataset_dir=None, device='cpu', show=True):
    # Paths
    if model_dir is None:
        # Utilise models_trained/{dataset_dir}/{model_name}
        if dataset_dir is None:
            raise FileNotFoundError("No dataset_dir provided for model path resolution.")
        model_dir = Path("models_trained") / Path(dataset_dir) / model_name.lower()
    else:
        model_dir = Path(model_dir)
    model_path = model_dir / f"{model_name.lower()}_best_model.pth"
    log_path = model_dir / f"{model_name.lower()}_train_log.json"
    scaler_path = model_dir / f"{model_name.lower()}_scaler.json"

    # Load config from log
    if log_path.exists():
        with open(log_path, 'r') as f:
            log = json.load(f)
        dataset_dir = log['config']['dataset_dir'] if dataset_dir is None else dataset_dir
    else:
        if dataset_dir is None:
            raise FileNotFoundError("No log file found and no dataset_dir provided.")

    # Correction: n'ajoute 'datasets_pytorch' que si nécessaire
    if dataset_dir is not None:
        if not (dataset_dir.startswith('datasets_pytorch') or os.path.isabs(dataset_dir)):
            dataset_dir = os.path.join("datasets_pytorch", dataset_dir)

    # Load normalization stats
    norm_stats = load_normalization_stats(dataset_dir)
    labels_mean = norm_stats['labels_mean']
    labels_std = norm_stats['labels_std']
    coord_system = norm_stats.get('coordinate_system', 'cartesian')
    label_names = ['Ar', 'theta', 'phi'] if coord_system == 'spherical' else ['Ax', 'Ay', 'Az']

    # Load test set
    train_set, val_set, test_set = train_val_test_split(dataset_dir, multi_config=models.__dict__[model_name].requires_multi_config)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    # Load model
    model_class = models.__dict__[model_name]
    if model_name == 'HybridODMRPredictor':
        # Use_attention from log if available
        use_attention = log['config'].get('use_attention', False) if log_path.exists() else False
        model = model_class(n_freq=201, use_attention=use_attention)
    else:
        model = model_class(n_freq=201, output_dim=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_labels, dim=0)

    # Denormalize
    y_pred_denorm = denormalize_labels(y_pred, labels_mean, labels_std).numpy()
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std).numpy()

    # Metrics
    results = {}
    units = ['A', 'A', 'A'] if coord_system == 'cartesian' else ['A', 'rad', 'rad']
    for i, name in enumerate(label_names):
        mae = float(np.abs(y_pred_denorm[:, i] - y_true_denorm[:, i]).mean())
        mean_abs = float(np.abs(y_true_denorm[:, i]).mean())
        std = float(np.std(y_true_denorm[:, i]))
        if show:
            print(f"Mean Abs = {mean_abs:.3f} {units[i]}, Std = {std:.3f} {units[i]}")
        val_range = float(np.max(y_true_denorm[:, i]) - np.min(y_true_denorm[:, i]))
        rmse = float(np.sqrt(((y_pred_denorm[:, i] - y_true_denorm[:, i]) ** 2).mean()))
        # Normalisations
        nmae_mean = (mae / (mean_abs + 1e-8)) * 100
        nmae_std = (mae / (std + 1e-8)) * 100
        nmae_range = (mae / (val_range + 1e-8)) * 100
        nrmse_mean = (rmse / (mean_abs + 1e-8)) * 100
        nrmse_std = (rmse / (std + 1e-8)) * 100
        nrmse_range = (rmse / (val_range + 1e-8)) * 100
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y_true_denorm[:, i], y_pred_denorm[:, i]))
        results[name] = {
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'NMAE_mean': round(nmae_mean, 3),
            'NMAE_std': round(nmae_std, 3),
            'NMAE_range': round(nmae_range, 3),
            'NRMSE_mean': round(nrmse_mean, 3),
            'NRMSE_std': round(nrmse_std, 3),
            'NRMSE_range': round(nrmse_range, 3),
            'R²': round(r2, 3)
        }

    if show:
        print("\n===== Évaluation du modèle sur le jeu de test ====")
        # Tableau
        units = ['A', 'A', 'A'] if coord_system == 'cartesian' else ['A', 'rad', 'rad']
        header = [f"{name} ({units[i]})" for i, name in enumerate(label_names)]
        metrics = [
            ('MAE', 'MAE (unité)'),
            ('RMSE', 'RMSE (unité)'),
            ('NMAE_mean', 'NMAE (mean)'),
            ('NMAE_std', 'NMAE (std)'),
            ('NMAE_range', 'NMAE (range)'),
            ('NRMSE_mean', 'NRMSE (mean)'),
            ('NRMSE_std', 'NRMSE (std)'),
            ('NRMSE_range', 'NRMSE (range)'),
            ('R²', 'R²'),
        ]
        col_width = max(12, max(len(h) for h in header))
        metric_width = 16
        # Header
        print("| {:<{w}} |".format("Metrics", w=metric_width) + " ".join([f"{h:>{col_width}}" for h in header]) + " |")
        print("|" + "-"*(metric_width+2) + "|" + "|".join(["-"*col_width]*len(header)) + "-|")
        for metric, label in metrics:
            if metric == 'R²':
                row = [f"{results[name][metric]:.3f}" for name in label_names]
            elif metric in ['MAE', 'RMSE']:
                row = [f"{results[name][metric]:.3f} {units[i]}" for i, name in enumerate(label_names)]
            else:
                row = [f"{results[name][metric]:.3f} %" for name in label_names]
            print("| {:<{w}} |".format(label, w=metric_width) + " ".join([f"{val:>{col_width}}" for val in row]) + " |")
        print()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on its test set.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g. ODMR_CNN_Compact, HybridODMRPredictor, MWConfig_CNN, etc.)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to model directory (default: models/{model})')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Dataset directory (default: from log)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    evaluate_model(model_name=args.model, model_dir=args.model_dir, dataset_dir=args.dataset_dir, device=args.device)