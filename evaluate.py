import os
import torch
import numpy as np
import argparse
import models
from pathlib import Path
from torch.utils.data import DataLoader
from train_zone_models import ZoneSubset
from utils import load_normalization_stats, denormalize_labels, compute_zones_for_dataset
from dataset import train_val_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def load_model(model_name, model_dir, dataset_dir, device='cpu'):
    model_class = models.__dict__[model_name]

    if model_name == 'HybridODMRPredictor':
        model = model_class(n_channels=10, n_freq=201, use_attention=False)
    elif model_name == 'ZoneClassifier':
        n_zones = compute_zones_for_dataset(dataset_dir)[0].max() + 1
        model = model_class(n_channels=10, n_freq=201, n_zones=n_zones)
    elif model_name == 'ZoneAwareRegressor' or model_name == 'ZoneAwareTwoStage' or model_name == 'ZoneAwareTwoStage_joint':
        n_zones = compute_zones_for_dataset(dataset_dir)[0].max() + 1
        model = model_class(n_channels=10, n_freq=201, n_zones=n_zones, zone_emb_dim=32, output_dim=3)
    else:  # standard CNN regressor
        model = model_class(n_channels=10, n_freq=201, output_dim=3)

    if model_name == 'ZoneAwareTwoStage' and 'zoneawaretwostage_joint' in str(model_dir):
        state_path = Path(model_dir) / "zoneawaretwostage_joint_best_model.pth"
    else:
        state_path = Path(model_dir) / f"{model_name.lower()}_best_model.pth"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_test_loader(dataset_dir, model_type, batch_size=64):
    if 'zone' in model_type.lower():
        zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_dir)
        train_set, val_set, test_set = train_val_test_split(dataset_dir)
        if 'classifier' in model_type.lower():
            test_set = ZoneSubset(test_set, zones, regression=False)
        else:
            test_set = ZoneSubset(test_set, zones)
    else:
        norm_stats = load_normalization_stats(dataset_dir)
        labels_mean = norm_stats['labels_mean']
        labels_std = norm_stats['labels_std']
        train_set, val_set, test_set = train_val_test_split(dataset_dir)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader, labels_mean, labels_std


def evaluate_cnn(model, test_loader, device='cpu'):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            preds = model(signals)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    return y_pred, y_true


def evaluate_regressor(model, test_loader, device='cpu'):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            signals, labels, zones = data
            signals, labels, zones = signals.to(device), labels.to(device), zones.to(device)
            preds = model(signals, zones)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    return y_pred, y_true, zones


def evaluate_classifier(model, test_loader, device='cpu'):
    correct, total = 0, 0
    with torch.no_grad():
        for signals, zones in test_loader:
            signals, zones = signals.to(device), zones.to(device)
            logits = model(signals)
            preds = logits.argmax(dim=1)
            correct += (preds == zones).sum().item()
            total += zones.size(0)
    accuracy = correct / total
    print(f"[Classifier] Test Accuracy: {accuracy*100:.2f}%")
    return accuracy


def evaluate_two_stage(model, test_loader, device='cpu'):
    all_preds, all_labels, all_zones = [], [], []
    with torch.no_grad():
        for signals, labels, zones_true in test_loader:
            signals, labels, zones_true = signals.to(device), labels.to(device), zones_true.to(device)
            logits = model.forward_classifier(signals)
            zones_pred = logits.argmax(dim=1)
            preds = model.forward_regressor(signals, zones_pred)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_zones.append(zones_true.cpu())
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    zones_cat = torch.cat(all_zones, dim=0)
    return y_pred, y_true, zones_cat


def compute_metrics(y_pred, y_true, labels_mean, labels_std):
    y_pred_denorm = denormalize_labels(y_pred, labels_mean, labels_std).numpy()
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std).numpy()
    units = ['A']*3
    label_names = ['Ax', 'Ay', 'Az']
    results = {}
    for i, name in enumerate(label_names):
        mae = float(np.abs(y_pred_denorm[:, i] - y_true_denorm[:, i]).mean())
        mean_abs = float(np.abs(y_true_denorm[:, i]).mean())
        std = float(np.std(y_true_denorm[:, i]))
        val_range = float(np.max(y_true_denorm[:, i]) - np.min(y_true_denorm[:, i]))
        rmse = float(np.sqrt(((y_pred_denorm[:, i] - y_true_denorm[:, i])**2).mean()))
        results[name] = {
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'NMAE_mean': round(mae / (mean_abs + 1e-8) * 100, 3),
            'NMAE_std': round(mae / (std + 1e-8) * 100, 3),
            'NMAE_range': round(mae / (val_range + 1e-8) * 100, 3),
            'NRMSE_mean': round(rmse / (mean_abs + 1e-8) * 100, 3),
            'NRMSE_std': round(rmse / (std + 1e-8) * 100, 3),
            'NRMSE_range': round(rmse / (val_range + 1e-8) * 100, 3),
            'R²': round(float(r2_score(y_true_denorm[:, i], y_pred_denorm[:, i])), 3)
        }
    mae_mean = np.mean([results[name]['MAE'] for name in label_names])

    print("\n===== Evaluation Results (on the test set) =====")
    print(f"MAE mean: {mae_mean:.3f} {units[0]}\n")
    header = [f"{name} ({units[i]})" for i, name in enumerate(label_names)]
    metric_width = 16
    col_width = max(12, max(len(h) for h in header))
    metrics = [
        ('MAE', 'MAE (unit)'), ('RMSE', 'RMSE (unit)'),
        ('NMAE_mean', 'NMAE (mean)'), ('NMAE_std', 'NMAE (std)'), ('NMAE_range', 'NMAE (range)'),
        ('NRMSE_mean', 'NRMSE (mean)'), ('NRMSE_std', 'NRMSE (std)'), ('NRMSE_range', 'NRMSE (range)'),
        ('R²', 'R²')
    ]
    print("| {:<{w}} |".format("Metrics", w=metric_width) + " ".join([f"{h:>{col_width}}" for h in header]) + " |")
    print("|" + "-"*(metric_width+2) + "|" + "|".join(["-"*col_width]*len(header)) + "-|")
    for metric, label in metrics:
        row = [f"{results[name][metric]:.3f} {'%' if 'N' in metric else units[i]}" for i, name in enumerate(label_names)]
        print("| {:<{w}} |".format(label, w=metric_width) + " ".join([f"{val:>{col_width}}" for val in row]) + " |")
    print()
    return results


def plot_test_precision(y_pred, y_true, labels_mean, labels_std, save_path=None):
    """
    Visualize test prediction precision: histograms and scatter plots of errors per axis.
    """
    import matplotlib.pyplot as plt
    y_pred_denorm = denormalize_labels(y_pred, labels_mean, labels_std).numpy()
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std).numpy()
    label_names = ['Ax', 'Ay', 'Az']
    units = ['A', 'A', 'A']
    errors = y_pred_denorm - y_true_denorm

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Test Prediction Precision', fontsize=16, fontweight='bold')
    for i, name in enumerate(label_names):
        # Histogram of errors
        ax = axes[0, i]
        ax.hist(errors[:, i], bins=40, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Error Histogram: {name} ({units[i]})')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

        # Scatter plot: true vs predicted
        ax = axes[1, i]
        ax.scatter(y_true_denorm[:, i], y_pred_denorm[:, i], s=10, alpha=0.6, c='orange')
        ax.plot([y_true_denorm[:, i].min(), y_true_denorm[:, i].max()],
                [y_true_denorm[:, i].min(), y_true_denorm[:, i].max()], 'k--', lw=2)
        ax.set_title(f'True vs Predicted: {name} ({units[i]})')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test precision plot saved to {save_path}")
    else:
        plt.show()

def plot_zone_precision(y_pred, y_true, zones, labels_mean, labels_std, save_path=None):
    """
    Visualize prediction errors as a heatmap: mean error per zone and axis.
    """
    y_pred_denorm = denormalize_labels(y_pred, labels_mean, labels_std).numpy()
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std).numpy()
    label_names = ['Ax', 'Ay', 'Az']
    units = ['A', 'A', 'A']
    errors = np.abs(y_pred_denorm - y_true_denorm)
    # Ensure zones is a numpy array and matches test set length
    zones_np = zones.cpu().numpy() if hasattr(zones, 'cpu') else np.array(zones)
    if len(zones_np.shape) > 1:
        zones_np = zones_np.squeeze()
    # If length mismatch, use only the last len(errors) elements
    if len(zones_np) != len(errors):
        zones_np = zones_np[-len(errors):]
    n_zones = int(np.max(zones_np)) + 1

    # Compute mean error per zone and axis
    mean_error = np.zeros((n_zones, len(label_names)))
    for z in range(n_zones):
        for i in range(len(label_names)):
            zone_mask = (zones_np == z)
            zone_err = errors[zone_mask, i]
            mean_error[z, i] = np.mean(zone_err) if len(zone_err) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mean_error, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title('Mean Absolute Error by Zone and Axis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Axis')
    ax.set_ylabel('Zone Index')
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_xticklabels([f'{name} ({units[i]})' for i, name in enumerate(label_names)])
    ax.set_yticks(np.arange(n_zones))
    ax.set_yticklabels([str(z) for z in range(n_zones)])
    cbar = fig.colorbar(im, ax=ax, label='Mean Absolute Error')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zone precision heatmap saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on its test set.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to evaluate (e.g., "ODMRPredictor", "ZoneAwareRegressor", "ZoneClassifier", "ZoneAwareTwoStage")')
    parser.add_argument('--dataset_dir', type=str, default="dataset_multi_mw_2")
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    # Special handling for joint-trained two-stage model
    if args.model.lower() == 'zoneawaretwostage_joint':
        model_dir = Path("models_trained") / Path(args.dataset_dir) / 'zoneawaretwostage_joint'
        model_name = 'ZoneAwareTwoStage'
    else:
        model_dir = Path("models_trained") / Path(args.dataset_dir) / args.model.lower()
        model_name = args.model

    # Add 'datasets_pytorch' if necessary
    if args.dataset_dir is not None:
        if not (args.dataset_dir.startswith('datasets_pytorch') or os.path.isabs(args.dataset_dir)):
            args.dataset_dir = os.path.join("datasets_pytorch", args.dataset_dir)

    test_loader, labels_mean, labels_std = get_test_loader(args.dataset_dir, model_name, batch_size=64)
    model = load_model(model_name, model_dir, args.dataset_dir, device=device)
    if args.model.lower() == 'zoneclassifier':
        evaluate_classifier(model, test_loader, device=device)
        return
    elif args.model.lower() == 'zoneawareregressor':
        y_pred, y_true, zones = evaluate_regressor(model, test_loader, device=device)
        plot_zone_precision(y_pred, y_true, zones, labels_mean, labels_std)
    elif args.model.lower() == 'zoneawaretwostage' or args.model.lower() == 'zoneawaretwostage_joint':
        y_pred, y_true, zones = evaluate_two_stage(model, test_loader, device=device)
        plot_zone_precision(y_pred, y_true, zones, labels_mean, labels_std)
    else:
        y_pred, y_true = evaluate_cnn(model, test_loader, device=device)
    
    compute_metrics(y_pred, y_true, labels_mean, labels_std)
    plot_test_precision(y_pred, y_true, labels_mean, labels_std)


if __name__ == '__main__':
    main()