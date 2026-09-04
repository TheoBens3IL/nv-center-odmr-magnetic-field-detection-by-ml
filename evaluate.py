import os
import time
import torch
import numpy as np
import argparse
import models
from pathlib import Path
from torch.utils.data import DataLoader
from train_zone_models import ZoneSubset
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from dataset import train_val_test_split, stratified_zone_split, resolve_mw_indices, detect_num_mw_configs, detect_n_freq, get_frequency_axis
from utils import (load_normalization_stats, denormalize_labels, compute_zones_for_dataset, resolve_model_output_dir, plot_zone_mae_on_sphere, plot_extreme_zone_signals)


def resolve_checkpoint_path(model_dir, model_name):
    """Return checkpoint path, with fallback to legacy filenames/directories."""
    model_dir = Path(model_dir)
    primary = model_dir / f"{model_name.lower()}_best_model.pth"
    if primary.exists():
        return primary
    legacy_ckpts = {
        'ZoneAwareTwoStageJoint': ['zoneawaretwostage_joint_best_model.pth'],
        'ZoneAwareTwoStageJointDeep': ['zoneawaretwostage_joint2_best_model.pth', 'zoneawaretwostage2_best_model.pth'],
    }
    for ckpt in legacy_ckpts.get(model_name, []):
        path = model_dir / ckpt
        if path.exists():
            return path
    return primary


def load_model(model_name, model_dir, dataset_dir, device='cpu', mw_indices=None, synthetic=False):
    models_list = models.available_models()
    if model_name not in models_list:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_list.keys())}")
    model_class = models_list[model_name]
    if mw_indices is not None:
        n_channels = len(mw_indices)
    else:
        n_channels = detect_num_mw_configs(dataset_dir, synthetic=synthetic)
    n_freq = detect_n_freq(dataset_dir, synthetic=synthetic)

    if model_name == 'AxisSplitRegressor':
        model = model_class(n_channels=n_channels, n_freq=n_freq, use_attention=False)
    elif model_name == 'ZoneClassifier' or model_name == 'ZoneClassifier2':
        n_zones = compute_zones_for_dataset(dataset_dir)[0].max() + 1
        model = model_class(n_channels=n_channels, n_freq=n_freq, n_zones=n_zones)
    elif model_name in (
        'ZoneAwareRegressor', 'ZoneAwareTwoStage', 'ZoneAwareTwoStageJoint',
        'ZoneAwareRegressor2', 'ZoneAwareTwoStageJointDeep',
    ):
        n_zones = compute_zones_for_dataset(dataset_dir)[0].max() + 1
        model = model_class(n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, zone_emb_dim=32, output_dim=3)
    else:  # standard CNN regressor
        model = model_class(n_channels=n_channels, n_freq=n_freq, output_dim=3)

    state_path = resolve_checkpoint_path(model_dir, model_name)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_test_loader(dataset_dir, model_type, batch_size=64, mw_indices=None, synthetic=False,
                    val_size=0.10, test_size=0.10, balanced_val=True, val_samples_per_zone=None,
                    balanced_test=False, test_samples_per_zone=1):
    if 'zone' in model_type.lower():
        zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_dir)
        _, _, test_set = stratified_zone_split(
            dataset_dir, synthetic=synthetic, mw_indices=mw_indices,
            val_size=val_size, test_size=test_size,
            balanced_val=balanced_val, val_samples_per_zone=val_samples_per_zone,
            balanced_test=balanced_test, test_samples_per_zone=test_samples_per_zone,
        )
        if 'classifier' in model_type.lower():
            test_set = ZoneSubset(test_set, zones, regression=False)
        else:
            test_set = ZoneSubset(test_set, zones)
    else:
        norm_stats = load_normalization_stats(dataset_dir)
        labels_mean = norm_stats['labels_mean']
        labels_std = norm_stats['labels_std']
        _, _, test_set = train_val_test_split(dataset_dir, synthetic=synthetic, mw_indices=mw_indices)

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


def collect_regression_outputs(model, test_loader, device='cpu', two_stage=False):
    """Run inference and keep signals alongside predictions (for signal visualization)."""
    all_signals, all_preds, all_labels, all_zones = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if two_stage:
                signals, labels, zones_batch = batch
                signals = signals.to(device)
                labels = labels.to(device)
                logits = model.forward_classifier(signals)
                zones_pred = logits.argmax(dim=1)
                preds = model.forward_regressor(signals, zones_pred)
                zones_batch = zones_batch.to(device)
            else:
                signals, labels, zones_batch = batch
                signals = signals.to(device)
                labels = labels.to(device)
                zones_batch = zones_batch.to(device)
                preds = model(signals, zones_batch)

            all_signals.append(signals.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_zones.append(zones_batch.cpu())

    return (
        torch.cat(all_signals),
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_zones),
    )


def compute_metrics(y_pred, y_true, labels_mean, labels_std, pred_labels_mean=None, pred_labels_std=None):
    """Denormalize y_true with labels_mean/std; y_pred with pred_* if cross-dataset."""
    if pred_labels_mean is None:
        pred_labels_mean = labels_mean
    if pred_labels_std is None:
        pred_labels_std = labels_std
    y_pred_denorm = denormalize_labels(y_pred, pred_labels_mean, pred_labels_std).numpy()
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std).numpy()
    units = ['A'] * 3
    label_names = ['Ax', 'Ay', 'Az']
    results = {}
    for i, name in enumerate(label_names):
        mae = float(np.abs(y_pred_denorm[:, i] - y_true_denorm[:, i]).mean())
        rmse = float(np.sqrt(((y_pred_denorm[:, i] - y_true_denorm[:, i]) ** 2).mean()))
        results[name] = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R²': round(float(r2_score(y_true_denorm[:, i], y_pred_denorm[:, i])), 4),
        }
    mae_mean = np.mean([results[name]['MAE'] for name in label_names])

    print("\n===== Evaluation Results (on the test set) =====")
    print(f"MAE mean: {mae_mean:.4f} {units[0]}\n")
    header = [f"{name} ({units[i]})" for i, name in enumerate(label_names)]
    metric_width = 16
    col_width = max(12, max(len(h) for h in header))
    metrics = [
        ('MAE', 'MAE (A)'),
        ('RMSE', 'RMSE (A)'),
        ('R²', 'R²'),
    ]
    print("| {:<{w}} |".format("Metrics", w=metric_width) + " ".join([f"{h:>{col_width}}" for h in header]) + " |")
    print("|" + "-" * (metric_width + 2) + "|" + "|".join(["-" * col_width] * len(header)) + "-|")
    for metric, label in metrics:
        if metric == 'R²':
            row = [f"{results[name][metric]:.4f}" for name in label_names]
        else:
            row = [f"{results[name][metric]:.4f} {units[i]}" for i, name in enumerate(label_names)]
        print("| {:<{w}} |".format(label, w=metric_width) + " ".join([f"{val:>{col_width}}" for val in row]) + " |")
    print()
    return results


def plot_test_precision(y_pred, y_true, labels_mean, labels_std, model_name=None, save_path=None,pred_labels_mean=None, pred_labels_std=None, show=False):
    """
    Visualize test prediction precision: histograms and scatter plots of errors per axis.
    """
    if pred_labels_mean is None:
        pred_labels_mean = labels_mean
    if pred_labels_std is None:
        pred_labels_std = labels_std
    y_pred_denorm = denormalize_labels(y_pred, pred_labels_mean, pred_labels_std).numpy()
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std).numpy()
    label_names = ['Ax', 'Ay', 'Az']
    units = ['A', 'A', 'A']
    errors = y_pred_denorm - y_true_denorm

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11), gridspec_kw={'hspace': 0.4, 'top': 0.87})
    fig.suptitle('Prediction Precision (on test_set)', fontsize=14, fontweight='bold', y=0.98)

    if model_name:
        fig.text(0.5, 0.93, f'Model: {model_name}', ha='center', fontsize=12, fontweight='bold')

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

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test precision plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_zone_precision(y_pred, y_true, zones, labels_mean, labels_std, save_path=None, show=False):
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
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_save_path(save_dir, filename):
    return os.path.join(save_dir, filename) if save_dir else None


def _extract_batch_item(batch, index):
    """Extract one sample from a DataLoader batch (index within the batch)."""
    if len(batch) == 2:
        signals, labels = batch
        return signals[index:index + 1], labels[index:index + 1], None
    signals, labels, zones = batch
    return signals[index:index + 1], labels[index:index + 1], zones[index:index + 1]


def get_test_sample(test_loader, sample_index=0):
    """Return (signals, labels, zones_or_none, global_index) for one test sample."""
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")
    seen = 0
    for batch in test_loader:
        batch_size = batch[0].size(0)
        if seen + batch_size > sample_index:
            local_idx = sample_index - seen
            signals, labels, zones = _extract_batch_item(batch, local_idx)
            return signals, labels, zones, sample_index
        seen += batch_size
    raise IndexError(f"sample_index={sample_index} out of range (test set has {seen} samples)")


def _is_two_stage_model(model_name):
    return model_name in (
        'ZoneAwareTwoStage', 'ZoneAwareTwoStageJoint', 'ZoneAwareTwoStageJointDeep',
    )


def _is_zone_regressor(model_name):
    return model_name in ('ZoneAwareRegressor', 'ZoneAwareRegressor2')


def predict_single(model, signals, zones, model_name, device):
    """Run one forward pass and return predictions (batch size 1)."""
    signals = signals.to(device)
    with torch.no_grad():
        if _is_two_stage_model(model_name):
            logits = model.forward_classifier(signals)
            zones_pred = logits.argmax(dim=1)
            return model.forward_regressor(signals, zones_pred)
        if _is_zone_regressor(model_name):
            if zones is None:
                raise ValueError("ZoneAwareRegressor requires zone labels for this evaluation path.")
            return model(signals, zones.to(device))
        return model(signals)


def measure_inference_time(model, signals, zones, model_name, device, warmup=5, repeats=50):
    """Measure single-sample inference latency (milliseconds)."""
    def forward():
        predict_single(model, signals, zones, model_name, device)

    if device.startswith('cuda'):
        torch.cuda.synchronize()
    for _ in range(warmup):
        forward()
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms = np.array(times_ms, dtype=np.float64)
    return {
        'mean_ms': float(times_ms.mean()),
        'std_ms': float(times_ms.std()),
        'min_ms': float(times_ms.min()),
        'max_ms': float(times_ms.max()),
        'median_ms': float(np.median(times_ms)),
        'repeats': repeats,
    }


def evaluate_single_sample(
    model,
    test_loader,
    model_name,
    device,
    labels_mean,
    labels_std,
    pred_labels_mean=None,
    pred_labels_std=None,
    sample_index=0,
    warmup=5,
    repeats=50,
):
    """Predict one test sample, report latency and denormalized errors."""
    if pred_labels_mean is None:
        pred_labels_mean = labels_mean
    if pred_labels_std is None:
        pred_labels_std = labels_std

    signals, labels, zones, global_idx = get_test_sample(test_loader, sample_index)
    timing = measure_inference_time(
        model, signals, zones, model_name, device, warmup=warmup, repeats=repeats,
    )
    pred = predict_single(model, signals, zones, model_name, device).cpu()
    labels = labels.cpu()

    pred_denorm = denormalize_labels(pred, pred_labels_mean, pred_labels_std).numpy()[0]
    true_denorm = denormalize_labels(labels, labels_mean, labels_std).numpy()[0]
    errors = pred_denorm - true_denorm
    abs_errors = np.abs(errors)

    label_names = ['Ax', 'Ay', 'Az']
    print("\n===== Single-sample inference =====")
    print(f"Test sample index : {global_idx}")
    print(f"Device            : {device}")
    print(f"Inference path    : {model_name}")
    if _is_two_stage_model(model_name):
        print("Zone at inference : predicted by classifier (deployable path)")
    elif _is_zone_regressor(model_name):
        print("Zone at inference : true zone from test labels (oracle — not deployable alone)")
    print(f"Latency ({timing['repeats']} runs): "
          f"mean={timing['mean_ms']:.3f} ms | median={timing['median_ms']:.3f} ms | "
          f"std={timing['std_ms']:.3f} ms | min={timing['min_ms']:.3f} ms | max={timing['max_ms']:.3f} ms")
    print("\n| Component | True (A) | Pred (A) | Error (A) | |Error| (A) |")
    print("|-----------|----------|----------|-----------|------------|")
    for i, name in enumerate(label_names):
        print(f"| {name:<9} | {true_denorm[i]:>8.4f} | {pred_denorm[i]:>8.4f} | "
              f"{errors[i]:>+9.4f} | {abs_errors[i]:>10.4f} |")
    print(f"| {'MAE mean':<9} |          |          |           | {abs_errors.mean():>10.4f} |")
    print()

    return {
        'sample_index': global_idx,
        'timing_ms': timing,
        'true_A': true_denorm.tolist(),
        'pred_A': pred_denorm.tolist(),
        'error_A': errors.tolist(),
        'abs_error_A': abs_errors.tolist(),
        'mae_mean_A': float(abs_errors.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on its test set.')
    parser.add_argument('--model', type=str, required=True,
                        help='Exact model class name (e.g. ODMR_CNN, ZoneAwareRegressor, ZoneClassifier, ZoneAwareTwoStageJoint, ZoneAwareTwoStageJointDeep)')
    parser.add_argument('--dataset_dir', type=str, default="dataset_multi_mw_2",
                        help='PyTorch dataset for evaluation (test split)')
    parser.add_argument('--train_dataset_dir', type=str, default=None,
                        help='Dataset the model was trained on (default: same as --dataset_dir). '
                             'Use for cross-dataset evaluation, e.g. train on dataset_new_2, test on dataset_new_4_homogeneous.')
    parser.add_argument('--mw_configs', type=int, nargs='+', default=None,
                        help='Subset of MW config indices (default: all channels in the dataset)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic dataset (5 MW configs, 400 freq)')
    parser.add_argument('--axis', type=str, default='mean', choices=['mean', 'Ax', 'Ay', 'Az'],
                        help='Component for per-zone MAE sphere plot (default: mean over Ax,Ay,Az)')
    parser.add_argument('--save_plots', type=str, default=None,
                        help='Directory to save evaluation plots (optional)')
    parser.add_argument('--plot', action='store_true',
                        help='Display evaluation plots interactively (heatmap, sphere, test precision)')
    parser.add_argument('--plot_extreme_zones', action='store_true',
                        help='Plot ODMR signals for best- and worst-MAE zones on the test set')
    parser.add_argument('--min_zone_samples', type=int, default=1,
                        help='Minimum test samples per zone when picking best/worst (default: 1, same as sphere)')
    parser.add_argument('--balanced_test', action='store_true', default=True,
                        help='Balanced test: 1 sample per zone (default: True, matches training)')
    parser.add_argument('--no_balanced_test', action='store_false', dest='balanced_test',
                        help='Use proportional val/test split instead of balanced test')
    parser.add_argument('--test_samples_per_zone', type=int, default=1,
                        help='Test samples per zone when --balanced_test (default: 1)')
    parser.add_argument('--val_samples_per_zone', type=int, default=None,
                        help='Val samples per zone (default: auto from 10%% target)')
    parser.add_argument('--no_balanced_val', action='store_true',
                        help='Proportional val/test per zone instead of homogeneous val')
    parser.add_argument('--single_sample', action='store_true',
                        help='Benchmark inference time and prediction error on one test sample')
    parser.add_argument('--single_sample_only', action='store_true',
                        help='With --single_sample, skip full test-set evaluation')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='Index of the test sample for --single_sample (default: 0)')
    parser.add_argument('--timing_warmup', type=int, default=5,
                        help='Warmup forward passes before timing (default: 5)')
    parser.add_argument('--timing_repeats', type=int, default=50,
                        help='Timed forward passes for latency stats (default: 50)')
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Add 'datasets_pytorch' if necessary
    def resolve_dataset_path(path):
        if path is None:
            return None
        if not (path.startswith('datasets_pytorch') or os.path.isabs(path)):
            return os.path.join("datasets_pytorch", path)
        return path

    args.dataset_dir = resolve_dataset_path(args.dataset_dir)
    train_dataset_dir = resolve_dataset_path(args.train_dataset_dir or args.dataset_dir)

    mw_indices = resolve_mw_indices(synthetic=args.synthetic, mw_configs=args.mw_configs, dataset_dir=args.dataset_dir)

    # Paths
    if args.model not in models.available_models():
        raise ValueError(f"Unknown model: {args.model}. Available: {list(models.available_models().keys())}")
    model_name = args.model
    model_key = model_name.lower()
    model_dir = resolve_model_output_dir(train_dataset_dir, model_key, mw_indices)

    print(f"Evaluation dataset : {args.dataset_dir}")
    if train_dataset_dir != args.dataset_dir:
        print(f"Model train dataset: {train_dataset_dir}  (cross-dataset)")
    print(f"Loading model from   : {model_dir}")
    print(f"MW configs: {mw_indices} ({len(mw_indices)}/{detect_num_mw_configs(args.dataset_dir, args.synthetic)} channels)")

    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)

    test_loader, labels_mean, labels_std = get_test_loader(
        args.dataset_dir, model_name, batch_size=64, mw_indices=mw_indices, synthetic=args.synthetic,
        balanced_val=not args.no_balanced_val, val_samples_per_zone=args.val_samples_per_zone,
        balanced_test=args.balanced_test, test_samples_per_zone=args.test_samples_per_zone,
    )
    train_norm = load_normalization_stats(train_dataset_dir)
    pred_labels_mean = train_norm['labels_mean']
    pred_labels_std = train_norm['labels_std']
    model = load_model(model_name, model_dir, train_dataset_dir, device=device, mw_indices=mw_indices, synthetic=args.synthetic)

    if args.single_sample:
        if model_name in ('ZoneClassifier', 'ZoneClassifier2'):
            print("Single-sample timing is not supported for classifiers.")
            return
        evaluate_single_sample(
            model, test_loader, model_name, device,
            labels_mean, labels_std,
            pred_labels_mean=pred_labels_mean, pred_labels_std=pred_labels_std,
            sample_index=args.sample_index,
            warmup=args.timing_warmup, repeats=args.timing_repeats,
        )
        if args.single_sample_only:
            return

    if model_name in ('ZoneClassifier', 'ZoneClassifier2'):
        evaluate_classifier(model, test_loader, device=device)
        return
    elif model_name in ('ZoneAwareRegressor', 'ZoneAwareRegressor2'):
        signals, y_pred, y_true, zones = collect_regression_outputs(model, test_loader, device=device)
        if args.plot or args.save_plots:
            plot_zone_precision(y_pred, y_true, zones, labels_mean, labels_std,
                                save_path=plot_save_path(args.save_plots, 'zone_mae_heatmap.png'),
                                show=args.plot)
            plot_zone_mae_on_sphere(y_pred, y_true, labels_mean, labels_std, zones=zones, axis=args.axis,
                                    title=model_name, show=args.plot,
                                    save_path=plot_save_path(args.save_plots, 'zone_mae_sphere.png'))
        if args.plot_extreme_zones and (args.plot or args.save_plots):
            plot_extreme_zone_signals(
                signals, y_pred, y_true, zones, labels_mean, labels_std,
                get_frequency_axis(args.dataset_dir), mw_indices=mw_indices, axis=args.axis,
                min_samples=args.min_zone_samples, title=model_name,
                show=args.plot,
                save_path=plot_save_path(args.save_plots, 'extreme_zone_signals.png'),
            )
    elif model_name in (
        'ZoneAwareTwoStage', 'ZoneAwareTwoStageJoint', 'ZoneAwareTwoStageJointDeep',
    ):
        signals, y_pred, y_true, zones = collect_regression_outputs(
            model, test_loader, device=device, two_stage=True,
        )
        if args.plot or args.save_plots:
            plot_zone_precision(y_pred, y_true, zones, labels_mean, labels_std,
                                save_path=plot_save_path(args.save_plots, 'zone_mae_heatmap.png'),
                                show=args.plot)
            plot_zone_mae_on_sphere(y_pred, y_true, labels_mean, labels_std, zones=zones, axis=args.axis,
                                    title=model_name, show=args.plot,
                                    save_path=plot_save_path(args.save_plots, 'zone_mae_sphere.png'))
        if args.plot_extreme_zones and (args.plot or args.save_plots):
            plot_extreme_zone_signals(
                signals, y_pred, y_true, zones, labels_mean, labels_std,
                get_frequency_axis(args.dataset_dir), mw_indices=mw_indices, axis=args.axis,
                min_samples=args.min_zone_samples, title=model_name,
                show=args.plot,
                save_path=plot_save_path(args.save_plots, 'extreme_zone_signals.png'),
            )
    else:
        y_pred, y_true = evaluate_cnn(model, test_loader, device=device)
        if args.plot or args.save_plots:
            plot_zone_mae_on_sphere(y_pred, y_true, labels_mean, labels_std, zones=None, axis=args.axis, title=model_name, show=args.plot,save_path=plot_save_path(args.save_plots, 'zone_mae_sphere.png'))

    compute_metrics(y_pred, y_true, labels_mean, labels_std, pred_labels_mean=pred_labels_mean, pred_labels_std=pred_labels_std)

    if args.plot or args.save_plots:
        plot_test_precision(
            y_pred, y_true, labels_mean, labels_std, model_name=model_name,
            save_path=plot_save_path(args.save_plots, 'test_precision.png'),
            pred_labels_mean=pred_labels_mean, pred_labels_std=pred_labels_std,
            show=args.plot,
        )


if __name__ == '__main__':
    main()