import torch
from torch import nn
import numpy as np
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from dataset import train_val_test_split, print_dataset_statistics, get_data_loaders, get_frequency_axis
from utils import EarlyStopping, load_normalization_stats, denormalize_labels, plot_training_history
from physics_informed import extract_odmr_peak_frequencies, physics_loss
import models


# WeightedMSELoss for HybridODMRPredictor
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
    def forward(self, pred, target):
        # Broadcast weights if needed
        if pred.shape[-1] != self.weights.shape[0]:
            raise ValueError(f"Weights shape {self.weights.shape} does not match prediction shape {pred.shape}")
        loss = ((pred - target) ** 2) * self.weights
        return loss.mean()


def train(batch_size=32, epochs=200, lr=2e-4, weight_decay=5e-4, dataset_dir="dataset_multi_mw", patience=20, min_delta=1e-6, model_name="FrequencyAttention", loss_weights=None, use_attention=False, physics_loss_weight=0.1, show_dataset_stats=False, synthetic=False):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = os.path.join("datasets_pytorch", dataset_dir)


    # Check if model is available
    if model_name not in models.available_models():
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.available_models().keys())}")

    # Select input shape based on dataset type
    if synthetic:
        n_channels, n_freq = 5, 400
    else:
        n_channels, n_freq = 10, 201

    model_class = models.available_models()[model_name]
    if model_name == 'HybridODMRPredictor':
        model = model_class(n_channels=n_channels, n_freq=n_freq, use_attention=use_attention).to(DEVICE)
    else:
        model = model_class(n_channels=n_channels, n_freq=n_freq).to(DEVICE)

    # Load dataset and create data loaders
    train_set, val_set, test_set = train_val_test_split(DATASET_DIR, synthetic=synthetic)
    train_loader, val_loader, test_loader = get_data_loaders(train_set, val_set, test_set, batch_size=batch_size, device=DEVICE)

    # Load normalization stats for denormalization during evaluation
    norm_stats = load_normalization_stats(DATASET_DIR)
    labels_mean = norm_stats['labels_mean']
    labels_std = norm_stats['labels_std']
    label_names = ['Ax', 'Ay', 'Az']
    
    if show_dataset_stats:
        print_dataset_statistics(train_set, val_set, test_set, label_names, labels_mean, labels_std)

    # Model is already instantiated above with correct n_channels and n_freq

    # Loss function
    if model_name == 'HybridODMRPredictor' and loss_weights is not None:
        criterion = WeightedMSELoss(loss_weights)
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Physics-informed loss (optional)
    physics_loss_weight = getattr(args, 'physics_loss_weight', 0.1)  # adjustable parameter
    use_physics_loss = getattr(args, 'physic_informed', False)
    freq_axis_Hz = get_frequency_axis(DATASET_DIR)  # Get frequency axis (in Hz) for physics-informed loss

    # Cosine annealing with warm restarts - good for larger datasets
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=30,        # Initial restart period (longer for more data)
    #     T_mult=2,      # Period multiplier
    #     eta_min=1e-6   # Minimum LR
    # )

    # ReduceLROnPlateau scheduler - reduces LR when a metric has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # Reduce LR by a factor of 0.5
        patience=5,      # Number of epochs with no improvement after which LR will be reduced
        threshold=1e-4,  # Threshold for measuring the new optimum
        min_lr=1e-6      # Minimum LR
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # Metrics history for plotting
    mae_keys = [f"mae_{name.lower()}" for name in label_names]  # Dynamically set MAE keys based on label_names
    history = {
        'train_loss': [],
        'val_loss': [],
        **{k: [] for k in mae_keys},
        'physics_loss': [],
    }

    # ========== TRAINING LOOP ========== #
    for epoch in range(epochs):
        # ===== Training Phase ===== #
        model.train()
        train_loss = 0.0
        physics_loss_accum = 0.0
        for batch in train_loader:
            signals, labels = batch
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(signals)
            loss = criterion(preds, labels)
            if use_physics_loss:
                # Extract measured peak frequencies from each spectrum in the batch (in Hz, 8 peaks)
                signals_np = signals.detach().cpu().numpy()
                measured_freqs = []
                for s in signals_np:
                    # If multi-channel, use mean or first channel for each spectrum
                    if s.ndim == 2:
                        spectrum = s[0]
                    else:
                        spectrum = s
                    peak_freqs = extract_odmr_peak_frequencies(spectrum, freq_axis_Hz, num_peaks=8)
                    measured_freqs.append(peak_freqs)
                measured_freqs = torch.tensor(np.array(measured_freqs), dtype=torch.float32, device=DEVICE)  # (batch, 8) in Hz
                preds_denorm = denormalize_labels(preds, labels_mean, labels_std).to(DEVICE)
                # Compute physics-informed loss using new geometry
                loss_phys = physics_loss(preds_denorm, measured_freqs)
                total_loss = loss + physics_loss_weight * loss_phys
                physics_loss_accum += loss_phys.item() * signals.size(0)
            else:
                total_loss = loss
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item() * signals.size(0)
        train_loss /= len(train_loader.dataset)
        if use_physics_loss:
            physics_loss_val = physics_loss_accum / len(train_loader.dataset)
        else:
            physics_loss_val = 0.0

        # ===== Validation Phase ===== #
        model.eval()
        val_loss = 0.0   # validation loss on normalized labels
        abs_error_denorm = torch.zeros(3)  # MAE in original scale
        n_samples = 0
        
        with torch.no_grad(): # disable gradient computation for validation
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                
                # Loss on normalized labels (for training)
                val_loss += criterion(pred, y).item()
                
                # Denormalize for evaluation metrics in original scale
                pred_denorm = denormalize_labels(pred.cpu(), labels_mean, labels_std)
                y_denorm = denormalize_labels(y.cpu(), labels_mean, labels_std)
                abs_error_denorm += torch.sum(torch.abs(pred_denorm - y_denorm), dim=0)
                n_samples += x.size(0)

        val_loss /= len(val_loader)
        mae_denorm = abs_error_denorm / n_samples
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for i, k in enumerate(mae_keys):
            history[k].append(mae_denorm[i].item())
        history['physics_loss'].append(physics_loss_val)

        # Scheduler step (Cosine annealing updates every epoch)
        # scheduler.step()

        # Scheduler step (ReduceLROnPlateau updates on validation loss)
        scheduler.step(val_loss)

        # Format units for display
        units = ['A', 'A', 'A']

        if epoch == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            mae_str = " | ".join([
                f"{label_names[i]} {mae_denorm[i]:.4f} {units[i]}" for i in range(3)
            ])
            print(
                f"Epoch [{epoch+1:03d}/{epochs}] :\n"
                f" -> Train_loss: {train_loss:.4f} | Val_loss: {val_loss:.4f} | Physics_loss: {physics_loss_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} \n"
                f" -> MAE :       {mae_str}"
            )

        # Early stopping check
        if early_stopping.step(val_loss, model):
            print("Early stopping triggered")
            break

    # Restore best model (whether early stopping was triggered or not)
    model.load_state_dict(early_stopping.best_state)

    # ===== Test eval ===== #
    model.eval()
    abs_error_denorm = torch.zeros(3)
    sq_error_denorm = torch.zeros(3)
    n_samples = 0
    all_labels_denorm = []
    all_preds_denorm = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            pred_denorm = denormalize_labels(pred.cpu(), labels_mean, labels_std)
            y_denorm = denormalize_labels(y.cpu(), labels_mean, labels_std)
            abs_error_denorm += torch.sum(torch.abs(pred_denorm - y_denorm), dim=0)
            sq_error_denorm += torch.sum((pred_denorm - y_denorm) ** 2, dim=0)
            n_samples += x.size(0)
            all_labels_denorm.append(y_denorm.numpy())
            all_preds_denorm.append(pred_denorm.numpy())
    mae = (abs_error_denorm / n_samples).numpy()
    rmse = torch.sqrt(sq_error_denorm / n_samples).numpy()
    print(f"\nTest set metrics (denormalized units):")
    print(f"  MAE  Ax/Ay/Az: {mae[0]:.4f} / {mae[1]:.4f} / {mae[2]:.4f} A")
    print(f"  RMSE Ax/Ay/Az: {rmse[0]:.4f} / {rmse[1]:.4f} / {rmse[2]:.4f} A")

    # === Smart saving system ===

    # 1. Dossier du modèle
    dataset_subdir = Path("models_trained") / Path(dataset_dir)
    model_dir = dataset_subdir / model_name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name.lower()}_best_model.pth"
    log_path = model_dir / f"{model_name.lower()}_train_log.json"
    plot_path = model_dir / f"{model_name.lower()}_training_plot.png"

    # 2. Calcul du meilleur score déjà enregistré (MAE)
    best_mae = None
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                prev_log = json.load(f)
                prev_mae = prev_log.get('metrics', {}).get('mae', None)
                if prev_mae is not None:
                    best_mae = sum(prev_mae) / len(prev_mae)
        except Exception:
            best_mae = None

    # 3. Calcul du MAE moyen courant
    current_mae = [float(history[mae_keys[i]][-1]) for i in range(3)]
    current_mae_mean = sum(current_mae) / len(current_mae)

    # 4. Si pas de modèle ou meilleur MAE, on sauvegarde
    if best_mae is not None:
        print(f"current_mae_mean = {current_mae_mean:.4f}, best_mae = {best_mae:.4f}")
    else:
        print(f"current_mae_mean = {current_mae_mean:.4f}, best_mae = None")
    should_save = (
        best_mae is None or current_mae_mean < best_mae
    )
    if not should_save:
        print(f"Model not saved: previous best MAE={best_mae:.4f} is better or equal to current MAE={current_mae_mean:.4f}.")
    else:
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved as {model_path} (MAE: {current_mae_mean:.4f})")
        # 4. Plot training history et sauvegarde
        fig = plot_training_history(history, label_names=label_names, show=False)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved as {plot_path}")

        # 5. Sauvegarde des scalers si pertinents
        norm_stats_path = model_dir / f"{model_name.lower()}_scaler.json"
        norm_stats = {
            'labels_mean': labels_mean.tolist() if hasattr(labels_mean, 'tolist') else list(labels_mean),
            'labels_std': labels_std.tolist() if hasattr(labels_std, 'tolist') else list(labels_std)
        }
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)

        # 6. Log complet (structure, config, timestamp, résultats)
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'model_structure': str(model),
            'config': {
                'batch_size': batch_size,
                'epochs': epochs,
                'lr': lr,
                'weight_decay': weight_decay,
                'dataset_dir': dataset_dir,
                'patience': patience,
                'min_delta': min_delta,
                'loss_weights': loss_weights,
                'use_attention': use_attention,
                'physic_informed': use_physics_loss is not None,
                'physics_loss_weight': physics_loss_weight if use_physics_loss is not None else None
            },
            'val_loss': round(float(early_stopping.best_loss), 3),
            'train_loss': round(float(history['train_loss'][-1]), 3),
            'metrics': {
                'mae': [round(float(history[mae_keys[i]][-1]), 4) for i in range(3)],
                'physics_loss': round(float(history['physics_loss'][-1]), 4) if use_physics_loss is not None else None,
            }
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"Training log saved as {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ODMR CNN model for magnetic field detection')

    # Model selection
    parser.add_argument('--model', type=str, default='FrequencyAttention', choices=['ODMR_CNN', 'ODMR_CNN_Compact', 'ODMR_CNN_Deep', 'FrequencyAttention', 'MWConfig_CNN', 'HybridODMRPredictor'], help='Model architecture to use (default: FrequencyAttention)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer (default: 5e-4)')

    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    parser.add_argument('--min_delta', type=float, default=1e-6, help='Minimum delta for early stopping (default: 1e-6)')

    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='dataset_multi_mw_2', help='Path to dataset directory (default: dataset_multi_mw_2 in datasets_pytorch/)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic dataset (5 MW configs, 400 freq)')

    # Hybrid-specific arguments
    parser.add_argument('--loss_weights', type=float, nargs=3, default=None, help='Loss weights for [Ax, Ay, Az] (only for HybridODMRPredictor)')
    parser.add_argument('--use_attention', action='store_true', help='Use attention branch in HybridODMRPredictor (default: CNN)')
    
    # Physics-informed loss parameters
    parser.add_argument('--physic_informed', action='store_true', help='Enable physics-guided loss (PINN)')
    parser.add_argument('--physics_loss_weight', type=float, default=0.1, help='Weight for physics-guided loss (PINN)')

    parser.add_argument('--show_dataset_stats', action='store_true', help='Show dataset statistics (optional)')

    args = parser.parse_args()

    # Set input shape for model param count
    if args.synthetic:
        n_channels, n_freq = 5, 400
    else:
        n_channels, n_freq = 10, 201

    if args.model == 'HybridODMRPredictor':
        nb_param = sum(p.numel() for p in models.available_models()[args.model](n_channels=n_channels, n_freq=n_freq, use_attention=args.use_attention).parameters())
    else:
        nb_param = sum(p.numel() for p in models.available_models()[args.model](n_channels=n_channels, n_freq=n_freq).parameters())
    dataset_dir_arg = os.path.join("datasets_pytorch", args.dataset_dir)
    train_set, _, _ = train_val_test_split(dataset_dir_arg, synthetic=args.synthetic)
    nb_param_per_sample = nb_param / len(train_set) if len(train_set) > 0 else float('nan')

    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"Model:          {args.model} ({nb_param:,} params -> {nb_param_per_sample:.2f} params/train sample)")
    print(f"Training on :   {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning rate:  {args.lr}")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"Patience:       {args.patience}")
    print(f"Min delta:      {args.min_delta}")
    print(f"Dataset dir:    {args.dataset_dir}")
    print(f"Synthetic:      {args.synthetic}")
    if args.model == 'HybridODMRPredictor':
        print(f"Loss weights:   {args.loss_weights}")
        print(f"Use attention:  {args.use_attention}")
    print("=" * 60)

    try:
        train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, dataset_dir=args.dataset_dir, patience=args.patience, min_delta=args.min_delta, model_name=args.model, loss_weights=args.loss_weights, use_attention=args.use_attention, physics_loss_weight=args.physics_loss_weight, show_dataset_stats=args.show_dataset_stats, synthetic=args.synthetic)
    except Exception as e:
        print(f"Error: {e}")