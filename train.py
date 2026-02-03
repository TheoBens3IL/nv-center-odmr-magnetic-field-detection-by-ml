import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import numpy as np
from split_dataset import train_val_test_split
from utils import load_normalization_stats, denormalize_labels
import models  # Import the models module

class EarlyStopping:
    '''
    Early stopping to halt training when validation loss doesn't improve after a set number (patience) of epochs.
    If no improvement after 'patience' epochs, training stops.
    '''
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience       # epochs to wait for improvement
        self.min_delta = min_delta     # minimum change to qualify as improvement
        self.best_loss = float('inf')  # best validation loss observed
        self.counter = 0               # epochs since last improvement
        self.best_state = None         # best model state

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta: # improvement observed
            self.best_loss = val_loss                  # update best loss
            self.counter = 0                           # reset counter
            self.best_state = model.state_dict()       # save best model state
        else:
            self.counter += 1                          # if no improvement, increment counter

        return self.counter >= self.patience           # return True if early stopping criterion met  


def train(batch_size=32, epochs=200, lr=2e-4, weight_decay=5e-4, dataset_dir="dataset_multi_mw", 
          patience=20, min_delta=1e-6, model_name="ODMR_CNN_Compact"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = dataset_dir
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LR = lr
    WEIGHT_DECAY = weight_decay
    MODEL_NAME = model_name

    train_set, val_set, test_set = train_val_test_split(DATASET_DIR)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    full_dataset = train_set.dataset              # full dataset for input/output dimensions
    output_dim = 3                                # (Ax, Ay, Az)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")
    print(f"  Total: {len(train_set) + len(val_set) + len(test_set)} samples\n")
    
    # Load normalization stats for denormalization during evaluation
    norm_stats = load_normalization_stats(DATASET_DIR)
    labels_mean = norm_stats['labels_mean']
    labels_std = norm_stats['labels_std']
    print(f"Label normalization stats loaded:")
    print(f"  Mean: {labels_mean}")
    print(f"  Std:  {labels_std}\n")

    # Verify datas metrics (min, max, mean) of labels and signals
    print("NORMALIZED labels statistics (from dataset):")
    all_labels = torch.stack([full_dataset[i][1] for i in range(len(full_dataset))], dim=0)
    print(f"Min: {all_labels.min(dim=0).values}")
    print(f"Max: {all_labels.max(dim=0).values}")
    print(f"Mean: {all_labels.mean(dim=0)}")
    print(f"Std: {all_labels.std(dim=0)}")
    print("Signals statistics:")
    all_signals = torch.cat([full_dataset[i][0] for i in range(len(full_dataset))], dim=0)
    print(f"Min: {all_signals.min()}")
    print(f"Max: {all_signals.max()}")
    print(f"Std: {all_signals.std()}\n")

    # Create model based on model_name
    available_models = {
        'ODMR_CNN': models.ODMR_CNN,
        'ODMR_CNN_Compact': models.ODMR_CNN_Compact,
        'ODMR_CNN_Deep': models.ODMR_CNN_Deep,
        'FrequencyAttention': models.FrequencyAttention,
        'MWConfig_CNN': models.MWConfig_CNN,
    }
    
    if MODEL_NAME not in available_models:
        raise ValueError(f"Unknown model: {MODEL_NAME}. Available models: {list(available_models.keys())}")
    
    model_class = available_models[MODEL_NAME]
    model = model_class(n_freq=201, output_dim=output_dim).to(DEVICE)
    
    print(f"Model: {MODEL_NAME}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Parameter to data ratio: 1:{len(train_set) / sum(p.numel() for p in model.parameters()):.2f}")
    print(f"Training on device: {DEVICE}\n")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts - good for larger datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,        # Initial restart period (longer for more data)
        T_mult=2,      # Period multiplier
        eta_min=1e-6   # Minimum LR
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # Metrics history for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'nmae_ax': [], 'nmae_ay': [], 'nmae_az': [],
        'nrmse_ax': [], 'nrmse_ay': [], 'nrmse_az': [],
    }

    for epoch in range(EPOCHS):
        # ===== Training Phase ===== #
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()       # clear gradients
            pred = model(x)             # forward pass
            loss = criterion(pred, y)   # compute loss
            loss.backward()             # backward pass
            optimizer.step()            # update weights

            train_loss += loss.item()   # accumulate loss

        train_loss /= len(train_loader) # average training loss

        # ===== Validation Phase ===== #
        model.eval()
        val_loss = 0.0   # validation loss on normalized labels
        abs_error_denorm = torch.zeros(3)  # MAE in original scale
        sq_error_denorm = 0.0   # MSE in original scale
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
                sq_error_denorm += torch.sum((pred_denorm - y_denorm) ** 2, dim=0)
                n_samples += x.size(0)

        val_loss /= len(val_loader)  # average validation loss (normalized)
        mae_denorm = abs_error_denorm / n_samples  # MAE per axis (original scale)
        rmse_denorm = torch.sqrt(sq_error_denorm / n_samples)  # RMSE per axis (original scale)
        
        # Normalized metrics (NMAE, NRMSE) using original scale std
        label_range_tensor = torch.tensor(labels_std * 6, dtype=torch.float32)  # ~3 sigma range
        
        nrmse = rmse_denorm / label_range_tensor  # Normalized RMSE by range
        nmae = mae_denorm / label_range_tensor  # Normalized MAE by range

        # Store metrics for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['nmae_ax'].append(nmae[0].item() * 100)
        history['nmae_ay'].append(nmae[1].item() * 100)
        history['nmae_az'].append(nmae[2].item() * 100)
        history['nrmse_ax'].append(nrmse[0].item() * 100)
        history['nrmse_ay'].append(nrmse[1].item() * 100)
        history['nrmse_az'].append(nrmse[2].item() * 100)

        # Scheduler step (Cosine annealing updates every epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1:03d} :\n"
            f" -> Train_loss: {train_loss:.3e} | Val_loss: {val_loss:.3e} \n"
            f" -> LR: {optimizer.param_groups[0]['lr']:.2e} \n"
            f" -> NMAE:  Ax {nmae[0]*100:.2f}%  - Ay {nmae[1]*100:.2f}%  - Az {nmae[2]*100:.2f}% \n"
            f" -> NRMSE: Ax {nrmse[0]*100:.2f}% - Ay {nrmse[1]*100:.2f}% - Az {nrmse[2]*100:.2f}%"
        )

        # Early stopping check
        if early_stopping.step(val_loss, model):
            print("Early stopping triggered")
            break

    # Restore best model (whether early stopping was triggered or not)
    model.load_state_dict(early_stopping.best_state)
    
    # Save the best model
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / f"cnn_odmr_loss_{early_stopping.best_loss:.3e}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved as {model_path} (val_loss: {early_stopping.best_loss:.3e})")

    # Plot training history
    plot_training_history(history, early_stopping.best_loss)


def plot_training_history(history, best_loss):
    """Plot training and validation metrics over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics Evolution', fontsize=16, fontweight='bold')
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. NMAE per axis
    ax = axes[0, 1]
    ax.plot(epochs, history['nmae_ax'], label='Ax', linewidth=2)
    ax.plot(epochs, history['nmae_ay'], label='Ay', linewidth=2)
    ax.plot(epochs, history['nmae_az'], label='Az', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NMAE (%)')
    ax.set_title('Normalized Mean Absolute Error by Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. NRMSE per axis
    ax = axes[1, 0]
    ax.plot(epochs, history['nrmse_ax'], label='Ax', linewidth=2)
    ax.plot(epochs, history['nrmse_ay'], label='Ay', linewidth=2)
    ax.plot(epochs, history['nrmse_az'], label='Az', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NRMSE (%)')
    ax.set_title('Normalized Root Mean Square Error by Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Average NMAE and NRMSE
    ax = axes[1, 1]
    avg_nmae = [(history['nmae_ax'][i] + history['nmae_ay'][i] + history['nmae_az'][i]) / 3 
                for i in range(len(epochs))]
    avg_nrmse = [(history['nrmse_ax'][i] + history['nrmse_ay'][i] + history['nrmse_az'][i]) / 3 
                 for i in range(len(epochs))]
    ax.plot(epochs, avg_nmae, label='Avg NMAE', linewidth=2)
    ax.plot(epochs, avg_nrmse, label='Avg NRMSE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (%)')
    ax.set_title('Average Normalized Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    fig_path = save_dir / f"training_history_loss_{best_loss:.3e}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved as {fig_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ODMR CNN model for magnetic field detection')
    
    # Model selection
    parser.add_argument('--model', type=str, default='ODMR_CNN_Compact',
                        choices=['ODMR_CNN', 'ODMR_CNN_Compact', 'ODMR_CNN_Deep', 'FrequencyAttention', 'MWConfig_CNN'],
                        help='Model architecture to use (default: ODMR_CNN_Compact)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Maximum number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=2e-4, 
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, 
                        help='Weight decay for optimizer (default: 5e-4)')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=20, 
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--min_delta', type=float, default=1e-6, 
                        help='Minimum delta for early stopping (default: 1e-6)')
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='dataset_multi_mw', 
                        help='Path to dataset directory (default: dataset_multi_mw)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning rate:  {args.lr}")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"Patience:       {args.patience}")
    print(f"Min delta:      {args.min_delta}")
    print(f"Dataset dir:    {args.dataset_dir}")
    print("=" * 60)
    print()
    
    train(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dataset_dir=args.dataset_dir,
        patience=args.patience,
        min_delta=args.min_delta,
        model_name=args.model
    )