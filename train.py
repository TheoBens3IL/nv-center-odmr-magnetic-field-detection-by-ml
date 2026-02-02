import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import numpy as np
from split_dataset import train_val_test_split
from models import ODMR_CNN

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


def train(batch_size=16, epochs=200, lr=3e-4, weight_decay=1e-4, dataset_dir="pytorch_dataset_example", 
          patience=15, min_delta=1e-5):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = dataset_dir
    BATCH_SIZE = batch_size  # Number of samples processed together in one forward/backward pass through the neural network before updating model weights
    EPOCHS = epochs     # Larger number of epochs to let early stopping decide when to stop
    LR = lr        # Learning rate
    WEIGHT_DECAY = weight_decay  # Weight decay for optimizer

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
    input_channels = full_dataset[0][0].shape[0]  # = 1
    output_dim = 1                                # Magnitude

    # Verify datas metrics (min, max, mean) of labels and signals
    print("Labels statistics (Magnitude):")
    all_labels = torch.stack([full_dataset[i][1] for i in range(len(full_dataset))], dim=0)
    print(f"Min: {all_labels.min().item():.6f}")
    print(f"Max: {all_labels.max().item():.6f}")
    print(f"Mean: {all_labels.mean().item():.6f}")
    print("Signals statistics:")
    all_signals = torch.cat([full_dataset[i][0] for i in range(len(full_dataset))], dim=0)
    print(f"Min: {all_signals.min()}")
    print(f"Max: {all_signals.max()}")
    print(f"Std: {all_signals.std()}\n")

    model = ODMR_CNN(input_channels, output_dim).to(DEVICE)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Parameter to data ratio: 1:{len(train_set) / sum(p.numel() for p in model.parameters()):.1f}")
    print(f"Training on device: {DEVICE}\n")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing scheduler - better for small datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,        # Initial restart period
        T_mult=2,      # Period multiplier
        eta_min=1e-6   # Minimum LR
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # Metrics history for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'nmae': [],
        'nrmse': [],
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
        val_loss = 0.0   # validation loss
        abs_error = 0.0  # for mean absolute error per component
        sq_error = 0.0   # for root mean square error per component
        with torch.no_grad(): # disable gradient computation for validation
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                abs_error += torch.mean(torch.abs(pred - y), dim=0)
                sq_error += torch.mean((pred - y) ** 2, dim=0) * x.size(0)

        val_loss /= len(val_loader)  # average validation loss
        abs_error /= len(val_loader) # MAE
        rmse = torch.sqrt(sq_error / len(val_loader.dataset)) # RMSE
        
        # Compute magnitude range from metadata
        metadata_magnitudes = np.sqrt(full_dataset.metadata["Ax"]**2 + 
                                      full_dataset.metadata["Ay"]**2 + 
                                      full_dataset.metadata["Az"]**2)
        label_range = torch.tensor(metadata_magnitudes.max() - metadata_magnitudes.min(),
                                   dtype=torch.float32, device=DEVICE)
        label_mean = torch.tensor(metadata_magnitudes.mean(),
                                  dtype=torch.float32, device=DEVICE)
        
        nrmse = rmse / label_range  # Normalized RMSE by range
        nmae = abs_error / label_range  # Normalized MAE by range
        mae_rel_mean = abs_error / torch.abs(label_mean)  # MAE relative to mean

        # Store metrics for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['nmae'].append(nmae.item() * 100)
        history['nrmse'].append(nrmse.item() * 100)

        # Scheduler step (Cosine annealing updates every epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1:03d} :\n"
            f" -> Train_loss: {train_loss:.3e} | Val_loss: {val_loss:.3e} \n"
            f" -> LR: {optimizer.param_groups[0]['lr']:.2e} \n"
            f" -> NMAE:  {nmae.item()*100:.2f}% \n"
            f" -> NRMSE: {nrmse.item()*100:.2f}%"
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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Training Metrics Evolution (Magnitude Prediction)', fontsize=16, fontweight='bold')
    
    # 1. Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. NMAE
    ax = axes[1]
    ax.plot(epochs, history['nmae'], label='NMAE', linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NMAE (%)')
    ax.set_title('Normalized Mean Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. NRMSE
    ax = axes[2]
    ax.plot(epochs, history['nrmse'], label='NRMSE', linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NRMSE (%)')
    ax.set_title('Normalized Root Mean Square Error')
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
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Maximum number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=3e-4, 
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay for optimizer (default: 1e-4)')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=30, 
                        help='Early stopping patience (default: 30)')
    parser.add_argument('--min_delta', type=float, default=1e-5, 
                        help='Minimum delta for early stopping (default: 1e-5)')
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='pytorch_dataset_example', 
                        help='Path to dataset directory (default: pytorch_dataset_example)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
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
        min_delta=args.min_delta
    )