import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
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


def train(batch_size=64, epochs=300, lr=3e-4, weight_decay=1e-3, dataset_dir="pytorch_dataset_example"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = dataset_dir
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LR = lr
    WEIGHT_DECAY = weight_decay

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
    output_dim = 3                                # (Ax, Ay, Az)

    # Verify data metrics from metadata (faster than loading all samples)
    print("Labels statistics (from metadata):")
    label_cols = full_dataset.label_cols
    print(f"Min: [{full_dataset.metadata[label_cols[0]].min():.4f}, {full_dataset.metadata[label_cols[1]].min():.4f}, {full_dataset.metadata[label_cols[2]].min():.4f}]")
    print(f"Max: [{full_dataset.metadata[label_cols[0]].max():.4f}, {full_dataset.metadata[label_cols[1]].max():.4f}, {full_dataset.metadata[label_cols[2]].max():.4f}]")
    print(f"Mean: [{full_dataset.metadata[label_cols[0]].mean():.4f}, {full_dataset.metadata[label_cols[1]].mean():.4f}, {full_dataset.metadata[label_cols[2]].mean():.4f}]")
    
    # Sample a few signals to check normalization
    print("Signals statistics (sample of 100 signals):")
    sample_indices = np.random.choice(len(full_dataset), min(100, len(full_dataset)), replace=False)
    sample_signals = torch.cat([full_dataset[i][0] for i in sample_indices], dim=0)
    print(f"Min: {sample_signals.min():.4f}")
    print(f"Max: {sample_signals.max():.4f}")
    print(f"Std: {sample_signals.std():.4f}\n")

    # Auto-adjust dropout based on dataset size
    dataset_size = len(train_set)
    if dataset_size < 500:
        dropout = 0.5  # High dropout for very small datasets
    elif dataset_size < 2000:
        dropout = 0.4
    else:
        dropout = 0.35  # Moderate dropout for larger datasets
    
    model = ODMR_CNN(input_channels, output_dim, dropout=dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model created with {n_params} parameters")
    print(f"Dataset size: {dataset_size} samples")
    if n_params > 0:
        ratio = dataset_size / n_params
        print(f"Samples per parameter: {ratio:.1f} (want >10, ideally >100)")
    print(f"Dropout: {dropout}")
    print(f"Training on device: {DEVICE}\n")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # OneCycleLR scheduler - proven best for many datasets
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 10,  # Peak learning rate
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,   # 30% warmup
        anneal_strategy='cos',
        div_factor=25.0,  # Initial LR = max_lr/25
        final_div_factor=1000.0  # Final LR = max_lr/1000
    )

    early_stopping = EarlyStopping(patience=30, min_delta=1e-5)

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
            scheduler.step()            # step scheduler per batch for OneCycleLR

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
        abs_error /= len(val_loader) # MAE per axis
        rmse = torch.sqrt(sq_error / len(val_loader.dataset)) # RMSE per axis
        
        # Normalized metrics (by range and by mean)
        # Use auto-detected label column names
        label_cols = full_dataset.label_cols
        label_range = torch.tensor([full_dataset.metadata[label_cols[0]].max() - full_dataset.metadata[label_cols[0]].min(),
                                    full_dataset.metadata[label_cols[1]].max() - full_dataset.metadata[label_cols[1]].min(),
                                    full_dataset.metadata[label_cols[2]].max() - full_dataset.metadata[label_cols[2]].min()],
                                    dtype=torch.float32, device=DEVICE)
        label_mean = torch.tensor([full_dataset.metadata[label_cols[0]].mean(),
                                   full_dataset.metadata[label_cols[1]].mean(),
                                   full_dataset.metadata[label_cols[2]].mean()],
                                   dtype=torch.float32, device=DEVICE)
        
        nrmse = rmse / label_range  # Normalized RMSE by range
        nmae = abs_error / label_range  # Normalized MAE by range
        mae_rel_mean = abs_error / torch.abs(label_mean)  # MAE relative to mean

        # Store metrics for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['nmae_ax'].append(nmae[0].item() * 100)
        history['nmae_ay'].append(nmae[1].item() * 100)
        history['nmae_az'].append(nmae[2].item() * 100)
        history['nrmse_ax'].append(nrmse[0].item() * 100)
        history['nrmse_ay'].append(nrmse[1].item() * 100)
        history['nrmse_az'].append(nrmse[2].item() * 100)

        # Get current LR (OneCycleLR updates per batch, so just check current value)
        current_lr = optimizer.param_groups[0]['lr']

        # Use label names for display
        label_cols = full_dataset.label_cols
        print(
            f"Epoch {epoch+1:03d} :\n"
            f" -> Train_loss: {train_loss:.3e} | Val_loss: {val_loss:.3e} \n"
            f" -> LR: {optimizer.param_groups[0]['lr']:.2e} \n"
            f" -> NMAE:  {label_cols[0]} {nmae[0]*100:.2f}%  - {label_cols[1]} {nmae[1]*100:.2f}%  - {label_cols[2]} {nmae[2]*100:.2f}% \n"
            f" -> NRMSE: {label_cols[0]} {nrmse[0]*100:.2f}% - {label_cols[1]} {nrmse[1]*100:.2f}% - {label_cols[2]} {nrmse[2]*100:.2f}%"
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
    train(dataset_dir="synthetic_dataset")