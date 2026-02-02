import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from split_dataset import train_val_test_split
from models import ODMR_CNN


def evaluate():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_DIR = "pytorch_dataset_example"
    MODEL_PATH = Path("saved_models/cnn_odmr_best.pt")
    BATCH_SIZE = 64

    # ===== Dataset =====
    train_set, val_set, test_set = train_val_test_split(DATASET_DIR)

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    full_dataset = train_set.dataset

    # ===== Model =====
    input_channels = full_dataset[0][0].shape[0]  # = 1
    output_dim = 1  # Magnitude

    model = ODMR_CNN(input_channels, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    criterion = nn.MSELoss(reduction="mean")

    # ===== Metrics accumulators =====
    mse_sum = 0.0
    abs_error_sum = 0.0
    sq_error_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            mse_sum += criterion(pred, y).item() * y.size(0)
            abs_error_sum += torch.sum(torch.abs(pred - y)).item()
            sq_error_sum += torch.sum((pred - y) ** 2).item()
            n_samples += y.size(0)

    # ===== Final metrics =====
    mse = mse_sum / n_samples
    mae = abs_error_sum / n_samples
    rmse = np.sqrt(sq_error_sum / n_samples)

    # Compute magnitude range from metadata
    metadata_magnitudes = np.sqrt(full_dataset.metadata["Ax"]**2 + 
                                  full_dataset.metadata["Ay"]**2 + 
                                  full_dataset.metadata["Az"]**2)
    label_range = metadata_magnitudes.max() - metadata_magnitudes.min()

    nrmse = rmse / label_range

    # ===== Print results =====
    print("\n===== TEST SET EVALUATION (Magnitude) =====")
    print(f"MSE   : {mse:.3e}")
    print(f"MAE   : {mae:.3e}")
    print(f"RMSE  : {rmse:.3e}")
    print(f"NRMSE : {nrmse*100:.2f}%")

    # ===== Plot predictions vs true values =====
    y_true = np.zeros(n_samples, dtype=np.float32)
    y_pred = np.zeros(n_samples, dtype=np.float32)
    idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            batch_size = y.size(0)
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy().flatten()
            y_true[idx:idx + batch_size] = y.cpu().numpy().flatten()
            y_pred[idx:idx + batch_size] = pred
            idx += batch_size
    plot_magnitude_predictions(y_true, y_pred)


def plot_magnitude_predictions(y_true, y_pred):
    """
    Plot predicted vs true magnitude values.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        True magnitude values
    y_pred : np.ndarray, shape (N,)
        Predicted magnitude values
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Scatter plot: Predicted vs True
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('True Magnitude')
    ax.set_ylabel('Predicted Magnitude')
    ax.set_title('Predicted vs True Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # 2. Error distribution
    ax = axes[1]
    errors = y_pred - y_true
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics on the plot
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax.text(0.05, 0.95, f'Mean: {mean_error:.3e}\nStd: {std_error:.3e}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()