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
    output_dim = 3

    model = ODMR_CNN(input_channels, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    criterion = nn.MSELoss(reduction="mean")

    # ===== Metrics accumulators =====
    mse_sum = 0.0
    abs_error_sum = torch.zeros(3, device=DEVICE)
    sq_error_sum = torch.zeros(3, device=DEVICE)
    n_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            mse_sum += criterion(pred, y).item() * y.size(0)
            abs_error_sum += torch.sum(torch.abs(pred - y), dim=0)
            sq_error_sum += torch.sum((pred - y) ** 2, dim=0)
            n_samples += y.size(0)

    # ===== Final metrics =====
    mse = mse_sum / n_samples
    mae = abs_error_sum / n_samples
    rmse = torch.sqrt(sq_error_sum / n_samples)

    label_range = torch.tensor([
        full_dataset.metadata["Ax"].max() - full_dataset.metadata["Ax"].min(),
        full_dataset.metadata["Ay"].max() - full_dataset.metadata["Ay"].min(),
        full_dataset.metadata["Az"].max() - full_dataset.metadata["Az"].min(),
    ], dtype=torch.float32, device=DEVICE)

    nrmse = rmse / label_range

    # ===== Print results =====
    print("\n===== TEST SET EVALUATION =====")
    print(f"MSE  : {mse:.3e}")
    print(f"MAE  : Ax {mae[0]:.3e} | Ay {mae[1]:.3e} | Az {mae[2]:.3e}")
    print(f"RMSE : Ax {rmse[0]:.3e} | Ay {rmse[1]:.3e} | Az {rmse[2]:.3e}")
    print(
        f"NRMSE: Ax {nrmse[0]*100:.2f}% | "
        f"Ay {nrmse[1]*100:.2f}% | "
        f"Az {nrmse[2]*100:.2f}%"
    )

    # ===== Plot error heatmaps =====
    y_true = np.zeros((n_samples, 3), dtype=np.float32)
    y_pred = np.zeros((n_samples, 3), dtype=np.float32)
    idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            batch_size = y.size(0)
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()
            y_true[idx:idx + batch_size, :] = y.cpu().numpy()
            y_pred[idx:idx + batch_size, :] = pred
            idx += batch_size
    plot_error_heatmaps_IxIy_by_Iz(y_true, y_pred)


def plot_error_heatmaps_IxIy_by_Iz(
    y_true,
    y_pred,
    n_bins_xy=40,
    n_slices_z=4,
):
    """
    Plot Ix–Iy heatmaps of 3D prediction error, sliced by Iz.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, 3)
        True currents (Ix, Iy, Iz)
    y_pred : np.ndarray, shape (N, 3)
        Predicted currents (Ix, Iy, Iz)
    n_bins_xy : int
        Number of bins for Ix and Iy
    n_slices_z : int
        Number of Iz slices
    """

    assert y_true.shape == y_pred.shape
    assert y_true.shape[1] == 3

    Ix, Iy, Iz = y_true[:, 0], y_true[:, 1], y_true[:, 2]

    # 3D Euclidean error
    error_3d = np.linalg.norm(y_pred - y_true, axis=1)

    # Iz slicing
    z_edges = np.linspace(Iz.min(), Iz.max(), n_slices_z + 1)

    fig, axes = plt.subplots(
        1, n_slices_z, figsize=(5 * n_slices_z, 4), sharey=True
    )

    if n_slices_z == 1:
        axes = [axes]

    for k in range(n_slices_z):
        z_min, z_max = z_edges[k], z_edges[k + 1]
        mask = (Iz >= z_min) & (Iz < z_max)

        if np.sum(mask) < 10:
            axes[k].set_title(f"Iz ∈ [{z_min:.2e}, {z_max:.2e}]\n(not enough data)")
            axes[k].axis("off")
            continue

        # 2D binning: mean error per (Ix, Iy) bin
        heatmap, xedges, yedges = np.histogram2d(
            Ix[mask],
            Iy[mask],
            bins=n_bins_xy,
            weights=error_3d[mask],
        )

        counts, _, _ = np.histogram2d(
            Ix[mask],
            Iy[mask],
            bins=n_bins_xy,
        )

        heatmap = np.divide(
            heatmap,
            counts,
            out=np.full_like(heatmap, np.nan),
            where=counts > 0,
        )

        im = axes[k].imshow(
            heatmap.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )

        axes[k].set_title(f"Iz ∈ [{z_min:.2e}, {z_max:.2e}]")
        axes[k].set_xlabel("Ix")

        if k == 0:
            axes[k].set_ylabel("Iy")

        fig.colorbar(im, ax=axes[k], label="||ΔI|| (3D error)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()