import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataset import train_val_test_split
from utils import denormalize_labels, compute_zones_for_dataset
import models


class ZoneRegSubset(Dataset):
    """
    Wrap a Subset[ODMRDatasetMultiConfig] to return (signals, labels, zone_index) for zone-aware regression.
    """
    def __init__(self, subset, zones_array):
        self.subset = subset
        self.zones_array = np.asarray(zones_array, dtype=np.int64)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        base_idx = self.subset.indices[idx]
        signals, labels = self.subset.dataset[base_idx]
        zone = int(self.zones_array[base_idx])
        return signals, labels, torch.tensor(zone, dtype=torch.long)


def train_zone_regressor(dataset_dir="dataset_multi_mw_2", batch_size=64, epochs=200, lr=2e-4, weight_decay=5e-4, patience=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join("datasets_pytorch", dataset_dir)

    print("=" * 60)
    print("ZONE-AWARE REGRESSOR TRAINING")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Dataset:       {dataset_path}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs (max):  {epochs}")
    print(f"LR:            {lr}")
    print(f"Weight decay:  {weight_decay}")
    print(f"Patience:      {patience}")
    print("=" * 60)

    # Compute zones and load normalization stats
    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)
    n_classes = int(zones.max() + 1)
    print(f"\nNumber of zone classes found: {n_classes}")

    # Base splits
    train_base, val_base, test_base = train_val_test_split(dataset_path)

    # Wrap with ZoneRegSubset
    train_set = ZoneRegSubset(train_base, zones)
    val_set = ZoneRegSubset(val_base, zones)
    test_set = ZoneRegSubset(test_base, zones)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nDataset sizes (experiments):")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    # Model
    model = models.ZoneAwareRegressor(n_channels=10, n_freq=201, n_zones=n_classes, zone_emb_dim=32, output_dim=3).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float("inf")
            self.counter = 0
            self.best_state = None

        def step(self, val_loss, model_):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.best_state = model_.state_dict()
            else:
                self.counter += 1
            return self.counter >= self.patience

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-5)

    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for signals, labels, zones_batch in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            zones_batch = zones_batch.to(device)

            optimizer.zero_grad()
            preds = model(signals, zones_batch)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * signals.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        abs_err_denorm = torch.zeros(3, dtype=torch.float64)
        n_samples = 0
        with torch.no_grad():
            for signals, labels, zones_batch in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                zones_batch = zones_batch.to(device)

                preds = model(signals, zones_batch)
                loss = criterion(preds, labels)
                val_loss += loss.item() * signals.size(0)

                preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
                labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)
                abs_err_denorm += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
                n_samples += signals.size(0)

        val_loss /= len(val_loader.dataset)
        mae_denorm = (abs_err_denorm / n_samples).tolist()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(mae_denorm)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:03d}/{epochs}] | "
                f"Train Loss: {train_loss:.4e} | "
                f"Val Loss: {val_loss:.4e} | "
                f"Val MAE (Ax,Ay,Az): "
                f"({mae_denorm[0]:.4f}, {mae_denorm[1]:.4f}, {mae_denorm[2]:.4f})"
            )

        if early_stopping.step(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch+1}, best val loss = {early_stopping.best_loss:.4e}")
            break

    # Restore best model
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    # Test metrics in physical units
    model.eval()
    abs_err_denorm = torch.zeros(3, dtype=torch.float64)
    sq_err_denorm = torch.zeros(3, dtype=torch.float64)
    all_labels_denorm = []
    all_preds_denorm = []
    n_samples = 0

    with torch.no_grad():
        for signals, labels, zones_batch in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            zones_batch = zones_batch.to(device)

            preds = model(signals, zones_batch)

            preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
            labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)

            abs_err_denorm += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
            sq_err_denorm += torch.sum((preds_denorm - labels_denorm) ** 2, dim=0)
            n_samples += signals.size(0)

            all_labels_denorm.append(labels_denorm.numpy())
            all_preds_denorm.append(preds_denorm.numpy())

    mae = (abs_err_denorm / n_samples).numpy()
    rmse = torch.sqrt(sq_err_denorm / n_samples).numpy()

    all_labels_denorm = np.concatenate(all_labels_denorm, axis=0)
    all_preds_denorm = np.concatenate(all_preds_denorm, axis=0)

    from sklearn.metrics import r2_score

    r2_ax = r2_score(all_labels_denorm[:, 0], all_preds_denorm[:, 0])
    r2_ay = r2_score(all_labels_denorm[:, 1], all_preds_denorm[:, 1])
    r2_az = r2_score(all_labels_denorm[:, 2], all_preds_denorm[:, 2])

    print("\nTest set metrics (zone-aware regressor, denormalized units):")
    print(f"  MAE  Ax/Ay/Az: {mae[0]:.4f} / {mae[1]:.4f} / {mae[2]:.4f} A")
    print(f"  RMSE Ax/Ay/Az: {rmse[0]:.4f} / {rmse[1]:.4f} / {rmse[2]:.4f} A")
    print(f"  R²   Ax/Ay/Az: {r2_ax:.3f} / {r2_ay:.3f} / {r2_az:.3f}")

    # Save only if test MAE is best so far (tracked by a file)
    test_mae_mean = np.mean(mae)
    model_dir = os.path.join("models_trained", dataset_dir, "zoneawareregressor")
    os.makedirs(model_dir, exist_ok=True)
    mae_record_path = os.path.join(model_dir, "best_test_mae.txt")
    save_model = True
    if os.path.exists(mae_record_path):
        with open(mae_record_path, "r") as f:
            try:
                best_prev_mae = float(f.read().strip())
                if test_mae_mean >= best_prev_mae:
                    save_model = False
            except Exception:
                pass
    if save_model:
        model_path = os.path.join(model_dir, "zoneawareregressor_best_model.pth")
        torch.save(model.state_dict(), model_path)
        with open(mae_record_path, "w") as f:
            f.write(str(test_mae_mean))
        print(f"\nBest ZoneAwareRegressor saved to: {model_path}")
    else:
        print(f"\nZoneAwareRegressor model NOT saved (test MAE not improved)")


def main():
    parser = argparse.ArgumentParser(description="Train zone-aware regressor on ODMR multi-MW dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset_multi_mw_2",
                        help="Dataset directory name inside datasets_pytorch/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)

    args = parser.parse_args()

    train_zone_regressor(dataset_dir=args.dataset_dir, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience)


if __name__ == "__main__":
    main()

