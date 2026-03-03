import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataset import train_val_test_split
from utils import compute_zones_for_dataset
import models


class ZoneSubset(Dataset):
    """Wrap a Subset[ODMRDatasetMultiConfig] to return (signals, zone_index)."""
    def __init__(self, subset, zones_array):
        self.subset = subset
        self.zones_array = np.asarray(zones_array, dtype=np.int64)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        base_idx = self.subset.indices[idx]
        signals, _ = self.subset.dataset[base_idx]
        zone = int(self.zones_array[base_idx])
        return signals, torch.tensor(zone, dtype=torch.long)


def train_zone_classifier(
    dataset_dir="dataset_multi_mw_2",
    batch_size=16,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join("datasets_pytorch", dataset_dir)

    print("=" * 60)
    print("ZONE CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Dataset:       {dataset_path}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs (max):  {epochs}")
    print(f"LR:            {lr}")
    print(f"Weight decay:  {weight_decay}")
    print(f"Patience:      {patience}")
    print("=" * 60)

    # Compute zones from labels
    zones, _ , _ = compute_zones_for_dataset(dataset_path)
    n_classes = int(zones.max() + 1)
    print(f"\nNumber of zone classes found: {n_classes}")

    # Base splits (multi_config=True → ODMRDatasetMultiConfig)
    train_base, val_base, test_base = train_val_test_split(dataset_path)

    # Wrap with ZoneSubset
    train_set = ZoneSubset(train_base, zones)
    val_set = ZoneSubset(val_base, zones)
    test_set = ZoneSubset(test_base, zones)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nDataset sizes (experiments):")
    print(f"  Train: {len(train_set)}")
    print(f"  Val:   {len(val_set)}")
    print(f"  Test:  {len(test_set)}")

    # Model
    model = models.ZoneClassifier(n_channels=10, n_freq=201, n_zones=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

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

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    best_model_state = None
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for signals, zones_batch in train_loader:
            signals = signals.to(device)
            zones_batch = zones_batch.to(device)

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, zones_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * signals.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, zones_batch in val_loader:
                signals = signals.to(device)
                zones_batch = zones_batch.to(device)
                logits = model(signals)
                loss = criterion(logits, zones_batch)
                val_loss += loss.item() * signals.size(0)

                preds = logits.argmax(dim=1)
                correct += (preds == zones_batch).sum().item()
                total += zones_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:03d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if early_stopping.step(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch+1}, best val loss = {early_stopping.best_loss:.4f}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, zones_batch in test_loader:
            signals = signals.to(device)
            zones_batch = zones_batch.to(device)
            logits = model(signals)
            preds = logits.argmax(dim=1)
            correct += (preds == zones_batch).sum().item()
            total += zones_batch.size(0)
    test_acc = correct / total if total > 0 else 0.0
    print(f"\nTest accuracy (zone classification): {test_acc*100:.2f}%")

    # Save only if test accuracy is best so far (tracked by a file)
    model_dir = os.path.join("models_trained", dataset_dir, "zoneclassifier")
    os.makedirs(model_dir, exist_ok=True)
    acc_record_path = os.path.join(model_dir, "best_test_acc.txt")
    save_model = True
    if os.path.exists(acc_record_path):
        with open(acc_record_path, "r") as f:
            try:
                best_prev_acc = float(f.read().strip())
                if test_acc <= best_prev_acc:
                    save_model = False
            except Exception:
                pass
    if save_model:
        model_path = os.path.join(model_dir, "zoneclassifier_best_model.pth")
        torch.save(model.state_dict(), model_path)
        with open(acc_record_path, "w") as f:
            f.write(str(test_acc))
        print(f"Best ZoneClassifier saved to: {model_path}")
    else:
        print(f"ZoneClassifier model NOT saved (test accuracy not improved)")


def main():
    parser = argparse.ArgumentParser(description="Train zone classifier on ODMR multi-MW dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset_multi_mw_2",
                        help="Dataset directory name inside datasets_pytorch/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    train_zone_classifier(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()

