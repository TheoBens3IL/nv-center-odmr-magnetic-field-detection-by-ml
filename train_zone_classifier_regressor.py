import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataset import train_val_test_split
from utils import denormalize_labels, compute_zones_for_dataset
import models


class ODMRWithZones(Dataset):
    """
    Wrap a Subset[ODMRDatasetMultiConfig] to expose (signals, labels, zone_true).
    """

    def __init__(self, subset, zones_array=None):
        self.subset = subset
        self.zones_array = zones_array

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        base_idx = self.subset.indices[idx]
        signals, labels = self.subset.dataset[base_idx]
        if self.zones_array is not None:
            zone = int(self.zones_array[base_idx])
            return signals, labels, torch.tensor(zone, dtype=torch.long)
        return signals, labels


def train_zone_classifier_stage(
    dataset_path,
    train_base,
    val_base,
    test_base,
    zones,
    device,
    batch_size=64,
    epochs=200,
    lr=1e-4,
    weight_decay=1e-4,
    patience=20,
):
    """
    Train ZoneClassifier exactly as in train_zone_classifier.py.
    Returns the trained model and path where it is saved.
    """
    print("=" * 60)
    print("STAGE 1: ZONE CLASSIFIER TRAINING")
    print("=" * 60)

    train_set = ODMRWithZones(train_base, zones)
    val_set = ODMRWithZones(val_base, zones)
    test_set = ODMRWithZones(test_base, zones)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    n_classes = int(zones.max() + 1)
    model = models.ZoneClassifier(n_channels=10, n_freq=201, n_zones=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
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

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for signals, _, zones_batch in train_loader:
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
            for signals, _, zones_batch in val_loader:
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

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Classifier] Epoch [{epoch+1:03d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if early_stopping.step(val_loss, model):
            print(f"\n[Classifier] Early stopping at epoch {epoch+1}, best val loss = {early_stopping.best_loss:.4f}")
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, _, zones_batch in test_loader:
            signals = signals.to(device)
            zones_batch = zones_batch.to(device)
            logits = model(signals)
            preds = logits.argmax(dim=1)
            correct += (preds == zones_batch).sum().item()
            total += zones_batch.size(0)
    test_acc = correct / total if total > 0 else 0.0
    print(f"\n[Classifier] Test accuracy (zones): {test_acc*100:.2f}%")

    # Save only if test accuracy is best so far (tracked by a file)
    dataset_name = os.path.basename(dataset_path)
    model_dir = os.path.join("models_trained", dataset_name, "zoneclassifier")
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
        print(f"[Classifier] Best ZoneClassifier saved to: {model_path}")
        return model, model_path
    else:
        print(f"[Classifier] Model NOT saved (test accuracy not improved)")
        return model, None


def train_zone_regressor_stage(
    dataset_path,
    train_base,
    val_base,
    test_base,
    labels_mean,
    labels_std,
    classifier,
    device,
    batch_size=32,
    epochs=200,
    lr=1e-4,
    weight_decay=5e-4,
    patience=20,
):
    """
    Train ZoneAwareRegressor using zones predicted by a fixed classifier.
    This is a realistic two-stage training: regressor only ever sees
    classifier-predicted zone indices, not the ground-truth zones.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: ZONE-AWARE REGRESSOR TRAINING (USING PREDICTED ZONES)")
    print("=" * 60)

    # Base datasets (no zones injected here; classifier will infer them)
    train_set = ODMRWithZones(train_base, zones_array=None)
    val_set = ODMRWithZones(val_base, zones_array=None)
    test_set = ODMRWithZones(test_base, zones_array=None)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    classifier.eval()
    n_zones = classifier.classifier[-1].out_features  # last Linear has out_features = n_zones

    model = models.ZoneAwareRegressor(
        n_channels=10, n_freq=201, n_zones=n_zones, zone_emb_dim=32, output_dim=3
    ).to(device)

    criterion = nn.MSELoss()
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

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-5)
    best_val_mae = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = classifier(signals)
                zones_pred = logits.argmax(dim=1)

            optimizer.zero_grad()
            preds = model(signals, zones_pred)
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
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)

                logits = classifier(signals)
                zones_pred = logits.argmax(dim=1)
                preds = model(signals, zones_pred)
                loss = criterion(preds, labels)
                val_loss += loss.item() * signals.size(0)

                preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
                labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)
                abs_err_denorm += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
                n_samples += signals.size(0)

        val_loss /= len(val_loader.dataset)
        mae_denorm = (abs_err_denorm / n_samples).tolist()
        val_mae_mean = np.mean(mae_denorm)
        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_model_state = model.state_dict()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Regressor] Epoch [{epoch+1:03d}/{epochs}] | "
                f"Train Loss: {train_loss:.4e} | "
                f"Val Loss: {val_loss:.4e} | "
                f"Val MAE (Ax,Ay,Az): "
                f"({mae_denorm[0]:.4f}, {mae_denorm[1]:.4f}, {mae_denorm[2]:.4f})"
            )

        if early_stopping.step(val_loss, model):
            print(f"\n[Regressor] Early stopping at epoch {epoch+1}, best val loss = {early_stopping.best_loss:.4e}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test metrics (denormalized) using classifier-predicted zones
    model.eval()
    abs_err_denorm = torch.zeros(3, dtype=torch.float64)
    sq_err_denorm = torch.zeros(3, dtype=torch.float64)
    all_labels_denorm = []
    all_preds_denorm = []
    n_samples = 0

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            logits = classifier(signals)
            zones_pred = logits.argmax(dim=1)
            preds = model(signals, zones_pred)

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

    print("\n[Regressor] Test set metrics (two-stage, denormalized units):")
    print(f"  MAE  Ax/Ay/Az: {mae[0]:.4f} / {mae[1]:.4f} / {mae[2]:.4f} A")
    print(f"  RMSE Ax/Ay/Az: {rmse[0]:.4f} / {rmse[1]:.4f} / {rmse[2]:.4f} A")
    print(f"  R²   Ax/Ay/Az: {r2_ax:.3f} / {r2_ay:.3f} / {r2_az:.3f}")

    # Save only if test MAE is best so far (tracked by a file)
    test_mae_mean = np.mean(mae)
    dataset_name = os.path.basename(dataset_path)
    model_dir = os.path.join("models_trained", dataset_name, "zoneawareregressor_two_stage")
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
        print(f"\n[Regressor] Best ZoneAwareRegressor (two-stage) saved to: {model_path}")
    else:
        print(f"\n[Regressor] Model NOT saved (test MAE not improved)")


def main():
    parser = argparse.ArgumentParser(description="Two-stage training: ZoneClassifier then ZoneAwareRegressor using classifier predictions.")
    parser.add_argument("--dataset_dir", type=str, default="dataset_multi_mw_2",
                        help="Dataset directory name inside datasets_pytorch/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cls_epochs", type=int, default=200)
    parser.add_argument("--reg_epochs", type=int, default=200)
    parser.add_argument("--cls_lr", type=float, default=1e-3)
    parser.add_argument("--reg_lr", type=float, default=2e-4)
    parser.add_argument("--cls_weight_decay", type=float, default=1e-4)
    parser.add_argument("--reg_weight_decay", type=float, default=5e-4)
    parser.add_argument("--cls_patience", type=int, default=10)
    parser.add_argument("--reg_patience", type=int, default=20)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join("datasets_pytorch", args.dataset_dir)

    print("=" * 60)
    print("TWO-STAGE ZONE TRAINING")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Dataset:       {dataset_path}")
    print(f"Batch size:    {args.batch_size}")
    print("=" * 60)

    # Load base dataset splits
    train_base, val_base, test_base = train_val_test_split(dataset_path)

    # Compute zones and normalization stats once
    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)

    # Stage 1: classifier
    classifier, _ = train_zone_classifier_stage(
        dataset_path=dataset_path,
        train_base=train_base,
        val_base=val_base,
        test_base=test_base,
        zones=zones,
        device=device,
        batch_size=args.batch_size,
        epochs=args.cls_epochs,
        lr=args.cls_lr,
        weight_decay=args.cls_weight_decay,
        patience=args.cls_patience,
    )

    # Stage 2: regressor using classifier predictions
    train_zone_regressor_stage(
        dataset_path=dataset_path,
        train_base=train_base,
        val_base=val_base,
        test_base=test_base,
        labels_mean=labels_mean,
        labels_std=labels_std,
        classifier=classifier,
        device=device,
        batch_size=args.batch_size,
        epochs=args.reg_epochs,
        lr=args.reg_lr,
        weight_decay=args.reg_weight_decay,
        patience=args.reg_patience,
    )


if __name__ == "__main__":
    main()

