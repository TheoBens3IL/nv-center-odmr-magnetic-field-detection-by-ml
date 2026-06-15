import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataset import stratified_zone_split
from utils import EarlyStopping, denormalize_labels, compute_zones_for_dataset
import models

# Wrapper to add zone indices to the base dataset splits (Subset[ODMRDatasetMultiConfig]) for zone-aware training 
class ZoneSubset(Dataset):
    def __init__(self, subset, zones_array, regression=True):
        self.subset = subset
        self.zones_array = zones_array
        self.regression = regression

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        base_idx = self.subset.indices[idx]
        signals, labels = self.subset.dataset[base_idx]
        zone = int(self.zones_array[base_idx])
        zone_tensor = torch.tensor(zone, dtype=torch.long)

        if self.regression:
            return signals, labels, zone_tensor
        else:
            return signals, zone_tensor


def train_classifier(dataset_dir, batch_size, epochs, lr, weight_decay, patience, synthetic=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join('datasets_pytorch', dataset_dir)

    # Set input shape based on dataset type
    if synthetic:
        n_channels, n_freq = 5, 400
    else:
        n_channels, n_freq = 10, 201

    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)
    print('DEBUG zones uniques:', np.unique(zones))
    print('DEBUG zone counts:', np.bincount(zones))
    n_classes = int(zones.max() + 1)

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
    print(f"Number of zones: {n_classes}")
    print("=" * 60)

    train_base, val_base, test_base = stratified_zone_split(dataset_path, synthetic=synthetic)

    train_set = ZoneSubset(train_base, zones, regression=False)
    val_set = ZoneSubset(val_base, zones, regression=False)
    test_set = ZoneSubset(test_base, zones, regression=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = models.ZoneClassifier(n_channels=n_channels, n_freq=n_freq, n_zones=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience)
    
    best_val_acc = 0.0
    best_model_state = None
    for epoch in range(epochs):
        # ===== Train ==== #
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

        # ===== Validation ===== #
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

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Classifier] Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if early_stopping.step(val_loss, model):
            print(f"\n[Classifier] Early stopping at epoch {epoch+1}, best val loss = {early_stopping.best_loss:.4f}")
            break
    
    # Restore best model of this training run
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # ===== Test eval ===== #
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

    return model, test_acc


def train_regressor(dataset_dir,batch_size, epochs, lr, weight_decay, patience, synthetic=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join('datasets_pytorch', dataset_dir)

    # Set input shape based on dataset type
    if synthetic:
        n_channels, n_freq = 5, 400
    else:
        n_channels, n_freq = 10, 201

    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)
    n_classes = int(zones.max() + 1)

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
    print(f"Number of zones: {n_classes}")
    print("=" * 60)

    train_base, val_base, test_base = stratified_zone_split(dataset_path, synthetic=synthetic)
    train_set = ZoneSubset(train_base, zones)
    val_set = ZoneSubset(val_base, zones)
    test_set = ZoneSubset(test_base, zones)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.ZoneAwareRegressor(n_channels=n_channels, n_freq=n_freq, n_zones=n_classes, zone_emb_dim=32, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-5)

    best_val_mae = float('inf')
    best_model_state = None
    # --- History for plotting ---
    history = {
        'train_loss': [],
        'val_loss': [],
        'mae_ax': [],
        'mae_ay': [],
        'mae_az': [],
        'physics_loss': [],  # Not used here, but for compatibility
    }
    for epoch in range(epochs):
        # ===== Train ==== #
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

        # ===== Validation ===== #
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
        val_mae_mean = np.mean(mae_denorm)

        # --- Update history ---
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae_ax'].append(mae_denorm[0])
        history['mae_ay'].append(mae_denorm[1])
        history['mae_az'].append(mae_denorm[2])
        history['physics_loss'].append(0.0)

        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_model_state = model.state_dict()

        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Regressor] Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | Val MAE (Ax,Ay,Az): ({mae_denorm[0]:.4f}, {mae_denorm[1]:.4f}, {mae_denorm[2]:.4f})")

        if early_stopping.step(val_mae_mean, model):
            print(f"\n[Regressor] Early stopping at epoch {epoch+1}, best val MAE = {early_stopping.best_loss:.4f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # --- Plot training history ---
    from utils import plot_training_history
    fig = plot_training_history(history, label_names=['Ax', 'Ay', 'Az'], show=True)
    fig.savefig(os.path.join("models_trained", dataset_dir, "zoneawareregressor", "training_history.png"), dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to models_trained/{dataset_dir}/zoneawareregressor/training_history.png")

    # ===== Test eval ===== #
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
    print(f"\nTest set metrics (zone-aware regressor, denormalized units):")
    print(f"  MAE  Ax/Ay/Az: {mae[0]:.4f} / {mae[1]:.4f} / {mae[2]:.4f} A")
    print(f"  RMSE Ax/Ay/Az: {rmse[0]:.4f} / {rmse[1]:.4f} / {rmse[2]:.4f} A")

    # Save only if test MAE is best so far
    test_mae_mean = np.mean(mae)
    dataset_name = os.path.basename(dataset_path)
    model_dir = os.path.join("models_trained", dataset_name, "zoneawareregressor")
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
        print(f"Best ZoneAwareRegressor saved to: {model_path}")
    else:
        print(f"ZoneAwareRegressor model NOT saved (test MAE not improved)")


def train_two_stage(dataset_dir, batch_size, cls_epochs, reg_epochs, cls_lr, reg_lr, cls_weight_decay, reg_weight_decay, cls_patience, reg_patience, pretrained_classifier=None, synthetic=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join('datasets_pytorch', dataset_dir)
    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)

    print("=" * 60)
    print("TWO-STAGE TRAINING (ZoneAwareTwoStage: classifier + regressor)")
    print("=" * 60)

    # ================= DATA ================= #
    train_base, val_base, test_base = stratified_zone_split(dataset_path, synthetic=synthetic)

    train_set = ZoneSubset(train_base, zones_array=zones)
    val_set = ZoneSubset(val_base, zones_array=zones)
    test_set = ZoneSubset(test_base, zones_array=zones)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # ================= MODEL ================= #
    n_zones = int(zones.max() + 1)
    # Set input shape based on dataset type
    if synthetic:
        n_channels, n_freq = 5, 400
    else:
        n_channels, n_freq = 10, 201
    model = models.ZoneAwareTwoStage(n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, zone_emb_dim=32, output_dim=3).to(device)
    if pretrained_classifier:
        classifier_path = os.path.join("models_trained", dataset_dir, "zoneclassifier", "zoneclassifier_best_model.pth")
        classifier_state = torch.load(classifier_path, map_location=device)
        model.classifier.load_state_dict(classifier_state)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # ==========================================================
    # STAGE 1 : CLASSIFIER
    # ==========================================================
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=cls_lr, weight_decay=cls_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=cls_patience)

    best_val_acc = 0.0
    best_cls_state = None
    # --- History for plotting ---
    history = {
        'train_loss': [],
        'val_loss': [],
        'mae_ax': [],
        'mae_ay': [],
        'mae_az': [],
        'physics_loss': [],
    }
    for epoch in range(cls_epochs):
        # Train classifier
        model.train()
        train_loss = 0.0
        for signals, labels, zones_true in train_loader:
            signals = signals.to(device)
            zones_true = zones_true.to(device)
            optimizer.zero_grad()
            logits = model.forward_classifier(signals)
            loss = criterion_cls(logits, zones_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * signals.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation classifier
        model.eval()
        val_loss = 0.0
        abs_err_denorm = torch.zeros(3, dtype=torch.float64)
        n_samples = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels, zones_true in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                zones_true = zones_true.to(device)
                logits = model.forward_classifier(signals)
                loss = criterion_cls(logits, zones_true)
                val_loss += loss.item() * signals.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == zones_true).sum().item()
                total += zones_true.size(0)
                # For plotting, get regressor MAE (dummy, zeros)
                abs_err_denorm += torch.zeros(3)
                n_samples += signals.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0
        # For classifier stage, MAE is zeros
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae_ax'].append(0.0)
        history['mae_ay'].append(0.0)
        history['mae_az'].append(0.0)
        history['physics_loss'].append(0.0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[TwoStage - Classifier] Epoch [{epoch+1:03d}/{cls_epochs}] | Val Accuracy: {val_acc*100:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cls_state = model.state_dict()
        
        if early_stopping.step(val_loss, model):
            print(f"\n[TwoStage - Classifier] Early stopping at epoch {epoch+1} during classifier training, best val acc = {early_stopping.best_loss:.4f}")
            break

    # Test classifier
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels, zones_true in test_loader:
            signals = signals.to(device)
            zones_true = zones_true.to(device)
            logits = model.forward_classifier(signals)
            preds = logits.argmax(dim=1)
            correct += (preds == zones_true).sum().item()
            total += zones_true.size(0)
    test_acc = correct / total if total > 0 else 0.0
    print(f"\n[TwoStage - Classifier] Test Accuracy: {test_acc*100:.2f}%\n")

    # Restore best classifier state before next stage
    if best_cls_state is not None:
        model.load_state_dict(best_cls_state)

    # ==========================================================
    # STAGE 2 : REGRESSOR
    # ==========================================================

    # Freeze classifier parameters
    for param in model.classifier.parameters():
        param.requires_grad = False

    # Only optimize regressor parameters now
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=reg_lr, weight_decay=reg_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=reg_patience)
    best_val_mae = float('inf')
    best_model_state = None

    for epoch in range(reg_epochs):
        # Train regressor (using predicted zones from classifier)
        model.train()
        train_loss = 0.0
        for signals, labels, zones_true in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            zones_true = zones_true.to(device)

            optimizer.zero_grad()
            model.classifier.eval()  # freeze classifier behavior
            with torch.no_grad():
                logits = model.forward_classifier(signals)
                zones_pred = logits.argmax(dim=1)  # predicted zones
            
            preds = model.forward_regressor(signals, zones_pred)
            loss = criterion_reg(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * signals.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation Regressor (using predicted zones from classifier)
        model.eval()
        val_loss = 0.0
        abs_err_denorm = torch.zeros(3, dtype=torch.float64)
        n_samples = 0

        with torch.no_grad():
            for signals, labels, zones_true in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                zones_true = zones_true.to(device)

                logits = model.forward_classifier(signals)
                zones_pred = logits.argmax(dim=1)  # predicted zones

                preds = model.forward_regressor(signals, zones_pred)
                loss = criterion_reg(preds, labels)
                val_loss += loss.item() * signals.size(0)

                # denormalized MAE
                preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
                labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)
                abs_err_denorm += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
                n_samples += signals.size(0)

        val_loss /= len(val_loader.dataset)
        mae_denorm = (abs_err_denorm / n_samples).tolist()
        val_mae_mean = np.mean(mae_denorm)

        # Save best model based on MAE
        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_model_state = model.state_dict()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[TwoStage - Regressor] Epoch [{epoch+1:03d}/{reg_epochs}] | Val MAE (Ax,Ay,Az): ({mae_denorm[0]:.4f}, {mae_denorm[1]:.4f}, {mae_denorm[2]:.4f})\n")                

        if early_stopping.step(val_mae_mean, model):
            print(f"[TwoStage - Regressor] Early stopping at epoch {epoch+1}, best val MAE = {early_stopping.best_loss:.4f}")
            break

    # Restore best model before test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ===== Test evaluation ===== #
    model.eval()
    abs_err_denorm = torch.zeros(3, dtype=torch.float64)
    sq_err_denorm = torch.zeros(3, dtype=torch.float64)
    n_samples = 0
    all_labels_denorm = []
    all_preds_denorm = []
    correct_cls_test = 0
    total_cls_test = 0

    with torch.no_grad():
        for signals, labels, zones_true in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            zones_true = zones_true.to(device)

            logits = model.forward_classifier(signals)
            zones_pred = logits.argmax(dim=1)

            preds = model.forward_regressor(signals, zones_pred)

            preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
            labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)
            abs_err_denorm += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
            sq_err_denorm += torch.sum((preds_denorm - labels_denorm) ** 2, dim=0)
            n_samples += signals.size(0)
            all_labels_denorm.append(labels_denorm.numpy())
            all_preds_denorm.append(preds_denorm.numpy())

            # classification accuracy
            correct_cls_test += (zones_pred == zones_true).sum().item()
            total_cls_test += zones_true.size(0)

    mae = (abs_err_denorm / n_samples).numpy()
    rmse = torch.sqrt(sq_err_denorm / n_samples).numpy()
    cls_acc_test = correct_cls_test / total_cls_test if total_cls_test > 0 else 0.0

    print(f"\n[TwoStage - Regressor] Test set metrics (denormalized units):")
    print(f"  MAE Ax/Ay/Az: {mae[0]:.4f} / {mae[1]:.4f} / {mae[2]:.4f} A")
    print(f"  RMSE Ax/Ay/Az: {rmse[0]:.4f} / {rmse[1]:.4f} / {rmse[2]:.4f} A")
    print(f"  Classifier accuracy on test set: {cls_acc_test*100:.2f}%")

    # ===== Save model only if test MAE improves ===== #
    test_mae_mean = np.mean(mae)
    model_dir = os.path.join("models_trained", os.path.basename(dataset_path), "zoneawaretwostage")
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
        model_path = os.path.join(model_dir, "zoneawaretwostage_best_model.pth")
        torch.save(model.state_dict(), model_path)
        with open(mae_record_path, "w") as f:
            f.write(str(test_mae_mean))
        print(f"Best ZoneAwareTwoStage model saved to: {model_path}")
    else:
        print("ZoneAwareTwoStage model NOT saved (test MAE not improved)")


def train_two_stage_joint(dataset_dir, batch_size, epochs, cls_lr, reg_lr, cls_weight_decay, reg_weight_decay, patience, lambda_reg=1.0, synthetic=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join('datasets_pytorch', dataset_dir)
    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)

    print("="*60)
    print("JOINT TRAINING (ZoneAwareTwoStage: classifier + regressor)")
    print("="*60)

    # ===== DATA ===== #
    train_base, val_base, test_base = stratified_zone_split(dataset_path, synthetic=synthetic)

    train_set = ZoneSubset(train_base, zones_array=zones)
    val_set = ZoneSubset(val_base, zones_array=zones)
    test_set = ZoneSubset(test_base, zones_array=zones)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ===== MODEL ===== #
    n_zones = int(zones.max() + 1)
    # Set input shape based on dataset type
    if synthetic:
        n_channels, n_freq = 5, 400
    else:
        n_channels, n_freq = 10, 201
    model = models.ZoneAwareTwoStage(n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, zone_emb_dim=32, output_dim=3).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': cls_lr, 'weight_decay': cls_weight_decay},
        {'params': model.regressor.parameters(), 'lr': reg_lr, 'weight_decay': reg_weight_decay}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    best_val_mae = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # ===== TRAIN ===== #
        model.train()
        train_loss = 0.0
        cls_correct = 0
        cls_total = 0

        for signals, labels, zones_true in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            zones_true = zones_true.to(device)

            optimizer.zero_grad()
            # Classifier forward + loss
            logits = model.forward_classifier(signals)
            zones_pred = logits.argmax(dim=1)
            loss_cls = criterion_cls(logits, zones_true)
            # Regressor forward + loss (using predicted zones)
            preds = model.forward_regressor(signals, zones_pred)
            loss_reg = criterion_reg(preds, labels)

            loss = loss_cls + lambda_reg * loss_reg
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * signals.size(0)
            cls_correct += (zones_pred == zones_true).sum().item()
            cls_total += zones_true.size(0)

        train_loss /= len(train_loader.dataset)
        train_cls_acc = cls_correct / cls_total

        # ===== VALIDATION ===== #
        model.eval()
        val_loss = 0.0
        abs_err_denorm = torch.zeros(3, dtype=torch.float64)
        n_samples = 0
        cls_correct_val = 0
        cls_total_val = 0

        with torch.no_grad():
            for signals, labels, zones_true in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                zones_true = zones_true.to(device)

                logits = model.forward_classifier(signals)
                zones_pred = logits.argmax(dim=1)

                loss_cls = criterion_cls(logits, zones_true)
                preds = model.forward_regressor(signals, zones_pred)
                loss_reg = criterion_reg(preds, labels)

                loss_total = loss_cls + lambda_reg * loss_reg
                val_loss += loss_total.item() * signals.size(0)

                # classification metrics
                cls_correct_val += (zones_pred == zones_true).sum().item()
                cls_total_val += zones_true.size(0)

                # denormalized regression MAE
                preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
                labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)
                abs_err_denorm += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
                n_samples += signals.size(0)

        val_loss /= len(val_loader.dataset)
        cls_acc_val = cls_correct_val / cls_total_val
        mae_denorm = (abs_err_denorm / n_samples).tolist()
        val_mae_mean = np.mean(mae_denorm)

        # Save best model based on MAE
        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_model_state = model.state_dict()

        scheduler.step(val_loss)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"[TwoStageJoint] Epoch [{epoch+1}/{epochs}] | "
                  f"Val MAE (Ax,Ay,Az): ({mae_denorm[0]:.4f}, {mae_denorm[1]:.4f}, {mae_denorm[2]:.4f}) | "
                  f"Val Classifier Acc: {cls_acc_val*100:.2f}%")

        if early_stopping.step(val_mae_mean, model):
            print(f"\n[TwoStageJoint] Early stopping at epoch {epoch+1}, best val MAE = {early_stopping.best_loss:.4f}")
            break

    # ===== RESTORE BEST MODEL ===== #
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ===== TEST EVALUATION ===== #
    model.eval()
    abs_err_denorm = torch.zeros(3, dtype=torch.float64)
    sq_err_denorm = torch.zeros(3, dtype=torch.float64)
    n_samples = 0
    cls_correct_test = 0
    cls_total_test = 0

    with torch.no_grad():
        for signals, labels, zones_true in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            zones_true = zones_true.to(device)

            logits = model.forward_classifier(signals)
            zones_pred = logits.argmax(dim=1)

            preds = model.forward_regressor(signals, zones_pred)

            preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
            labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)

            abs_err_denorm += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
            sq_err_denorm += torch.sum((preds_denorm - labels_denorm)**2, dim=0)
            n_samples += signals.size(0)

            cls_correct_test += (zones_pred == zones_true).sum().item()
            cls_total_test += zones_true.size(0)

    mae = (abs_err_denorm / n_samples).numpy()
    rmse = torch.sqrt(sq_err_denorm / n_samples).numpy()
    cls_acc_test = cls_correct_test / cls_total_test if cls_total_test > 0 else 0.0

    print(f"\n[TwoStageJoint] Test set metrics (denormalized units):")
    print(f"  MAE Ax/Ay/Az: {mae[0]:.4f} / {mae[1]:.4f} / {mae[2]:.4f} A")
    print(f"  RMSE Ax/Ay/Az: {rmse[0]:.4f} / {rmse[1]:.4f} / {rmse[2]:.4f} A")
    print(f"  Classifier accuracy on test set: {cls_acc_test*100:.2f}%")

    # ===== SAVE BEST MODEL BASED ON TEST MAE ===== #
    test_mae_mean = np.mean(mae)
    print(f"\n[TwoStageJoint] Test MAE (mean over Ax/Ay/Az): {test_mae_mean:.4f} A")
    model_dir = os.path.join("models_trained", os.path.basename(dataset_path), "zoneawaretwostage_joint")
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
        model_path = os.path.join(model_dir, "zoneawaretwostage_joint_best_model.pth")
        torch.save(model.state_dict(), model_path)
        with open(mae_record_path, "w") as f:
            f.write(str(test_mae_mean))
        print(f"Best ZoneAwareTwoStage joint model saved to: {model_path}")
    else:
        print("ZoneAwareTwoStage joint model NOT saved (test MAE not improved)")

        
def main():
    parser = argparse.ArgumentParser(description="Unified zone model training script")
    parser.add_argument('--mode', choices=['classifier', 'regressor', 'two-stage', 'two-stage-joint'], required=True)
    parser.add_argument('--dataset_dir', type=str, default='dataset_multi_mw_2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    # For two-stage
    parser.add_argument('--cls_epochs', type=int, default=200)
    parser.add_argument('--reg_epochs', type=int, default=200)
    parser.add_argument('--cls_lr', type=float, default=1e-3)
    parser.add_argument('--reg_lr', type=float, default=2e-4)
    parser.add_argument('--cls_weight_decay', type=float, default=1e-4)
    parser.add_argument('--reg_weight_decay', type=float, default=5e-4)
    parser.add_argument('--pretrained_classifier', type=str, default=False)
    parser.add_argument('--cls_patience', type=int, default=20)
    parser.add_argument('--reg_patience', type=int, default=20)
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic dataset (5 MW configs, 400 freq)')
    args = parser.parse_args()

    # Add --synthetic flag to all modes
    synthetic = getattr(args, 'synthetic', False)

    if args.mode == 'classifier':
        train_classifier(args.dataset_dir, args.batch_size, args.epochs, args.lr, args.weight_decay, args.patience, synthetic=synthetic)
    elif args.mode == 'regressor':
        train_regressor(args.dataset_dir, args.batch_size, args.epochs, args.lr, args.weight_decay, args.patience, synthetic=synthetic)
    elif args.mode == 'two-stage':
        train_two_stage(args.dataset_dir, args.batch_size, args.cls_epochs, args.reg_epochs, args.cls_lr, args.reg_lr, args.cls_weight_decay, args.reg_weight_decay, args.cls_patience, args.reg_patience, args.pretrained_classifier, synthetic=synthetic)
    elif args.mode == 'two-stage-joint':
        train_two_stage_joint(args.dataset_dir, args.batch_size, args.epochs, args.cls_lr, args.reg_lr, args.cls_weight_decay, args.reg_weight_decay, args.patience,
                              lambda_reg=1.0, synthetic=synthetic)

if __name__ == '__main__':
    main()
