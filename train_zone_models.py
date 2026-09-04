import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataset import stratified_zone_split, resolve_mw_indices, detect_num_mw_configs, detect_n_freq, get_frequency_axis, resolve_dataset_path
from utils import EarlyStopping, denormalize_labels, compute_zones_for_dataset, get_model_output_dir, try_save_best_checkpoint, print_training_timing, format_duration
from physics_informed import extract_measured_peaks_batch, physics_loss_from_current, CURRENT_TO_FIELD_MT_PER_A
import models

def zone_arch_config(deep=False):
    """Return model classes and checkpoint/dir names for v1 or deep zone architectures."""
    if deep:
        return {
            'classifier_cls': models.ZoneClassifier2,
            'regressor_cls': models.ZoneAwareRegressor2,
            'twostage_cls': models.ZoneAwareTwoStageJointDeep,
            'classifier_dir': 'zoneclassifier2',
            'regressor_dir': 'zoneawareregressor2',
            'twostage_dir': 'zoneawaretwostage2',
            'joint_dir': 'zoneawaretwostagejointdeep',
            'classifier_ckpt': 'zoneclassifier2_best_model.pth',
            'regressor_ckpt': 'zoneawareregressor2_best_model.pth',
            'twostage_ckpt': 'zoneawaretwostage2_best_model.pth',
            'joint_ckpt': 'zoneawaretwostagejointdeep_best_model.pth',
            'arch_label': 'deep',
        }
    return {
        'classifier_cls': models.ZoneClassifier,
        'regressor_cls': models.ZoneAwareRegressor,
        'twostage_cls': models.ZoneAwareTwoStage,
        'classifier_dir': 'zoneclassifier',
        'regressor_dir': 'zoneawareregressor',
        'twostage_dir': 'zoneawaretwostage',
        'joint_dir': 'zoneawaretwostagejoint',
        'classifier_ckpt': 'zoneclassifier_best_model.pth',
        'regressor_ckpt': 'zoneawareregressor_best_model.pth',
        'twostage_ckpt': 'zoneawaretwostage_best_model.pth',
        'joint_ckpt': 'zoneawaretwostagejoint_best_model.pth',
        'arch_label': 'v1',
    }

# Wrapper to add zone indices to the base dataset splits (Subset[ODMRDataset]) for zone-aware training
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


def train_classifier(dataset_dir, batch_size, epochs, lr, weight_decay, patience, synthetic=False, mw_configs=None,
                     val_size=0.10, test_size=0.10, balanced_val=True, val_samples_per_zone=None,
                     balanced_test=False, test_samples_per_zone=1, deep=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = resolve_dataset_path(dataset_dir)
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    mw_indices = resolve_mw_indices(synthetic=synthetic, mw_configs=mw_configs, dataset_dir=dataset_path)

    n_freq = detect_n_freq(dataset_path, synthetic)
    n_channels = len(mw_indices)

    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)
    n_classes = int(zones.max() + 1)
    arch = zone_arch_config(deep)

    print("=" * 60)
    print(f"ZONE CLASSIFIER TRAINING ({arch['arch_label']})")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Dataset:       {dataset_path}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs (max):  {epochs}")
    print(f"LR:            {lr}")
    print(f"Weight decay:  {weight_decay}")
    print(f"Patience:      {patience}")
    print(f"Number of zones: {n_classes}")
    print(f"MW configs:    {mw_indices} ({len(mw_indices)}/{detect_num_mw_configs(dataset_path, synthetic)} channels)")
    print("=" * 60)

    train_base, val_base, test_base = stratified_zone_split(
        dataset_path, synthetic=synthetic, mw_indices=mw_indices,
        val_size=val_size, test_size=test_size,
        balanced_val=balanced_val, val_samples_per_zone=val_samples_per_zone,
        balanced_test=balanced_test, test_samples_per_zone=test_samples_per_zone,
    )

    train_set = ZoneSubset(train_base, zones, regression=False)
    val_set = ZoneSubset(val_base, zones, regression=False)
    test_set = ZoneSubset(test_base, zones, regression=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = arch['classifier_cls'](n_channels=n_channels, n_freq=n_freq, n_zones=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience, mode='max')

    best_val_acc = 0.0
    best_model_state = None
    t_train_start = time.perf_counter()
    epochs_completed = 0
    for epoch in range(epochs):
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

        epochs_completed = epoch + 1
        if early_stopping.step(val_acc, model):
            print(f"\n[Classifier] Early stopping at epoch {epoch+1}, best val acc = {early_stopping.best_metric*100:.2f}%")
            break

    print_training_timing("Classifier", t_train_start, epochs_completed, epochs)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

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

    for per_mw_count in (False, True):
        model_dir = get_model_output_dir(dataset_name, arch['classifier_dir'], mw_indices, per_mw_count=per_mw_count)
        try_save_best_checkpoint(
            model, model_dir, arch['classifier_ckpt'], "best_test_acc.txt",
            test_acc, higher_is_better=True,
        )

    return model, test_acc


def train_regressor(dataset_dir, batch_size, epochs, lr, weight_decay, patience, synthetic=False, mw_configs=None,
                    val_size=0.10, test_size=0.10, balanced_val=True, val_samples_per_zone=None,
                    balanced_test=False, test_samples_per_zone=1, deep=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = resolve_dataset_path(dataset_dir)
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    mw_indices = resolve_mw_indices(synthetic=synthetic, mw_configs=mw_configs, dataset_dir=dataset_path)

    n_freq = detect_n_freq(dataset_path, synthetic)
    n_channels = len(mw_indices)

    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)
    n_classes = int(zones.max() + 1)
    arch = zone_arch_config(deep)

    print("=" * 60)
    print(f"ZONE-AWARE REGRESSOR TRAINING ({arch['arch_label']})")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Dataset:       {dataset_path}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs (max):  {epochs}")
    print(f"LR:            {lr}")
    print(f"Weight decay:  {weight_decay}")
    print(f"Patience:      {patience}")
    print(f"Number of zones: {n_classes}")
    print(f"MW configs:    {mw_indices} ({len(mw_indices)}/{detect_num_mw_configs(dataset_path, synthetic)} channels)")
    print("=" * 60)

    train_base, val_base, test_base = stratified_zone_split(
        dataset_path, synthetic=synthetic, mw_indices=mw_indices,
        val_size=val_size, test_size=test_size,
        balanced_val=balanced_val, val_samples_per_zone=val_samples_per_zone,
        balanced_test=balanced_test, test_samples_per_zone=test_samples_per_zone,
    )
    train_set = ZoneSubset(train_base, zones)
    val_set = ZoneSubset(val_base, zones)
    test_set = ZoneSubset(test_base, zones)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = arch['regressor_cls'](n_channels=n_channels, n_freq=n_freq, n_zones=n_classes, zone_emb_dim=32, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-5)

    best_val_mae = float('inf')
    best_model_state = None
    history = {
        'train_loss': [],
        'val_loss': [],
        'mae_ax': [],
        'mae_ay': [],
        'mae_az': [],
        'physics_loss': [],
    }
    t_train_start = time.perf_counter()
    epochs_completed = 0
    for epoch in range(epochs):
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

        epochs_completed = epoch + 1
        if early_stopping.step(val_mae_mean, model):
            print(f"\n[Regressor] Early stopping at epoch {epoch+1}, best val MAE = {early_stopping.best_loss:.4f}")
            break

    print_training_timing("Regressor", t_train_start, epochs_completed, epochs)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    from utils import plot_training_history
    fig = plot_training_history(history, label_names=['Ax', 'Ay', 'Az'], show=True)
    fig.savefig(os.path.join("models_trained", dataset_name, arch['regressor_dir'], "training_history.png"), dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to models_trained/{dataset_name}/{arch['regressor_dir']}/training_history.png")

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

    test_mae_mean = np.mean(mae)
    for per_mw_count in (False, True):
        model_dir = get_model_output_dir(dataset_name, arch['regressor_dir'], mw_indices, per_mw_count=per_mw_count)
        try_save_best_checkpoint(
            model, model_dir, arch['regressor_ckpt'], "best_test_mae.txt",
            test_mae_mean, higher_is_better=False,
        )


def train_two_stage(dataset_dir, batch_size, cls_epochs, reg_epochs, cls_lr, reg_lr, cls_weight_decay, reg_weight_decay, cls_patience, reg_patience, pretrained_classifier=None, synthetic=False, mw_configs=None,
                    val_size=0.10, test_size=0.10, balanced_val=True, val_samples_per_zone=None,
                    balanced_test=False, test_samples_per_zone=1,
                    use_physics_loss=False, physics_loss_weight=0.1, deep=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = resolve_dataset_path(dataset_dir)
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)
    mw_indices = resolve_mw_indices(synthetic=synthetic, mw_configs=mw_configs, dataset_dir=dataset_path)
    arch = zone_arch_config(deep)

    print("=" * 60)
    print(f"TWO-STAGE TRAINING (ZoneAwareTwoStage {arch['arch_label']}: classifier + regressor)")
    print("=" * 60)
    print(f"MW configs:    {mw_indices} ({len(mw_indices)}/{detect_num_mw_configs(dataset_path, synthetic)} channels)")
    if use_physics_loss:
        print(f"Physics-informed regressor: 1 A -> {CURRENT_TO_FIELD_MT_PER_A} mT (weight={physics_loss_weight})")

    train_base, val_base, test_base = stratified_zone_split(
        dataset_path, synthetic=synthetic, mw_indices=mw_indices,
        val_size=val_size, test_size=test_size,
        balanced_val=balanced_val, val_samples_per_zone=val_samples_per_zone,
        balanced_test=balanced_test, test_samples_per_zone=test_samples_per_zone,
    )

    train_set = ZoneSubset(train_base, zones_array=zones)
    val_set = ZoneSubset(val_base, zones_array=zones)
    test_set = ZoneSubset(test_base, zones_array=zones)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    n_zones = int(zones.max() + 1)
    n_freq = detect_n_freq(dataset_path, synthetic)
    n_channels = len(mw_indices)
    model = arch['twostage_cls'](n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, zone_emb_dim=32, output_dim=3).to(device)
    if pretrained_classifier:
        classifier_path = os.path.join("models_trained", dataset_name, arch['classifier_dir'], arch['classifier_ckpt'])
        classifier_state = torch.load(classifier_path, map_location=device)
        model.classifier.load_state_dict(classifier_state)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # ==========================================================
    # STAGE 1 : CLASSIFIER
    # ==========================================================
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=cls_lr, weight_decay=cls_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=cls_patience, mode='max')

    best_val_acc = 0.0
    best_cls_state = None
    history = {
        'train_loss': [],
        'val_loss': [],
        'mae_ax': [],
        'mae_ay': [],
        'mae_az': [],
        'physics_loss': [],
    }
    t_run_start = time.perf_counter()
    t_cls_start = time.perf_counter()
    cls_epochs_completed = 0
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

                abs_err_denorm += torch.zeros(3)
                n_samples += signals.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0

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

        cls_epochs_completed = epoch + 1
        if early_stopping.step(val_acc, model):
            print(f"\n[TwoStage - Classifier] Early stopping at epoch {epoch+1}, best val acc = {early_stopping.best_metric*100:.2f}%")
            break

    print_training_timing("TwoStage - Classifier", t_cls_start, cls_epochs_completed, cls_epochs)

    if best_cls_state is not None:
        model.load_state_dict(best_cls_state)
    elif early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

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
    print(f"\n[TwoStage - Classifier] Test Accuracy: {test_acc*100:.2f}% (best val acc: {best_val_acc*100:.2f}%)\n")

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
    freq_axis_Hz = get_frequency_axis(dataset_path)

    t_reg_start = time.perf_counter()
    reg_epochs_completed = 0
    for epoch in range(reg_epochs):
        # Train regressor (using predicted zones from classifier)
        model.train()
        train_loss = 0.0
        physics_loss_accum = 0.0
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
            if use_physics_loss:
                measured_freqs = extract_measured_peaks_batch(signals, freq_axis_Hz, num_peaks=8)
                preds_denorm = denormalize_labels(preds, labels_mean, labels_std)
                labels_denorm = denormalize_labels(labels, labels_mean, labels_std)
                loss_phys = physics_loss_from_current(
                    preds_denorm, measured_freqs, current_true=labels_denorm,
                )
                total_loss = loss + physics_loss_weight * loss_phys
                physics_loss_accum += loss_phys.item() * signals.size(0)
            else:
                total_loss = loss
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item() * signals.size(0)
        train_loss /= len(train_loader.dataset)
        physics_loss_val = physics_loss_accum / len(train_loader.dataset) if use_physics_loss else 0.0

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

        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_model_state = model.state_dict()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            phys_str = f" | Physics_loss: {physics_loss_val:.4f} (norm.)" if use_physics_loss else ""
            print(f"[TwoStage - Regressor] Epoch [{epoch+1:03d}/{reg_epochs}] | Val MAE (Ax,Ay,Az): ({mae_denorm[0]:.4f}, {mae_denorm[1]:.4f}, {mae_denorm[2]:.4f}){phys_str}\n")

        reg_epochs_completed = epoch + 1
        if early_stopping.step(val_mae_mean, model):
            print(f"[TwoStage - Regressor] Early stopping at epoch {epoch+1}, best val MAE = {early_stopping.best_loss:.4f}")
            break

    print_training_timing("TwoStage - Regressor", t_reg_start, reg_epochs_completed, reg_epochs)
    print(f"[TwoStage] Total training time: {format_duration(time.perf_counter() - t_run_start)}")

    # Restore best model before test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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

    test_mae_mean = np.mean(mae)
    for per_mw_count in (False, True):
        model_dir = get_model_output_dir(dataset_name, arch['twostage_dir'], mw_indices, per_mw_count=per_mw_count)
        try_save_best_checkpoint(
            model, model_dir, arch['twostage_ckpt'], "best_test_mae.txt",
            test_mae_mean, higher_is_better=False,
        )


def train_two_stage_joint(dataset_dir, batch_size, epochs, cls_lr, reg_lr, cls_weight_decay, reg_weight_decay, patience, lambda_reg=1.0, synthetic=False, mw_configs=None,
                          val_size=0.10, test_size=0.10, balanced_val=True, val_samples_per_zone=None,
                          balanced_test=False, test_samples_per_zone=1,
                          use_physics_loss=False, physics_loss_weight=0.1, deep=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = resolve_dataset_path(dataset_dir)
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)
    mw_indices = resolve_mw_indices(synthetic=synthetic, mw_configs=mw_configs, dataset_dir=dataset_path)
    arch = zone_arch_config(deep)

    print("="*60)
    print(f"JOINT TRAINING (ZoneAwareTwoStage {arch['arch_label']}: classifier + regressor)")
    print("="*60)
    print(f"MW configs:    {mw_indices} ({len(mw_indices)}/{detect_num_mw_configs(dataset_path, synthetic)} channels)")
    if use_physics_loss:
        print(f"Physics-informed joint training: 1 A -> {CURRENT_TO_FIELD_MT_PER_A} mT (weight={physics_loss_weight})")

    # ===== DATA ===== #
    train_base, val_base, test_base = stratified_zone_split(
        dataset_path, synthetic=synthetic, mw_indices=mw_indices,
        val_size=val_size, test_size=test_size,
        balanced_val=balanced_val, val_samples_per_zone=val_samples_per_zone,
        balanced_test=balanced_test, test_samples_per_zone=test_samples_per_zone,
    )

    train_set = ZoneSubset(train_base, zones_array=zones)
    val_set = ZoneSubset(val_base, zones_array=zones)
    test_set = ZoneSubset(test_base, zones_array=zones)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ===== MODEL ===== #
    n_zones = int(zones.max() + 1)
    n_freq = detect_n_freq(dataset_path, synthetic)
    n_channels = len(mw_indices)
    model = arch['twostage_cls'](n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, zone_emb_dim=32, output_dim=3).to(device)

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
    freq_axis_Hz = get_frequency_axis(dataset_path)

    t_train_start = time.perf_counter()
    epochs_completed = 0
    for epoch in range(epochs):
        # ===== TRAIN ===== #
        model.train()
        train_loss = 0.0
        physics_loss_accum = 0.0
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
            if use_physics_loss:
                measured_freqs = extract_measured_peaks_batch(signals, freq_axis_Hz, num_peaks=8)
                preds_denorm = denormalize_labels(preds, labels_mean, labels_std)
                labels_denorm = denormalize_labels(labels, labels_mean, labels_std)
                loss_phys = physics_loss_from_current(
                    preds_denorm, measured_freqs, current_true=labels_denorm,
                )
                loss = loss + physics_loss_weight * loss_phys
                physics_loss_accum += loss_phys.item() * signals.size(0)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * signals.size(0)
            cls_correct += (zones_pred == zones_true).sum().item()
            cls_total += zones_true.size(0)

        train_loss /= len(train_loader.dataset)
        physics_loss_val = physics_loss_accum / len(train_loader.dataset) if use_physics_loss else 0.0
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


                cls_correct_val += (zones_pred == zones_true).sum().item()
                cls_total_val += zones_true.size(0)


                preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
                labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)
                abs_err_denorm += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
                n_samples += signals.size(0)

        val_loss /= len(val_loader.dataset)
        cls_acc_val = cls_correct_val / cls_total_val
        mae_denorm = (abs_err_denorm / n_samples).tolist()
        val_mae_mean = np.mean(mae_denorm)

        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_model_state = model.state_dict()

        scheduler.step(val_loss)

        if (epoch+1) % 10 == 0 or epoch == 0:
            phys_str = f" | Physics_loss: {physics_loss_val:.4f} (norm.)" if use_physics_loss else ""
            print(f"[TwoStageJoint] Epoch [{epoch+1}/{epochs}] | "
                  f"Val MAE (Ax,Ay,Az): ({mae_denorm[0]:.4f}, {mae_denorm[1]:.4f}, {mae_denorm[2]:.4f}) | "
                  f"Val Classifier Acc: {cls_acc_val*100:.2f}%{phys_str}")

        epochs_completed = epoch + 1
        if early_stopping.step(val_mae_mean, model):
            print(f"\n[TwoStageJoint] Early stopping at epoch {epoch+1}, best val MAE = {early_stopping.best_loss:.4f}")
            break

    print_training_timing("TwoStageJoint", t_train_start, epochs_completed, epochs)

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

    test_mae_mean = np.mean(mae)
    print(f"\n[TwoStageJoint] Test MAE (mean over Ax/Ay/Az): {test_mae_mean:.4f} A")
    for per_mw_count in (False, True):
        model_dir = get_model_output_dir(dataset_name, arch['joint_dir'], mw_indices, per_mw_count=per_mw_count)
        try_save_best_checkpoint(
            model, model_dir, arch['joint_ckpt'], "best_test_mae.txt",
            test_mae_mean, higher_is_better=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Unified zone model training script")
    parser.add_argument('--model', choices=['classifier', 'regressor', 'two-stage', 'two-stage-joint', 'two-stage-joint-deep'], required=True)
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
    parser.add_argument('--mw_configs', type=int, nargs='+', default=None, help='Subset of MW config indices (default: all channels in the dataset)')
    parser.add_argument('--physic_informed', action='store_true', help='Add physics-guided loss on regressor (Hamiltonian NV, 1 A -> 0.765 mT)')
    parser.add_argument('--physics_loss_weight', type=float, default=0.1,
                        help='Weight for physics loss (dimensionless, linewidth-normalized MHz; try 0.05–1)')
    parser.add_argument('--balanced_test', action='store_true', default=True,
                        help='Balanced test set: same number of samples per zone (default: True)')
    parser.add_argument('--no_balanced_test', action='store_false', dest='balanced_test',
                        help='Use proportional (legacy) test split instead')
    parser.add_argument('--test_samples_per_zone', type=int, default=1,
                        help='Test samples per zone when --balanced_test (default: 1)')
    parser.add_argument('--val_samples_per_zone', type=int, default=None,
                        help='Val samples per zone (default: auto from 10%% target)')
    parser.add_argument('--no_balanced_val', action='store_true',
                        help='Proportional val/test per zone instead of homogeneous val')
    args = parser.parse_args()

    t_run_start = time.perf_counter()
    synthetic = getattr(args, 'synthetic', False)
    split_kwargs = {
        'val_size': 0.10,
        'test_size': 0.10,
        'balanced_val': not args.no_balanced_val,
        'val_samples_per_zone': args.val_samples_per_zone,
        'balanced_test': args.balanced_test,
        'test_samples_per_zone': args.test_samples_per_zone,
    }

    physics_kwargs = {
        'use_physics_loss': args.physic_informed,
        'physics_loss_weight': args.physics_loss_weight,
    }

    if args.model == 'classifier':
        train_classifier(args.dataset_dir, args.batch_size, args.epochs, args.lr, args.weight_decay, args.patience, synthetic=synthetic, mw_configs=args.mw_configs, **split_kwargs)
    elif args.model == 'regressor':
        train_regressor(args.dataset_dir, args.batch_size, args.epochs, args.lr, args.weight_decay, args.patience, synthetic=synthetic, mw_configs=args.mw_configs, **split_kwargs)
    elif args.model == 'two-stage':
        train_two_stage(args.dataset_dir, args.batch_size, args.cls_epochs, args.reg_epochs, args.cls_lr, args.reg_lr, args.cls_weight_decay, args.reg_weight_decay, args.cls_patience, args.reg_patience, args.pretrained_classifier, synthetic=synthetic, mw_configs=args.mw_configs, **split_kwargs, **physics_kwargs)
    elif args.model == 'two-stage-joint':
        train_two_stage_joint(args.dataset_dir, args.batch_size, args.epochs, args.cls_lr, args.reg_lr, args.cls_weight_decay, args.reg_weight_decay, args.patience, lambda_reg=1.0, synthetic=synthetic, mw_configs=args.mw_configs, deep=False, **split_kwargs, **physics_kwargs)
    elif args.model == 'two-stage-joint-deep':
        train_two_stage_joint(args.dataset_dir, args.batch_size, args.epochs, args.cls_lr, args.reg_lr, args.cls_weight_decay, args.reg_weight_decay, args.patience, lambda_reg=1.0, synthetic=synthetic, mw_configs=args.mw_configs, deep=True, **split_kwargs, **physics_kwargs)

    print(f"\nTotal runtime: {format_duration(time.perf_counter() - t_run_start)}")

if __name__ == '__main__':
    main()
