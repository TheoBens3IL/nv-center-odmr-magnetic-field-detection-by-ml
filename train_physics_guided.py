"""
Physics-guided training with differentiable spectrum matching (no peak extraction).

Supports:
  - CNN regressors (ODMR_CNN, FrequencyAttention, …)
  - ZoneAwareTwoStage joint (classifier + regressor)

Physics terms (see physics_spectrum.py):
  - Lorentzian NV spectrum synthesis per MW channel
  - Multi-MW consistency via a single B field explaining all channels
  - Separability gating (skip physics on heavily overlapped spectra)
  - Optional post-hoc spectrum refinement at test time

Examples:
  py train_physics_guided.py --dataset_dir dataset_new_1 --mode two-stage-joint
  py train_physics_guided.py --dataset_dir dataset_new_1 --mode cnn --model ODMR_CNN
  py train_physics_guided.py --dataset_dir dataset_new_1 --mode two-stage-joint \\
      --spectrum_loss_weight 0.2 --refine_steps 10 --no_separability_gating
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import models
from dataset import (
    detect_n_freq,
    detect_num_mw_configs,
    get_frequency_axis,
    resolve_dataset_path,
    resolve_mw_indices,
    stratified_zone_split,
    train_val_test_split,
    get_data_loaders,
)
from physics_informed import CURRENT_TO_FIELD_MT_PER_A
from physics_spectrum import (
    combined_physics_guided_loss,
    refine_current_by_spectrum,
    resolve_mw_selectivity,
    spectrum_separability,
)
from train_zone_models import ZoneSubset
from utils import (
    EarlyStopping,
    compute_zones_for_dataset,
    denormalize_labels,
    get_model_output_dir,
    load_normalization_stats,
    try_save_best_checkpoint,
)


def _print_physics_header(args):
    print("=" * 60)
    print("PHYSICS-GUIDED TRAINING (spectrum-level, no peak extraction)")
    print("=" * 60)
    print(f"  Calibration:     1 A -> {CURRENT_TO_FIELD_MT_PER_A} mT")
    print(f"  Spectrum weight: {args.spectrum_loss_weight}")
    print(f"  Lorentzian HWHM: {args.lorentzian_gamma_mhz} MHz")
    print(f"  Dip depth:       {args.lorentzian_depth}")
    if args.no_separability_gating:
        print("  Gating:          OFF (physics on all samples)")
    else:
        print(f"  Gating:          ON (threshold={args.separability_threshold})")
    if args.refine_steps > 0:
        print(f"  Test refinement: {args.refine_steps} Adam steps on spectrum loss")
    if args.mw_config_json:
        print(f"  MW metadata:     {args.mw_config_json}")
    print("=" * 60)


def _evaluate_regression_mae(model, loader, labels_mean, labels_std, device, zone_model=False, two_stage=False,
                             refine_steps=0, freq_axis=None, mw_selectivity=None, refine_lr=0.05,
                             gamma_hz=50e6, dip_depth=0.12):
    model.eval()
    abs_err = torch.zeros(3, dtype=torch.float64)
    n = 0

    with torch.no_grad():
        for batch in loader:
            if zone_model:
                signals, labels, zones = batch
                signals = signals.to(device)
                labels = labels.to(device)
                zones = zones.to(device)
                if two_stage:
                    logits = model.forward_classifier(signals)
                    zones_pred = logits.argmax(dim=1)
                    preds = model.forward_regressor(signals, zones_pred)
                else:
                    preds = model(signals, zones)
            else:
                signals, labels = batch
                signals = signals.to(device)
                labels = labels.to(device)
                preds = model(signals)

            preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
            if refine_steps > 0 and freq_axis is not None and mw_selectivity is not None:
                preds_denorm = refine_current_by_spectrum(
                    preds_denorm, signals.cpu(), freq_axis, mw_selectivity,
                    n_steps=refine_steps, lr=refine_lr, gamma_hz=gamma_hz, dip_depth=dip_depth,
                )

            labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)
            abs_err += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
            n += signals.size(0)

    mae = (abs_err / max(n, 1)).numpy()
    return mae, float(np.mean(mae))


def train_cnn_physics(
    dataset_dir,
    model_name="ODMR_CNN",
    batch_size=16,
    epochs=200,
    lr=2e-4,
    weight_decay=5e-4,
    patience=20,
    synthetic=False,
    mw_configs=None,
    spectrum_loss_weight=0.1,
    lorentzian_gamma_mhz=50.0,
    lorentzian_depth=0.12,
    no_separability_gating=False,
    separability_threshold=0.35,
    mw_config_json=None,
    refine_steps=0,
    refine_lr=0.05,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = resolve_dataset_path(dataset_dir)
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    mw_indices = resolve_mw_indices(synthetic=synthetic, mw_configs=mw_configs, dataset_dir=dataset_path)
    n_freq = detect_n_freq(dataset_path, synthetic)
    n_channels = len(mw_indices)

    if model_name not in models.available_models():
        raise ValueError(f"Unknown model: {model_name}")

    model = models.available_models()[model_name](n_channels=n_channels, n_freq=n_freq).to(device)
    train_set, val_set, test_set = train_val_test_split(
        dataset_path, synthetic=synthetic, mw_indices=mw_indices,
    )
    train_loader, val_loader, test_loader = get_data_loaders(
        train_set, val_set, test_set, batch_size=batch_size, device=device,
    )

    norm_stats = load_normalization_stats(dataset_path)
    labels_mean = norm_stats["labels_mean"]
    labels_std = norm_stats["labels_std"]
    freq_axis = get_frequency_axis(dataset_path)
    gamma_hz = lorentzian_gamma_mhz * 1e6
    mw_selectivity = resolve_mw_selectivity(n_channels, device, mw_config_json)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    best_val_mae = float("inf")
    best_state = None
    gating = not no_separability_gating

    print(f"Model: {model_name} | Dataset: {dataset_path}")
    print(f"MW configs: {mw_indices} ({n_channels} channels)")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        phys_accum = 0.0
        gate_frac = 0.0
        n_batches = 0

        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            preds = model(signals)
            loss = criterion(preds, labels)

            if spectrum_loss_weight > 0:
                preds_denorm = denormalize_labels(preds, labels_mean, labels_std)
                loss_phys = combined_physics_guided_loss(
                    preds_denorm, signals, freq_axis, mw_selectivity,
                    spectrum_weight=1.0,
                    gamma_hz=gamma_hz, dip_depth=lorentzian_depth,
                    gating=gating, separability_threshold=separability_threshold,
                )
                loss = loss + spectrum_loss_weight * loss_phys
                phys_accum += loss_phys.item()
                if gating:
                    gate_frac += spectrum_separability(signals).mean().item()
                n_batches += 1

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * signals.size(0)

        train_loss /= len(train_loader.dataset)
        phys_mean = phys_accum / max(n_batches, 1)
        gate_mean = gate_frac / max(n_batches, 1)

        mae, val_mae_mean = _evaluate_regression_mae(
            model, val_loader, labels_mean, labels_std, device,
        )
        scheduler.step(val_mae_mean)

        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_state = model.state_dict()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            gate_str = f" | gate_mean={gate_mean:.3f}" if gating else ""
            print(
                f"[PhysicsCNN] Epoch [{epoch+1}/{epochs}] | "
                f"Val MAE: {val_mae_mean:.4f} A | Phys: {phys_mean:.4f}{gate_str}"
            )

        if early_stopping.step(val_mae_mean, model):
            print(f"\n[PhysicsCNN] Early stopping at epoch {epoch+1}, best val MAE = {early_stopping.best_metric:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    mae_raw, test_mae_raw = _evaluate_regression_mae(
        model, test_loader, labels_mean, labels_std, device,
    )
    mae_ref, test_mae_ref = mae_raw, test_mae_raw
    if refine_steps > 0:
        mae_ref, test_mae_ref = _evaluate_regression_mae(
            model, test_loader, labels_mean, labels_std, device,
            refine_steps=refine_steps, freq_axis=freq_axis, mw_selectivity=mw_selectivity,
            refine_lr=refine_lr, gamma_hz=gamma_hz, dip_depth=lorentzian_depth,
        )

    print(f"\n[PhysicsCNN] Test MAE (raw):      {test_mae_raw:.4f} A  (Ax={mae_raw[0]:.4f}, Ay={mae_raw[1]:.4f}, Az={mae_raw[2]:.4f})")
    if refine_steps > 0:
        print(f"[PhysicsCNN] Test MAE (refined): {test_mae_ref:.4f} A  (Ax={mae_ref[0]:.4f}, Ay={mae_ref[1]:.4f}, Az={mae_ref[2]:.4f})")

    metric = test_mae_ref if refine_steps > 0 else test_mae_raw
    for per_mw_count in (False, True):
        model_dir = get_model_output_dir(dataset_name, "physics_guided_cnn", mw_indices, per_mw_count=per_mw_count)
        saved = try_save_best_checkpoint(
            model, model_dir, "physics_guided_cnn_best_model.pth", "best_test_mae.txt",
            metric, higher_is_better=False,
        )
        if saved:
            meta = {
                "base_model": model_name,
                "spectrum_loss_weight": spectrum_loss_weight,
                "lorentzian_gamma_mhz": lorentzian_gamma_mhz,
                "refine_steps": refine_steps,
                "test_mae_raw": test_mae_raw,
                "test_mae_refined": test_mae_ref if refine_steps > 0 else None,
            }
            with open(model_dir / "physics_guided_config.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    return model, metric


def train_joint_physics(
    dataset_dir,
    batch_size=16,
    epochs=200,
    cls_lr=1e-3,
    reg_lr=2e-4,
    cls_weight_decay=1e-4,
    reg_weight_decay=5e-4,
    patience=20,
    lambda_reg=1.0,
    synthetic=False,
    mw_configs=None,
    val_size=0.10,
    test_size=0.10,
    balanced_val=True,
    val_samples_per_zone=None,
    balanced_test=False,
    test_samples_per_zone=1,
    spectrum_loss_weight=0.1,
    lorentzian_gamma_mhz=50.0,
    lorentzian_depth=0.12,
    no_separability_gating=False,
    separability_threshold=0.35,
    mw_config_json=None,
    refine_steps=0,
    refine_lr=0.05,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = resolve_dataset_path(dataset_dir)
    dataset_name = os.path.basename(dataset_path.rstrip("/\\"))
    zones, labels_mean, labels_std = compute_zones_for_dataset(dataset_path)
    mw_indices = resolve_mw_indices(synthetic=synthetic, mw_configs=mw_configs, dataset_dir=dataset_path)
    n_freq = detect_n_freq(dataset_path, synthetic)
    n_channels = len(mw_indices)
    n_zones = int(zones.max() + 1)

    train_base, val_base, test_base = stratified_zone_split(
        dataset_path, synthetic=synthetic, mw_indices=mw_indices,
        val_size=val_size, test_size=test_size,
        balanced_val=balanced_val, val_samples_per_zone=val_samples_per_zone,
        balanced_test=balanced_test, test_samples_per_zone=test_samples_per_zone,
    )
    train_loader = DataLoader(ZoneSubset(train_base, zones), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ZoneSubset(val_base, zones), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ZoneSubset(test_base, zones), batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.ZoneAwareTwoStage(
        n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, zone_emb_dim=32, output_dim=3,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": cls_lr, "weight_decay": cls_weight_decay},
        {"params": model.regressor.parameters(), "lr": reg_lr, "weight_decay": reg_weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    freq_axis = get_frequency_axis(dataset_path)
    gamma_hz = lorentzian_gamma_mhz * 1e6
    mw_selectivity = resolve_mw_selectivity(n_channels, device, mw_config_json)
    gating = not no_separability_gating

    print(f"Dataset: {dataset_path}")
    print(f"MW configs: {mw_indices} ({n_channels}/{detect_num_mw_configs(dataset_path, synthetic)} channels)")

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        phys_accum = 0.0
        gate_frac = 0.0
        n_phys = 0

        for signals, labels, zones_true in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            zones_true = zones_true.to(device)

            optimizer.zero_grad()
            logits = model.forward_classifier(signals)
            zones_pred = logits.argmax(dim=1)
            loss_cls = criterion_cls(logits, zones_true)
            preds = model.forward_regressor(signals, zones_pred)
            loss_reg = criterion_reg(preds, labels)
            loss = loss_cls + lambda_reg * loss_reg

            if spectrum_loss_weight > 0:
                preds_denorm = denormalize_labels(preds, labels_mean, labels_std)
                loss_phys = combined_physics_guided_loss(
                    preds_denorm, signals, freq_axis, mw_selectivity,
                    spectrum_weight=1.0,
                    gamma_hz=gamma_hz, dip_depth=lorentzian_depth,
                    gating=gating, separability_threshold=separability_threshold,
                )
                loss = loss + spectrum_loss_weight * loss_phys
                phys_accum += loss_phys.item()
                if gating:
                    gate_frac += spectrum_separability(signals).mean().item()
                n_phys += 1

            loss.backward()
            optimizer.step()

        phys_mean = phys_accum / max(n_phys, 1)
        gate_mean = gate_frac / max(n_phys, 1)

        # Validation
        model.eval()
        abs_err = torch.zeros(3, dtype=torch.float64)
        n_samples = 0
        cls_correct = cls_total = 0

        with torch.no_grad():
            for signals, labels, zones_true in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                zones_true = zones_true.to(device)
                logits = model.forward_classifier(signals)
                zones_pred = logits.argmax(dim=1)
                preds = model.forward_regressor(signals, zones_pred)

                cls_correct += (zones_pred == zones_true).sum().item()
                cls_total += zones_true.size(0)

                preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
                labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)
                abs_err += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
                n_samples += signals.size(0)

        mae_denorm = (abs_err / max(n_samples, 1)).tolist()
        val_mae_mean = float(np.mean(mae_denorm))
        cls_acc = cls_correct / max(cls_total, 1)

        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_state = model.state_dict()

        scheduler.step(val_mae_mean)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            gate_str = f" | gate_mean={gate_mean:.3f}" if gating else ""
            print(
                f"[PhysicsJoint] Epoch [{epoch+1}/{epochs}] | "
                f"Val MAE ({mae_denorm[0]:.4f}, {mae_denorm[1]:.4f}, {mae_denorm[2]:.4f}) | "
                f"Cls {cls_acc*100:.1f}% | Phys {phys_mean:.4f}{gate_str}"
            )

        if early_stopping.step(val_mae_mean, model):
            print(f"\n[PhysicsJoint] Early stopping at epoch {epoch+1}, best val MAE = {early_stopping.best_metric:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    # Test
    model.eval()
    abs_err = sq_err = torch.zeros(3, dtype=torch.float64)
    abs_err_ref = torch.zeros(3, dtype=torch.float64)
    n_samples = cls_correct = cls_total = 0

    with torch.no_grad():
        for signals, labels, zones_true in test_loader:
            signals_dev = signals.to(device)
            labels = labels.to(device)
            zones_true = zones_true.to(device)
            logits = model.forward_classifier(signals_dev)
            zones_pred = logits.argmax(dim=1)
            preds = model.forward_regressor(signals_dev, zones_pred)

            preds_denorm = denormalize_labels(preds.cpu(), labels_mean, labels_std)
            labels_denorm = denormalize_labels(labels.cpu(), labels_mean, labels_std)

            if refine_steps > 0:
                preds_ref = refine_current_by_spectrum(
                    preds_denorm, signals, freq_axis, mw_selectivity,
                    n_steps=refine_steps, lr=refine_lr, gamma_hz=gamma_hz, dip_depth=lorentzian_depth,
                )
                abs_err_ref += torch.sum(torch.abs(preds_ref - labels_denorm), dim=0)

            abs_err += torch.sum(torch.abs(preds_denorm - labels_denorm), dim=0)
            sq_err += torch.sum((preds_denorm - labels_denorm) ** 2, dim=0)
            n_samples += signals.size(0)
            cls_correct += (zones_pred == zones_true).sum().item()
            cls_total += zones_true.size(0)

    mae = (abs_err / max(n_samples, 1)).numpy()
    rmse = torch.sqrt(sq_err / max(n_samples, 1)).numpy()
    cls_acc = cls_correct / max(cls_total, 1)
    test_mae_raw = float(np.mean(mae))

    print(f"\n[PhysicsJoint] Test MAE (raw): Ax={mae[0]:.4f}, Ay={mae[1]:.4f}, Az={mae[2]:.4f} A | mean={test_mae_raw:.4f}")
    print(f"[PhysicsJoint] Test RMSE: {rmse[0]:.4f}, {rmse[1]:.4f}, {rmse[2]:.4f} A")
    print(f"[PhysicsJoint] Classifier accuracy: {cls_acc*100:.2f}%")

    test_mae_ref = test_mae_raw
    if refine_steps > 0:
        mae_ref = (abs_err_ref / max(n_samples, 1)).numpy()
        test_mae_ref = float(np.mean(mae_ref))
        print(f"[PhysicsJoint] Test MAE (refined): Ax={mae_ref[0]:.4f}, Ay={mae_ref[1]:.4f}, Az={mae_ref[2]:.4f} A | mean={test_mae_ref:.4f}")

    metric = test_mae_ref if refine_steps > 0 else test_mae_raw
    for per_mw_count in (False, True):
        model_dir = get_model_output_dir(dataset_name, "physics_guided_joint", mw_indices, per_mw_count=per_mw_count)
        saved = try_save_best_checkpoint(
            model, model_dir, "physics_guided_joint_best_model.pth", "best_test_mae.txt",
            metric, higher_is_better=False,
        )
        if saved:
            meta = {
                "base_model": "ZoneAwareTwoStage",
                "spectrum_loss_weight": spectrum_loss_weight,
                "lorentzian_gamma_mhz": lorentzian_gamma_mhz,
                "refine_steps": refine_steps,
                "test_mae_raw": test_mae_raw,
                "test_mae_refined": test_mae_ref if refine_steps > 0 else None,
            }
            with open(model_dir / "physics_guided_config.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    return model, metric


def main():
    parser = argparse.ArgumentParser(
        description="Physics-guided training via differentiable ODMR spectrum matching.",
    )
    parser.add_argument("--mode", choices=["cnn", "two-stage-joint"], required=True)
    parser.add_argument("--dataset_dir", type=str, default="dataset_new_1")
    parser.add_argument("--model", type=str, default="ODMR_CNN",
                        help="CNN model name when --mode cnn (default: ODMR_CNN)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4, help="CNN learning rate")
    parser.add_argument("--cls_lr", type=float, default=1e-3)
    parser.add_argument("--reg_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--cls_weight_decay", type=float, default=1e-4)
    parser.add_argument("--reg_weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--mw_configs", type=int, nargs="+", default=None)

    # Physics (spectrum-level)
    parser.add_argument("--spectrum_loss_weight", type=float, default=0.1,
                        help="Weight for differentiable spectrum matching loss (try 0.05–0.3)")
    parser.add_argument("--lorentzian_gamma_mhz", type=float, default=50.0,
                        help="HWHM of synthetic Lorentzian dips (MHz)")
    parser.add_argument("--lorentzian_depth", type=float, default=0.12,
                        help="Relative dip depth in synthetic spectrum")
    parser.add_argument("--no_separability_gating", action="store_true",
                        help="Apply physics loss on all samples (including overlapped spectra)")
    parser.add_argument("--separability_threshold", type=float, default=0.35,
                        help="Min separability score [0,1] to apply physics loss when gating is on")
    parser.add_argument("--mw_config_json", type=str, default=None,
                        help="JSON file with MW_field / MW_phase per channel (optional)")
    parser.add_argument("--refine_steps", type=int, default=0,
                        help="Adam refinement steps on spectrum loss at test time (0=off)")
    parser.add_argument("--refine_lr", type=float, default=0.05)

    # Split (zone models)
    parser.add_argument("--balanced_test", action="store_true", default=False)
    parser.add_argument("--no_balanced_test", action="store_false", dest="balanced_test")
    parser.add_argument("--test_samples_per_zone", type=int, default=1)
    parser.add_argument("--val_samples_per_zone", type=int, default=None)
    parser.add_argument("--no_balanced_val", action="store_true")

    args = parser.parse_args()
    _print_physics_header(args)

    split_kwargs = {
        "val_size": 0.10,
        "test_size": 0.10,
        "balanced_val": not args.no_balanced_val,
        "val_samples_per_zone": args.val_samples_per_zone,
        "balanced_test": args.balanced_test,
        "test_samples_per_zone": args.test_samples_per_zone,
    }
    physics_kwargs = {
        "spectrum_loss_weight": args.spectrum_loss_weight,
        "lorentzian_gamma_mhz": args.lorentzian_gamma_mhz,
        "lorentzian_depth": args.lorentzian_depth,
        "no_separability_gating": args.no_separability_gating,
        "separability_threshold": args.separability_threshold,
        "mw_config_json": args.mw_config_json,
        "refine_steps": args.refine_steps,
        "refine_lr": args.refine_lr,
    }

    if args.mode == "cnn":
        train_cnn_physics(
            args.dataset_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            synthetic=args.synthetic,
            mw_configs=args.mw_configs,
            **physics_kwargs,
        )
    else:
        train_joint_physics(
            args.dataset_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            synthetic=args.synthetic,
            mw_configs=args.mw_configs,
            cls_lr=args.cls_lr,
            reg_lr=args.reg_lr,
            cls_weight_decay=args.cls_weight_decay,
            reg_weight_decay=args.reg_weight_decay,
            **split_kwargs,
            **physics_kwargs,
        )


if __name__ == "__main__":
    main()
