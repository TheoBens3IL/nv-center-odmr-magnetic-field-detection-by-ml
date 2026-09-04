"""Predict (Bx, By, Bz) from an ODMR spectrum using a trained model."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import models
import numpy as np
import pandas as pd
import torch
from matplotlib.widgets import Slider

from dataset import (
    MWConfigSubset,
    ODMRDataset,
    ODMRDatasetSynthetic,
    detect_num_mw_configs,
    get_frequency_axis,
    resolve_dataset_path,
    resolve_mw_indices,
)
from evaluate import (
    _is_two_stage_model,
    _is_zone_regressor,
    load_model,
    predict_single,
)
from physics_informed import CURRENT_TO_FIELD_MT_PER_A
from utils import (
    compute_zones_for_dataset,
    denormalize_labels,
    load_normalization_stats,
    resolve_model_output_dir,
)

N_ZONES = 48
LABEL_NAMES = ["Ax", "Ay", "Az"]
FIELD_NAMES = ["Bx", "By", "Bz"]


def current_a_to_field_mt(current_a):
    """Convert coil current components (A) to B-field components (mT)."""
    return np.asarray(current_a, dtype=np.float64) * CURRENT_TO_FIELD_MT_PER_A


def attach_field_results(result):
    """Add mT fields derived from (Ax, Ay, Az) using lab calibration."""
    if result["kind"] != "regression":
        return result
    result["true_mT"] = current_a_to_field_mt(result["true_A"])
    result["pred_mT"] = current_a_to_field_mt(result["pred_A"])
    result["error_mT"] = result["pred_mT"] - result["true_mT"]
    result["abs_error_mT"] = np.abs(result["error_mT"])
    result["mae_mean_mT"] = float(result["abs_error_mT"].mean())
    return result


def format_zone(zone_index):
    return f"{zone_index} / {N_ZONES}"


def resolve_cli_dataset(path):
    if path is None:
        raise ValueError("dataset_dir is required")
    if not (path.startswith("datasets_pytorch") or os.path.isabs(path)):
        path = os.path.join("datasets_pytorch", path)
    return resolve_dataset_path(path)


def resolve_sample_index(metadata, sample_index=None, experiment_id=None):
    if sample_index is not None and experiment_id is not None:
        row = metadata.iloc[sample_index]
        if int(row["experiment_id"]) != int(experiment_id):
            raise ValueError(
                f"Inconsistent selection: sample_index={sample_index} has "
                f"experiment_id={int(row['experiment_id'])}, not {experiment_id}"
            )
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(metadata):
            raise IndexError(
                f"sample_index={sample_index} out of range [0, {len(metadata) - 1}]"
            )
        return sample_index
    if experiment_id is not None:
        matches = metadata.index[metadata["experiment_id"] == experiment_id].tolist()
        if not matches:
            raise ValueError(f"experiment_id={experiment_id} not found in metadata")
        return matches[0]
    return 0


def build_dataset(dataset_dir, mw_indices, synthetic=False):
    if synthetic:
        base = ODMRDatasetSynthetic(dataset_dir)
    else:
        base = ODMRDataset(dataset_dir)
    max_configs = detect_num_mw_configs(dataset_dir, synthetic=synthetic)
    if mw_indices != list(range(max_configs)):
        base = MWConfigSubset(base, mw_indices)
    return base


def load_sample(dataset_dir, sample_index, mw_indices, synthetic=False):
    dataset = build_dataset(dataset_dir, mw_indices, synthetic=synthetic)
    signals, labels = dataset[sample_index]
    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
    exp_id = int(metadata.iloc[sample_index]["experiment_id"])
    return signals.unsqueeze(0), labels.unsqueeze(0), sample_index, exp_id


def get_true_zone(dataset_dir, sample_index):
    zones, _, _ = compute_zones_for_dataset(dataset_dir)
    return int(zones[sample_index])


def get_predicted_zone(model, signals, device):
    with torch.no_grad():
        logits = model.forward_classifier(signals.to(device))
        return int(logits.argmax(dim=1).item())


def run_prediction(
    model,
    model_name,
    signals,
    labels,
    dataset_dir,
    sample_index,
    device,
    labels_mean,
    labels_std,
    pred_labels_mean,
    pred_labels_std,
):
    zones_tensor = None
    zone_info = None

    if model_name in ("ZoneClassifier", "ZoneClassifier2"):
        with torch.no_grad():
            logits = model(signals.to(device))
            zone_pred = int(logits.argmax(dim=1).item())
        true_zone = get_true_zone(dataset_dir, sample_index)
        return {
            "kind": "classifier",
            "true_zone": true_zone,
            "pred_zone": zone_pred,
            "zone_correct": true_zone == zone_pred,
        }

    if _is_zone_regressor(model_name):
        true_zone = get_true_zone(dataset_dir, sample_index)
        zones_tensor = torch.tensor([true_zone], dtype=torch.long)
        zone_info = ("oracle", true_zone, true_zone)
    elif _is_two_stage_model(model_name):
        true_zone = get_true_zone(dataset_dir, sample_index)
        pred_zone = get_predicted_zone(model, signals, device)
        zones_tensor = torch.tensor([pred_zone], dtype=torch.long)
        zone_info = ("classifier", true_zone, pred_zone)

    pred = predict_single(model, signals, zones_tensor, model_name, device).cpu()
    pred_denorm = denormalize_labels(pred, pred_labels_mean, pred_labels_std).numpy()[0]
    true_denorm = denormalize_labels(labels, labels_mean, labels_std).numpy()[0]
    errors = pred_denorm - true_denorm
    abs_errors = np.abs(errors)

    result = {
        "kind": "regression",
        "true_A": true_denorm,
        "pred_A": pred_denorm,
        "error_A": errors,
        "abs_error_A": abs_errors,
        "mae_mean_A": float(abs_errors.mean()),
        "zone_info": zone_info,
    }
    return attach_field_results(result)


def print_prediction(result, sample_index, experiment_id, model_name, device):
    print("\n===== Spectrum prediction =====")
    print(f"Sample index    : {sample_index}")
    print(f"Experiment id   : {experiment_id}")
    print(f"Model           : {model_name}")
    print(f"Device          : {device}")

    if result["kind"] == "classifier":
        print(f"True zone       : {format_zone(result['true_zone'])}")
        print(f"Predicted zone  : {format_zone(result['pred_zone'])}")
        print(f"Correct         : {'yes' if result['zone_correct'] else 'no'}")
        print()
        return

    if result["zone_info"] is not None:
        mode, true_zone, pred_zone = result["zone_info"]
        if mode == "oracle":
            print(
                f"Zone (oracle)   : {format_zone(true_zone)}"
                "  [true label zone — not deployable alone]"
            )
        else:
            print(f"True zone       : {format_zone(true_zone)}")
            print(f"Predicted zone  : {format_zone(pred_zone)}")

    print(
        f"\nField conversion: 1 A -> {CURRENT_TO_FIELD_MT_PER_A} mT "
        f"(lab calibration, physics_informed.py)"
    )
    print("\n| Component | True (A) | Pred (A) | Error (A) | |Error| (A) |")
    print("|-----------|----------|----------|-----------|------------|")
    for i, name in enumerate(LABEL_NAMES):
        print(
            f"| {name:<9} | {result['true_A'][i]:>8.4f} | {result['pred_A'][i]:>8.4f} | "
            f"{result['error_A'][i]:>+9.4f} | {result['abs_error_A'][i]:>10.4f} |"
        )
    print(f"| {'MAE mean':<9} |          |          |           | {result['mae_mean_A']:>10.4f} |")

    print("\n| Component | True (mT) | Pred (mT) | Error (mT) | |Error| (mT) |")
    print("|-----------|-----------|-----------|------------|-------------|")
    for i, name in enumerate(FIELD_NAMES):
        print(
            f"| {name:<9} | {result['true_mT'][i]:>9.4f} | {result['pred_mT'][i]:>9.4f} | "
            f"{result['error_mT'][i]:>+10.4f} | {result['abs_error_mT'][i]:>11.4f} |"
        )
    print(f"| {'MAE mean':<9} |           |           |            | {result['mae_mean_mT']:>11.4f} |")
    print()


def plot_spectrum_and_prediction(
    signals,
    result,
    frequencies,
    mw_indices,
    sample_index,
    experiment_id,
    model_name,
    save_path=None,
    show=True,
):
    freq = frequencies.copy()
    if freq.max() > 1e6:
        freq = freq / 1e9
        freq_unit = "GHz"
    else:
        freq_unit = "GHz"

    fig, ax = plt.subplots(figsize=(12, 6))
    signals_np = signals.squeeze(0).cpu().numpy()
    for ch, mw_idx in enumerate(mw_indices):
        ax.plot(freq, signals_np[ch], alpha=0.8, label=f"MW {mw_idx}")

    title = f"{model_name} — sample {sample_index} (exp {experiment_id})"
    if result["kind"] == "classifier":
        subtitle = (
            f"Zone: true={result['true_zone']}, pred={result['pred_zone']}"
            f" ({'OK' if result['zone_correct'] else 'KO'})"
        )
    else:
        subtitle = (
            f"True: B=({result['true_mT'][0]:.3f}, {result['true_mT'][1]:.3f}, "
            f"{result['true_mT'][2]:.3f}) mT\n"
            f"Pred: B=({result['pred_mT'][0]:.3f}, {result['pred_mT'][1]:.3f}, "
            f"{result['pred_mT'][2]:.3f}) mT  |  MAE={result['mae_mean_mT']:.4f} mT"
        )
        if result["zone_info"] is not None:
            mode, true_zone, pred_zone = result["zone_info"]
            if mode == "classifier":
                subtitle += f"\nZone: true={format_zone(true_zone)}, pred={format_zone(pred_zone)}"

    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("Normalized signal (z-score)")
    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_interactive(
    model,
    model_name,
    dataset_dir,
    mw_indices,
    synthetic,
    device,
    labels_mean,
    labels_std,
    pred_labels_mean,
    pred_labels_std,
    initial_index=0,
):
    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
    frequencies = get_frequency_axis(dataset_dir)
    n_samples = len(metadata)

    fig, (ax_spec, ax_text) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    plt.subplots_adjust(bottom=0.12, hspace=0.35)

    lines = []
    freq = frequencies.copy()
    if freq.max() > 1e6:
        freq = freq / 1e9

    first_signals, _, _, _ = load_sample(dataset_dir, initial_index, mw_indices, synthetic)
    signals_np = first_signals.squeeze(0).cpu().numpy()
    for ch, mw_idx in enumerate(mw_indices):
        line, = ax_spec.plot(freq, signals_np[ch], alpha=0.8, label=f"MW {mw_idx}")
        lines.append(line)

    ax_spec.set_xlabel("Frequency (GHz)")
    ax_spec.set_ylabel("Normalized signal (z-score)")
    ax_spec.legend(loc="upper right", fontsize=8, ncol=2)
    ax_spec.grid(True, alpha=0.3)

    ax_text.axis("off")
    text_artist = ax_text.text(
        0.02, 0.95, "", transform=ax_text.transAxes, va="top", ha="left",
        fontsize=11, family="monospace",
    )

    def refresh(sample_index):
        signals, labels, _, exp_id = load_sample(
            dataset_dir, sample_index, mw_indices, synthetic,
        )
        result = run_prediction(
            model, model_name, signals, labels, dataset_dir, sample_index, device,
            labels_mean, labels_std, pred_labels_mean, pred_labels_std,
        )

        signals_np = signals.squeeze(0).cpu().numpy()
        for line, ch in zip(lines, range(len(mw_indices))):
            line.set_ydata(signals_np[ch])

        ax_spec.set_title(f"{model_name} — sample {sample_index} / exp {exp_id}")
        ax_spec.relim()
        ax_spec.autoscale_view(scalex=False, scaley=True)

        if result["kind"] == "classifier":
            msg = (
                f"Experiment {exp_id}\n"
                f"True zone : {format_zone(result['true_zone'])}\n"
                f"Pred zone : {format_zone(result['pred_zone'])}  "
                f"({'correct' if result['zone_correct'] else 'wrong'})"
            )
        else:
            msg = (
                f"Experiment {exp_id}\n"
                f"True (A)  : Ax={result['true_A'][0]:+.4f}  Ay={result['true_A'][1]:+.4f}  "
                f"Az={result['true_A'][2]:+.4f}\n"
                f"Pred (A)  : Ax={result['pred_A'][0]:+.4f}  Ay={result['pred_A'][1]:+.4f}  "
                f"Az={result['pred_A'][2]:+.4f}\n"
                f"MAE (A)   : {result['mae_mean_A']:.4f}\n"
                f"True (mT) : Bx={result['true_mT'][0]:+.4f}  By={result['true_mT'][1]:+.4f}  "
                f"Bz={result['true_mT'][2]:+.4f}\n"
                f"Pred (mT) : Bx={result['pred_mT'][0]:+.4f}  By={result['pred_mT'][1]:+.4f}  "
                f"Bz={result['pred_mT'][2]:+.4f}\n"
                f"MAE (mT)  : {result['mae_mean_mT']:.4f}"
            )
            if result["zone_info"] is not None:
                mode, true_zone, pred_zone = result["zone_info"]
                if mode == "classifier":
                    msg += f"\nZone      : true={format_zone(true_zone)}, pred={format_zone(pred_zone)}"
                else:
                    msg += f"\nZone      : {format_zone(true_zone)} (oracle)"

        text_artist.set_text(msg)
        fig.canvas.draw_idle()

    refresh(initial_index)

    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.025])
    slider = Slider(ax_slider, "Sample", 0, n_samples - 1, valinit=initial_index, valstep=1)

    def on_slide(_val):
        refresh(int(slider.val))

    slider.on_changed(on_slide)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Pick a spectrum from a PyTorch dataset and predict with a trained model.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model class name (e.g. ODMR_CNN, ZoneAwareTwoStageJointDeep)",
    )
    parser.add_argument(
        "--list_models", action="store_true",
        help="List available model names and exit",
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="PyTorch dataset (name under datasets_pytorch/ or full path)",
    )
    parser.add_argument(
        "--train_dataset_dir", type=str, default=None,
        help="Dataset the model was trained on (default: same as --dataset_dir)",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Override checkpoint directory (default: models_trained/...)",
    )
    parser.add_argument(
        "--sample_index", type=int, default=None,
        help="Row index in metadata.csv (0-based)",
    )
    parser.add_argument(
        "--experiment_id", type=int, default=None,
        help="experiment_id from metadata.csv (alternative to --sample_index)",
    )
    parser.add_argument(
        "--mw_configs", type=int, nargs="+", default=None,
        help="Subset of MW config indices (default: all channels in dataset)",
    )
    parser.add_argument("--synthetic", action="store_true", help="Synthetic dataset format")
    parser.add_argument(
        "--interactive", action="store_true",
        help="Browse samples with a slider and update prediction live",
    )
    parser.add_argument("--plot", action="store_true", help="Show spectrum + prediction plot")
    parser.add_argument("--save_plot", type=str, default=None, help="Save plot to this path")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name in sorted(models.available_models().keys()):
            print(f"  - {name}")
        return

    if args.model is None:
        parser.error("--model is required (or use --list_models)")
    if args.dataset_dir is None:
        parser.error("--dataset_dir is required")

    if args.model not in models.available_models():
        raise ValueError(
            f"Unknown model: {args.model}. Available: {list(models.available_models().keys())}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = resolve_cli_dataset(args.dataset_dir)
    train_dataset_dir = resolve_cli_dataset(args.train_dataset_dir or args.dataset_dir)

    mw_indices = resolve_mw_indices(
        synthetic=args.synthetic, mw_configs=args.mw_configs, dataset_dir=dataset_dir,
    )

    if args.model_dir is not None:
        model_dir = Path(args.model_dir)
    else:
        model_dir = resolve_model_output_dir(train_dataset_dir, args.model, mw_indices)

    print(f"Dataset          : {dataset_dir}")
    if train_dataset_dir != dataset_dir:
        print(f"Train dataset    : {train_dataset_dir}")
    print(f"Model checkpoint : {model_dir}")
    print(f"MW configs       : {mw_indices}")

    norm_stats = load_normalization_stats(dataset_dir)
    train_norm = load_normalization_stats(train_dataset_dir)
    labels_mean = norm_stats["labels_mean"]
    labels_std = norm_stats["labels_std"]
    pred_labels_mean = train_norm["labels_mean"]
    pred_labels_std = train_norm["labels_std"]

    model = load_model(
        args.model, model_dir, train_dataset_dir,
        device=device, mw_indices=mw_indices, synthetic=args.synthetic,
    )

    metadata = pd.read_csv(os.path.join(dataset_dir, "metadata.csv"))
    sample_index = resolve_sample_index(
        metadata, sample_index=args.sample_index, experiment_id=args.experiment_id,
    )

    if args.interactive:
        run_interactive(
            model, args.model, dataset_dir, mw_indices, args.synthetic, device,
            labels_mean, labels_std, pred_labels_mean, pred_labels_std,
            initial_index=sample_index,
        )
        return

    signals, labels, sample_index, exp_id = load_sample(
        dataset_dir, sample_index, mw_indices, synthetic=args.synthetic,
    )
    result = run_prediction(
        model, args.model, signals, labels, dataset_dir, sample_index, device,
        labels_mean, labels_std, pred_labels_mean, pred_labels_std,
    )
    print_prediction(result, sample_index, exp_id, args.model, device)

    if args.plot or args.save_plot:
        frequencies = get_frequency_axis(dataset_dir)
        plot_spectrum_and_prediction(
            signals, result, frequencies, mw_indices,
            sample_index, exp_id, args.model,
            save_path=args.save_plot, show=args.plot,
        )


if __name__ == "__main__":
    main()
