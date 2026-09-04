# Magnetic Field prediction from NV centers ODMR spectra using Deep Learning

Predict coil currents `(Ax, Ay, Az)` — and thus the magnetic field `(Bx, By, Bz)` — from multi-configuration ODMR spectra measured on NV centers in diamond.

**Calibration:** 1 A → 0.765 mT at the NV sensor (`physics_informed.py`).

## Setup

```bash
pip install -r requirements.txt
```

Large artifacts (`datasets_pytorch/`, `models_trained/`, raw ESR files) are **not** tracked in Git. Place prepared datasets under `datasets_pytorch/` and trained checkpoints under `models_trained/` locally.

## Typical workflow

```bash
# 1. Raw ESR SPLIT files to PyTorch dataset
python prepare_pytorch_dataset.py --dataset_dir datasets_raw/dataset_x --output_dir datasets_pytorch/dataset_x

# 2. Train (main model: zone-aware-two-stage-joint)
python train_zone_models.py --dataset_dir dataset_x --model two-stage-joint-deep

# 3. Evaluate on test split
python evaluate.py --model ZoneAwareTwoStageJointDeep --dataset_dir dataset_x

# 4. Predict a single spectrum
python predict_spectrum.py --model ZoneAwareTwoStageJointDeep --dataset_dir dataset_x --experiment_id 10
```

Baseline CNN models: `python train.py --model ODMR_CNN --dataset_dir dataset_x`

## Project files

### Core library

| File | Role |
|------|------|
| `dataset.py` | PyTorch datasets, train/val/test splits, stratified zone splits, MW channel selection |
| `models.py` | CNN regressors, axis-split model, zone classifier/regressor, two-stage architectures |
| `utils.py` | Label normalization, NV zone partitioning (48 zones), training helpers, evaluation plots |
| `physics_informed.py` | NV Hamiltonian, ODMR peak extraction, optional physics loss, A -> mT conversion |

### Data preparation

| File | Role |
|------|------|
| `prepare_pytorch_dataset.py` | Convert raw ESR SPLIT files to `datasets_pytorch/` (contrast, averaging, z-score) |
| `prepare_mw_sweep_dataset.py` | Build unlabeled PyTorch datasets from MW amplitude sweeps |
| `apply_zero_field_offset.py` | Correct measured currents for zero-field coil offset |

### Training & evaluation

| File | Role |
|------|------|
| `train_zone_models.py` | **Main trainer** — zone classifier, zone regressor, two-stage / joint models |
| `train.py` | Train baseline CNN models (`ODMR_CNN`, `FrequencyAttention`, `AxisSplitRegressor`, …) |
| `evaluate.py` | Load checkpoints, compute MAE/R²/RMSE, zone heatmaps and diagnostic plots |
| `predict_spectrum.py` | Interactive or single-sample inference with field output in A and mT |
| `comparative_evaluation.py` | Compare MAE across several baseline models on one dataset |

### Analysis & visualization

| File | Role |
|------|------|
| `visualize_dataset.py` | Browse raw SPLIT files or processed `.npy` spectra |
| `experimental_noise.py` | Estimate experimental noise level from raw repetitions |
| `find_best_mw_configs.py` | Pick MW configurations that best resolve ODMR peaks |

### Synthetic data (`synthetical_data/`)

| File | Role |
|------|------|
| `simulate_esr_spectrum.py` | Quantum simulation of NV ESR spectra |
| `build_synthetic_dataset.py` | Generate a raw synthetic dataset |
| `normalize_synthetic_dataset.py` | Normalize synthetic data to the same format as real datasets |

## Models

| Name | Description |
|------|-------------|
| `ODMR_CNN`, `ODMR_CNN_Compact`, `ODMR_CNN_Deep` | Direct CNN regressors |
| `FrequencyAttention`, `MWConfig_CNN` | Alternative CNN architectures |
| `AxisSplitRegressor` | Linear branch for Ax, attention/CNN branch for Ay/Az |
| `ZoneClassifier` / `ZoneClassifier2` | 48-class direction zone classifier |
| `ZoneAwareRegressor` / `ZoneAwareRegressor2` | Regressor conditioned on zone index |
| `ZoneAwareTwoStageJoint` | Classifier + zone-conditioned regressor (deployable) |
| `ZoneAwareTwoStageJointDeep` | Deeper variant — best performer on recent datasets |

List all names: `python predict_spectrum.py --list_models`

## Experimental context

NV centers in diamond are probed with a 532 nm laser and a microwave frequency sweep. Each ODMR dip reflects a spin transition whose position depends on the magnetic field direction relative to the four NV `<111>` axes. Multiple MW power/phase configurations are recorded per field setting; the ML models use them as input channels to predict `(Ax, Ay, Az)` or `(Bx, By, Bz)`.