"""
Train Ridge regression model as a baseline for ODMR magnetic field prediction.
Ridge provides a simple linear baseline to compare against deep learning models.
"""

import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import argparse
from dataset import train_val_test_split
from utils import load_normalization_stats, denormalize_labels


def flatten_signals(dataset, mw_config=None):
    """
    Flatten multi-config signals to 1D feature vector.
    
    Parameters:
        dataset: PyTorch dataset with signals of shape (10, 201)
        mw_config: int or None
            - If None: use all 10 configs → (2010,) features
            - If 0-9: use only that config → (201,) features
    
    Returns:
        X: (n_samples, features) - flattened signals
        y: (n_samples, 3) - labels (Ax, Ay, Az)
    """
    X_list = []
    y_list = []
    
    for i in range(len(dataset)):
        signals, labels = dataset[i]
        
        if mw_config is not None:
            # Use only specified MW config: (10, 201) → (201,)
            X_list.append(signals[mw_config, :].flatten())
        else:
            # Use all configs: (10, 201) → (2010,)
            X_list.append(signals.flatten())
            
        y_list.append(labels)
    
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    
    return X, y


def train_ridge(alpha=1.0, dataset_dir="dataset_multi_mw", mw_config=None):
    """
    Train Ridge regression model on ODMR dataset.
    
    Parameters:
        alpha: Ridge regularization parameter (higher = more regularization)
        dataset_dir: Path to processed dataset
        mw_config: int or None (0-9 to use single config, None to use all 10)
    """
    print("="*60)
    print("RIDGE REGRESSION BASELINE")
    print("="*60)
    print(f"Alpha: {alpha}")
    if mw_config is not None:
        print(f"MW Config: {mw_config} (single config mode)")
    else:
        print(f"MW Config: All 10 configs")
    dataset_dir = os.path.join("datasets_pytorch", dataset_dir)
    print(f"Dataset: {dataset_dir}")
    print("="*60)
    print()
    
    # Load dataset with multi-config format
    train_set, val_set, test_set = train_val_test_split(dataset_dir, multi_config=True)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")
    print()
    
    # Load normalization stats
    norm_stats = load_normalization_stats(dataset_dir)
    labels_mean = norm_stats['labels_mean']
    labels_std = norm_stats['labels_std']
    coord_system = norm_stats.get('coordinate_system', 'cartesian')
    
    if coord_system == 'spherical':
        label_names = ['Ar', 'theta', 'phi']
        units = ['T', 'rad', 'rad']
    else:
        label_names = ['Ax', 'Ay', 'Az']
        units = ['A', 'A', 'A']
    
    print(f"Coordinate system: {coord_system.upper()}")
    print(f"Labels: {label_names}")
    print()
    
    # Flatten signals to feature vectors
    if mw_config is not None:
        print(f"Extracting MW config {mw_config} signals...")
    else:
        print("Flattening multi-config signals...")
    X_train, y_train = flatten_signals(train_set, mw_config=mw_config)
    X_val, y_val = flatten_signals(val_set, mw_config=mw_config)
    X_test, y_test = flatten_signals(test_set, mw_config=mw_config)
    
    print(f"  X_train shape: {X_train.shape} (samples × features)")
    print(f"  y_train shape: {y_train.shape} (samples × outputs)")
    print()
    
    # Create and train Ridge model
    print(f"Training Ridge model with alpha={alpha}...")
    model = MultiOutputRegressor(Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    print("Training complete!")
    print()
    
    # Evaluate on all sets
    print("="*60)
    print("EVALUATION RESULTS (NORMALIZED LABELS)")
    print("="*60)
    print()
    
    for split_name, X, y in [("Train", X_train, y_train), 
                              ("Val", X_val, y_val), 
                              ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        
        # Compute metrics on normalized labels
        mae = mean_absolute_error(y, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(y, y_pred, multioutput='raw_values'))
        
        # Denormalize for absolute metrics
        y_denorm = denormalize_labels(y, labels_mean, labels_std)
        y_pred_denorm = denormalize_labels(y_pred, labels_mean, labels_std)
        
        mae_abs = mean_absolute_error(y_denorm, y_pred_denorm, multioutput='raw_values')
        rmse_abs = np.sqrt(mean_squared_error(y_denorm, y_pred_denorm, multioutput='raw_values'))
        
        # Compute NMAE (Normalized Mean Absolute Error as %)
        nmae = (mae_abs / (np.abs(y_denorm).mean(axis=0) + 1e-8)) * 100
        
        # Compute NRMSE (Normalized RMSE as %)
        y_range = y_denorm.max(axis=0) - y_denorm.min(axis=0)
        nrmse = (rmse_abs / (y_range + 1e-8)) * 100
        
        print(f"{split_name} Set:")
        print(f"  MAE (normalized): {label_names[0]}={mae[0]:.4f}, {label_names[1]}={mae[1]:.4f}, {label_names[2]}={mae[2]:.4f}")
        print(f"  RMSE (normalized): {label_names[0]}={rmse[0]:.4f}, {label_names[1]}={rmse[1]:.4f}, {label_names[2]}={rmse[2]:.4f}")
        print(f"  MAE (absolute): {label_names[0]}={mae_abs[0]:.4f}{units[0]}, {label_names[1]}={mae_abs[1]:.4f}{units[1]}, {label_names[2]}={mae_abs[2]:.4f}{units[2]}")
        print(f"  NMAE: {label_names[0]}={nmae[0]:.2f}%, {label_names[1]}={nmae[1]:.2f}%, {label_names[2]}={nmae[2]:.2f}%")
        print(f"  NRMSE: {label_names[0]}={nrmse[0]:.2f}%, {label_names[1]}={nrmse[1]:.2f}%, {label_names[2]}={nrmse[2]:.2f}%")
        print()
        # Print MAE in a parseable format for Test set
        if split_name == "Test":
            print(f"MAE: {label_names[0]}={mae[0]:.4f}, {label_names[1]}={mae[1]:.4f}, {label_names[2]}={mae[2]:.4f}")
    
    # Compute R² scores
    from sklearn.metrics import r2_score
    y_pred_test = model.predict(X_test)
    r2_scores = r2_score(y_test, y_pred_test, multioutput='raw_values')
    
    print("="*60)
    print("R² SCORES (Test Set)")
    print("="*60)
    for i, name in enumerate(label_names):
        print(f"  {name}: {r2_scores[i]:.4f}")
    print()
    
    # Save model
    output_dir = Path(dataset_dir).parent / "models"
    output_dir.mkdir(exist_ok=True)
    
    import joblib
    if mw_config is not None:
        model_path = output_dir / f"ridge_alpha{alpha}_config{mw_config}.pkl"
    else:
        model_path = output_dir / f"ridge_alpha{alpha}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    print()
    
    return model, r2_scores


def main():
    parser = argparse.ArgumentParser(description='Train Ridge regression baseline')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Ridge regularization parameter (default: 1.0)')
    parser.add_argument('--dataset_dir', type=str, default='dataset_multi_mw',
                       help='Path to processed dataset (default: dataset_multi_mw in datasets_pytorch/)')
    parser.add_argument('--mw_config', type=int, default=None, choices=list(range(10)),
                       help='Use single MW config (0-9) instead of all 10 configs')
    
    args = parser.parse_args()
    
    train_ridge(alpha=args.alpha, dataset_dir=args.dataset_dir, mw_config=args.mw_config)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
