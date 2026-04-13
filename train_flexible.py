"""
Flexible training script to test different prediction targets and MW configurations.

Usage:
  # Train on Ay only
  py train_flexible.py --target_components Ay
  
  # Train on Az only
  py train_flexible.py --target_components Az
  
  # Train on Ay and Az (not Ax)
  py train_flexible.py --target_components Ay Az
  
  # Use only first 5 MW configs
  py train_flexible.py --num_mw_configs 5
  
  # Combine: Ay only with 3 configs
  py train_flexible.py --target_components Ay --num_mw_configs 3
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import numpy as np
from dataset import train_val_test_split
from utils import EarlyStopping, load_normalization_stats, denormalize_labels
from models import ODMR_CNN


class FlexibleODMRDataset:
    """Wrapper to select specific MW configs and target components."""
    
    def __init__(self, base_dataset, num_mw_configs=10, target_indices=None):
        """
        Args:
            base_dataset: Original dataset
            num_mw_configs: Number of MW configs to use (1-10)
            target_indices: List of component indices to predict (e.g., [1, 2] for Ay, Az)
        """
        self.base_dataset = base_dataset
        self.num_mw_configs = num_mw_configs
        self.target_indices = target_indices if target_indices is not None else [0, 1, 2]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        signals, labels = self.base_dataset[idx]
        
        # Select subset of MW configs
        signals = signals[:self.num_mw_configs, :]
        
        # Select subset of target components
        labels = labels[self.target_indices]
        
        return signals, labels


def train_flexible(batch_size=32, epochs=200, lr=2e-4, weight_decay=5e-4,
                   dataset_dir="dataset_multi_mw_2", patience=20,
                   target_components=['Ax', 'Ay', 'Az'], num_mw_configs=10):
    """
    Flexible training with custom target components and MW configs.
    
    Args:
        target_components: List of components to predict (e.g., ['Ay'] or ['Ay', 'Az'])
        num_mw_configs: Number of MW configurations to use (1-10)
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dir = os.path.join("datasets_pytorch", dataset_dir)
    
    # Map component names to indices
    component_map = {'Ax': 0, 'Ay': 1, 'Az': 2}
    target_indices = [component_map[c] for c in target_components]
    output_dim = len(target_indices)
    
    print("="*60)
    print("FLEXIBLE ODMR TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_dir}")
    print(f"Target components: {target_components} (output_dim={output_dim})")
    print(f"MW configurations: {num_mw_configs}/10")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Patience: {patience}")
    print("="*60)
    print()
    
    # Load base dataset
    train_set_base, val_set_base, test_set_base = train_val_test_split(dataset_dir)
    
    # Wrap with flexible dataset
    train_set = FlexibleODMRDataset(train_set_base, num_mw_configs, target_indices)
    val_set = FlexibleODMRDataset(val_set_base, num_mw_configs, target_indices)
    test_set = FlexibleODMRDataset(test_set_base, num_mw_configs, target_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")
    print()
    
    # Load normalization stats
    norm_stats = load_normalization_stats(dataset_dir)
    labels_mean = norm_stats['labels_mean'][target_indices]
    labels_std = norm_stats['labels_std'][target_indices]
    
    print(f"Target normalization:")
    for i, comp in enumerate(target_components):
        print(f"  {comp}: mean={labels_mean[i]:.4f}, std={labels_std[i]:.4f}")
    print()
    
    # Create model
    model = ODMR_CNN(
        n_channels=num_mw_configs,
        n_freq=201,
        output_dim=output_dim
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ODMR_CNN")
    print(f"Parameters: {total_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    print("Starting training...")
    print()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for signals, labels in train_loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping
        if early_stopping.step(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best val loss: {early_stopping.best_loss:.6f}")
            model.load_state_dict(early_stopping.best_state)
            break
    
    print("\nTraining completed!")
    print()
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(DEVICE)
            outputs = model(signals)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    # Denormalize
    y_pred_denorm = denormalize_labels(y_pred, labels_mean, labels_std)
    y_true_denorm = denormalize_labels(y_true, labels_mean, labels_std)
    
    # Metrics
    print("="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print()
    
    from sklearn.metrics import r2_score
    
    for i, comp in enumerate(target_components):
        mae = np.abs(y_pred_denorm[:, i] - y_true_denorm[:, i]).mean()
        r2 = r2_score(y_true_denorm[:, i], y_pred_denorm[:, i])
        print(f"{comp}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print()
    
    # Save model
    output_dir = Path("models_flexible")
    output_dir.mkdir(exist_ok=True)
    
    components_str = "_".join(target_components)
    model_name = f"flexible_{components_str}_mw{num_mw_configs}.pt"
    model_path = output_dir / model_name
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Save plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training History: {components_str} (MW={num_mw_configs})')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['lr'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / f"flexible_{components_str}_mw{num_mw_configs}_history.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Flexible ODMR Training')
    
    # Target selection
    parser.add_argument('--target_components', type=str, nargs='+', 
                       default=['Ax', 'Ay', 'Az'],
                       choices=['Ax', 'Ay', 'Az'],
                       help='Components to predict (e.g., Ay or Ay Az)')
    
    parser.add_argument('--num_mw_configs', type=int, default=10, choices=range(1, 11),
                       help='Number of MW configurations to use (1-10)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    
    # Dataset
    parser.add_argument('--dataset_dir', type=str, default='dataset_multi_mw_2')
    
    args = parser.parse_args()
    
    train_flexible(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dataset_dir=args.dataset_dir,
        patience=args.patience,
        target_components=args.target_components,
        num_mw_configs=args.num_mw_configs
    )


if __name__ == "__main__":
    main()
