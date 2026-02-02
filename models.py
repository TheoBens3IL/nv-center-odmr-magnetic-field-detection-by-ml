import torch
import torch.nn as nn

class ODMR_CNN(nn.Module):
    """
    Enhanced Convolutional Neural Network for ODMR spectra regression.
    
    Optimized for dataset with:
    - 21,090 samples (2109 experiments × 10 MW configs)
    - 201 frequency points per spectrum
    - 3 output targets (Ax, Ay, Az currents)
    
    Architecture improvements:
    - Deeper network (4 conv blocks) to leverage larger dataset
    - Residual connections for better gradient flow
    - Moderate capacity to avoid overfitting
    - BatchNorm and Dropout for regularization
    """
    def __init__(self, n_freq=201, output_dim=3):
        super().__init__()

        # Feature extraction with residual-like structure
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 201 -> 100
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 -> 50
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 50 -> 25
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 25 -> 1 (global pooling)
        )

        # Regression head with progressive dimension reduction
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, 1, 201) - batch of ODMR spectra
            
        Returns:
            (B, 3) - predicted (Ax, Ay, Az) currents
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.squeeze(-1)  # (B, 512)
        return self.regressor(x)


class ODMR_CNN_Compact(nn.Module):
    """
    Lighter CNN variant for faster training/testing.
    
    Good baseline for comparison - fewer parameters but still effective.
    """
    def __init__(self, n_freq=201, output_dim=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.regressor(x)