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


import torch
from torch import nn

# A deeper CNN model to capture local patterns on frequency 
class ODMR_CNN_Deep(nn.Module):
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),  # reduce freq dim
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
        )
        
        # Compute flattened size after pooling
        pooled_freq = n_freq // 4
        self.fc = nn.Sequential(
            nn.Linear(128 * pooled_freq, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, channels=10, freq=201)
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


# A model with frequency-wise attention mechanism (adaptative ponderation of frequency points)
class FrequencyAttention(nn.Module):
    """Simple frequency-wise attention"""
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)  # attention score per freq
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # x: (batch, channels=10, freq=201)
        h = self.conv(x)  # (batch, 128, freq)
        h = h.transpose(1, 2)  # (batch, freq, 128)
        
        attn_weights = torch.softmax(self.attention(h), dim=1)  # (batch, freq, 1)
        h_attn = (h * attn_weights).sum(dim=1)  # weighted sum over freq
        
        out = self.fc(h_attn)
        return out


# A model that processes each MW configuration separately and then aggregates
class MWConfig_CNN(nn.Module):
    """1D Conv + MW configs aggregation"""
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        # Treat each MW config separately
        self.conv_mw = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool freq dim
        
        # Combine MW configs
        self.fc = nn.Sequential(
            nn.Linear(16 * n_channels, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        # x: (batch, 10, 201)
        batch, n_mw, n_freq = x.shape
        x = x.unsqueeze(2)  # (batch, 10, 1, freq)
        x = x.view(batch * n_mw, 1, n_freq)
        
        h = self.conv_mw(x)  # (batch*n_mw, 16, freq)
        h = self.pool(h).squeeze(-1)  # (batch*n_mw, 16)
        h = h.view(batch, n_mw * 16)  # combine MW configs
        
        out = self.fc(h)
        return out
