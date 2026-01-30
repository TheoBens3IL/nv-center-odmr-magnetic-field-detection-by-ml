import torch.nn as nn

class ODMR_CNN(nn.Module):
    """
    Ultra-lightweight CNN optimized for SMALL datasets (179 samples).
    
    Key optimizations:
        - Minimal parameters (~2.5k instead of 50k+) to match tiny dataset
        - Aggressive dropout (0.5) to prevent overfitting
        - Strided convolutions to reduce parameters
        - Shallow architecture (2 conv layers only)
    
    With only 179 training samples, we need params << 5000.
    Previous model had ~50k params → severe overfitting!
    """
    def __init__(self, n_freq, output_dim=3, dropout=0.5):
        super().__init__()

        # Lightweight feature extraction with stride to reduce params
        self.features = nn.Sequential(
            # Layer 1: Aggressive dimensionality reduction
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            
            # Layer 2: Further compression
            nn.Conv1d(16, 24, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            
            # Adaptive pooling to fixed small size
            nn.AdaptiveAvgPool1d(4),
        )

        # Minimal regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 4, 24),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(24, output_dim),
        )

    def forward(self, x):
        # x: (B, 1, N_freq)
        x = self.features(x)
        return self.regressor(x)