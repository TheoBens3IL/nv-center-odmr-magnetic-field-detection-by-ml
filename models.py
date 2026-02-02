import torch.nn as nn

class ODMR_CNN(nn.Module):
    """
    Ultra-lightweight CNN optimized for 1000-10000 samples.
    
    Design for 5000 samples:
        - ~5-8k parameters (>500 samples/param)
        - Moderate dropout (0.3-0.4)
        - 3 conv layers for efficiency
        - Batch normalization for stable training
    """
    def __init__(self, input_channels=1, output_dim=3, dropout=0.35):
        super().__init__()

        # Feature extraction - ultra-compact
        self.features = nn.Sequential(
            # Block 1: 1 -> 16
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
            
            # Block 2: 16 -> 24
            nn.Conv1d(16, 24, kernel_size=5, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.7),
            
            # Block 3: 24 -> 32
            nn.Conv1d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Regression head - minimal
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        # x: (B, 1, N_freq)
        x = self.features(x)
        return self.regressor(x)