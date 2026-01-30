import torch.nn as nn

class ODMR_CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for processing ODMR spectra.
    Architecture:
        - Conv1D → BatchNorm → ReLU → MaxPool1D
        - Conv1D → BatchNorm → ReLU → MaxPool1D
        - Conv1D → BatchNorm → ReLU → AdaptiveAvgPool1D
        - Fully Connected → ReLU
        - Fully Connected (Output Layer)
    """
    def __init__(self, n_freq, output_dim=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # invariant to freq resolution
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # x: (B, 1, N_freq)
        x = self.features(x)
        x = x.squeeze(-1)  # (B, 64)
        return self.regressor(x)