import torch
import torch.nn as nn


class ODMR_CNN(nn.Module):
    """
    Enhanced Convolutional Neural Network for ODMR spectra regression.
    - Input: (batch, 10, 201) (10 MW configs)
    - Output: (batch, 3) (Ax, Ay, Az currents)
    """
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
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
            nn.AdaptiveAvgPool1d(1),  # 25 -> 1
        )
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
        Args:
            x: (batch, 10, 201) - batch of ODMR spectra (multi-config)
        Returns:
            (batch, 3) - predicted (Ax, Ay, Az) currents
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.squeeze(-1)  # (batch, 512)
        return self.regressor(x)


class ODMR_CNN_Compact(nn.Module):
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
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
        
        output = self.fc(h_attn)
        return output


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
        
        output = self.fc(h)
        return output


class HybridODMRPredictor(nn.Module):
    """
    Hybrid architecture with separate branches for different components.
    
    Architecture rationale based on Ridge regression diagnostic:
    - Ax: Linear relationship (R²=0.9996) → Simple linear branch
    - Ay, Az: No linear relationship (R²<0) → Non-linear CNN/Attention branch
    
    This allows each component to be predicted optimally according to its
    physical characteristics.
    """    
    def __init__(self, n_channels=10, n_freq=201, use_attention=True):
        """
        Args:
            n_channels: Number of MW configurations (default: 10)
            n_freq: Number of frequency points (default: 201)
            use_attention: Use FrequencyAttention for Ay/Az (True) or CNN (False)
        """
        super().__init__()
        
        # Branch 1: Linear predictor for Ax (flattened input)
        # Input: (batch, 10*201) = (batch, 2010)
        self.linear_branch = nn.Sequential(
            nn.Linear(n_channels * n_freq, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: Ax
        )
        
        # Branch 2: Non-linear predictor for Ay, Az
        # Input: (batch, 10, 201)
        if use_attention:
            # Use FrequencyAttention architecture
            self.nonlinear_branch = nn.Sequential(
                nn.Conv1d(n_channels, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)  # 201 -> 100
            )
            
            self.attention = nn.Sequential(
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            
            self.fc_nonlinear = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)  # Output: Ay, Az
            )
        else:
            # Use simple CNN
            self.nonlinear_branch = nn.Sequential(
                nn.Conv1d(n_channels, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),  # 201 -> 100
                
                nn.Conv1d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            
            self.fc_nonlinear = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # Output: Ay, Az
            )
        
        self.use_attention = use_attention
    
    def forward(self, x):
        """
        Args:
            x: (batch, 10, 201) - Multi-config ODMR signals
            
        Returns:
            (batch, 3) - [Ax, Ay, Az] predictions
        """
        batch_size = x.shape[0]
        
        # Branch 1: Linear prediction for Ax
        x_flat = x.view(batch_size, -1)  # (batch, 2010)
        ax_pred = self.linear_branch(x_flat)  # (batch, 1)
        
        # Branch 2: Non-linear prediction for Ay, Az
        h = self.nonlinear_branch(x)  # (batch, channels, freq)
        
        if self.use_attention:
            h = h.transpose(1, 2)  # (batch, freq, channels)
            attn_weights = torch.softmax(self.attention(h), dim=1)
            h_attn = (h * attn_weights).sum(dim=1)  # (batch, channels)
            ay_az_pred = self.fc_nonlinear(h_attn)  # (batch, 2)
        else:
            h = h.view(batch_size, -1)  # (batch, channels)
            ay_az_pred = self.fc_nonlinear(h)  # (batch, 2)
        
        # Concatenate predictions: [Ax, Ay, Az]
        output = torch.cat([ax_pred, ay_az_pred], dim=1)  # (batch, 3)
        
        return output


class ZoneClassifier(nn.Module):
    """
    Classifier that predicts a discrete direction zone index from multi-config ODMR signals.
    Expects input shape (batch, 10, 201) and outputs logits over n_zones classes.
    """
    def __init__(self, n_channels=10, n_freq=201, n_zones=48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 201 -> 100

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 -> 50

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 50 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_zones),
        )

    def forward(self, x):
        # x: (batch, 10, 201)
        h = self.conv(x)       # (batch, 256, 1)
        h = h.squeeze(-1)      # (batch, 256)
        output = self.classifier(h)  # (batch, n_zones)
        return output


class ZoneAwareRegressor(nn.Module):
    """
    Regressor that conditions on the discrete direction zone index.
    Input:
        - signals: (batch, 10, 201)
        - zones:   (batch,) int64 in [0, n_zones-1]
    Output:
        - (batch, 3) normalized (Ax, Ay, Az) components.
    """
    def __init__(self, n_channels=10, n_freq=201, n_zones=48, zone_emb_dim=32, output_dim=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 201 -> 100

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 -> 50

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 50 -> 1
        )

        self.zone_emb = nn.Embedding(n_zones, zone_emb_dim) # degrees of freedom the model has to describe how each of the 48 zones behaves differently.

        self.regressor = nn.Sequential(
            nn.Linear(256 + zone_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, zones):
        # x: (batch, 10, 201)
        # zones: (batch,) long
        h = self.feature_extractor(x)         # (batch, 256, 1)
        h = h.squeeze(-1)                     # (batch, 256)
        z_emb = self.zone_emb(zones)          # (batch, zone_emb_dim)
        h_cat = torch.cat([h, z_emb], dim=1)  # (batch, 256 + zone_emb_dim)
        output = self.regressor(h_cat)        # (batch, 3)
        return output


def available_models():
    return {
        'ODMR_CNN': ODMR_CNN,
        'ODMR_CNN_Compact': ODMR_CNN_Compact,
        'ODMR_CNN_Deep': ODMR_CNN_Deep,
        'FrequencyAttention': FrequencyAttention,
        'MWConfig_CNN': MWConfig_CNN,
        'HybridODMRPredictor': HybridODMRPredictor,
        'ZoneClassifier': ZoneClassifier,
        'ZoneAwareRegressor': ZoneAwareRegressor
    }

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params