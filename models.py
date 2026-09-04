import torch
import torch.nn as nn


class ODMR_CNN(nn.Module):
    """
    1D CNN regressor for multi-config ODMR spectra regression → (Ax, Ay, Az)
    - Input: (batch, n_mw_configs, n_fq_pts)
    - Output: (batch, 3) (Ax, Ay, Az currents)
    """
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.squeeze(-1)
        return self.regressor(x)


class ODMR_CNN_Compact(nn.Module):
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
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

class ODMR_CNN_Deep(nn.Module):
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class FrequencyAttention(nn.Module):
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):

        h = self.conv(x)
        h = h.transpose(1, 2)
        attn_weights = torch.softmax(self.attention(h), dim=1)
        h_attn = (h * attn_weights).sum(dim=1)
        output = self.fc(h_attn)
        return output


class MWConfig_CNN(nn.Module):
    """1D Conv + MW configs aggregation"""
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.conv_mw = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(16 * n_channels, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):

        batch, n_mw, n_freq = x.shape
        x = x.unsqueeze(2)
        x = x.view(batch * n_mw, 1, n_freq)

        h = self.conv_mw(x)
        h = self.pool(h).squeeze(-1)
        h = h.view(batch, n_mw * 16)

        output = self.fc(h)
        return output


class AxisSplitRegressor(nn.Module):
    """Separate linear branch for Ax and nonlinear branch for Ay/Az."""
    def __init__(self, n_channels=10, n_freq=201, use_attention=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.linear_branch = nn.Sequential(
            nn.Linear(n_channels * n_freq, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        if use_attention:
            self.nonlinear_branch = nn.Sequential(
                nn.Conv1d(n_channels, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
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
                nn.Linear(64, 2)
            )
        else:
            self.nonlinear_branch = nn.Sequential(
                nn.Conv1d(n_channels, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
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
                nn.Linear(64, 2)
            )
        self.use_attention = use_attention

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        ax_pred = self.linear_branch(x_flat)
        h = self.nonlinear_branch(x)
        if self.use_attention:
            h = h.transpose(1, 2)
            attn_weights = torch.softmax(self.attention(h), dim=1)
            h_attn = (h * attn_weights).sum(dim=1)
            ay_az_pred = self.fc_nonlinear(h_attn)
        else:
            h = h.view(batch_size, -1)
            ay_az_pred = self.fc_nonlinear(h)
        output = torch.cat([ax_pred, ay_az_pred], dim=1)
        return output


class ZoneClassifier(nn.Module):
    """
    Classifier that predicts a discrete direction zone index from multi-config ODMR signals.
    """
    def __init__(self, n_channels=10, n_freq=201, n_zones=48, dropout_rate=0.4):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
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

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_zones),
        )

    def forward(self, x):

        h = self.conv(x)
        h = h.squeeze(-1)
        output = self.classifier(h)
        return output


class ZoneAwareRegressor(nn.Module):
    """
    Regress (Ax, Ay, Az) conditioned on a discrete zone index
    Input:
        - signals: (batch, n_mw_configs, n_fq_pts)
        - zones:   (batch,) int in [0, n_zones-1]
    Output:
        - normalized (Ax, Ay, Az) components: (batch, 3)
    """
    def __init__(self, n_channels=10, n_freq=201, n_zones=48, zone_emb_dim=32, output_dim=3, dropout_rate=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
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
        self.zone_emb = nn.Embedding(n_zones, zone_emb_dim)
        self.regressor = nn.Sequential(
            nn.Linear(256 + zone_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(min(dropout_rate, 0.4)),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, zones):
        h = self.feature_extractor(x)
        h = h.squeeze(-1)
        z_emb = self.zone_emb(zones)
        h_cat = torch.cat([h, z_emb], dim=1)
        output = self.regressor(h_cat)
        return output


class ZoneAwareTwoStage(nn.Module):
    """Two-stage model: zone classifier + zone-aware regressor."""
    def __init__(self, n_channels=10, n_freq=201, n_zones=48, zone_emb_dim=32, output_dim=3, dropout_rate=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.classifier = ZoneClassifier(n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, dropout_rate=dropout_rate)
        self.regressor = ZoneAwareRegressor(n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, zone_emb_dim=zone_emb_dim, output_dim=output_dim, dropout_rate=dropout_rate)

    def forward(self, x):
        """Returns regressed (Ax, Ay, Az) using predicted zone"""
        logits = self.classifier(x)
        zones_pred = logits.argmax(dim=1)
        return self.regressor(x, zones_pred)

    def forward_classifier(self, x):
        """returns logits (zone prediction)"""
        return self.classifier(x)

    def forward_regressor(self, x, zones):
        """returns regressed (Ax, Ay, Az) for given zones"""
        return self.regressor(x, zones)


def _zone_conv_backbone_v2(n_channels):
    """4-block 1D CNN backbone (512-dim pooled features)."""
    return nn.Sequential(
        nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
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

        nn.Conv1d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )


class ZoneClassifier2(nn.Module):
    """Deeper zone classifier: 4 conv blocks + 3-layer MLP head."""
    def __init__(self, n_channels=10, n_freq=201, n_zones=48, dropout_rate=0.4):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.conv = _zone_conv_backbone_v2(n_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_zones),
        )

    def forward(self, x):
        h = self.conv(x).squeeze(-1)
        return self.classifier(h)


class ZoneAwareRegressor2(nn.Module):
    """Deeper zone-conditioned regressor: 4 conv blocks + wider MLP head."""
    def __init__(self, n_channels=10, n_freq=201, n_zones=48, zone_emb_dim=32, output_dim=3, dropout_rate=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.feature_extractor = _zone_conv_backbone_v2(n_channels)
        self.zone_emb = nn.Embedding(n_zones, zone_emb_dim)
        self.regressor = nn.Sequential(
            nn.Linear(512 + zone_emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(min(dropout_rate, 0.4)),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, zones):
        h = self.feature_extractor(x).squeeze(-1)
        z_emb = self.zone_emb(zones)
        h_cat = torch.cat([h, z_emb], dim=1)
        return self.regressor(h_cat)


class ZoneAwareTwoStageJointDeep(nn.Module):
    """Two-stage model with deeper classifier and regressor (joint training variant)."""
    def __init__(self, n_channels=10, n_freq=201, n_zones=48, zone_emb_dim=32, output_dim=3, dropout_rate=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.n_freq = n_freq
        self.classifier = ZoneClassifier2(
            n_channels=n_channels, n_freq=n_freq, n_zones=n_zones, dropout_rate=dropout_rate,
        )
        self.regressor = ZoneAwareRegressor2(
            n_channels=n_channels, n_freq=n_freq, n_zones=n_zones,
            zone_emb_dim=zone_emb_dim, output_dim=output_dim, dropout_rate=dropout_rate,
        )

    def forward(self, x):
        logits = self.classifier(x)
        zones_pred = logits.argmax(dim=1)
        return self.regressor(x, zones_pred)

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward_regressor(self, x, zones):
        return self.regressor(x, zones)


ZoneAwareTwoStageJoint = ZoneAwareTwoStage


def available_models():
    return {
        'ODMR_CNN': ODMR_CNN,
        'ODMR_CNN_Compact': ODMR_CNN_Compact,
        'ODMR_CNN_Deep': ODMR_CNN_Deep,
        'FrequencyAttention': FrequencyAttention,
        'MWConfig_CNN': MWConfig_CNN,
        'AxisSplitRegressor': AxisSplitRegressor,
        'ZoneClassifier': ZoneClassifier,
        'ZoneAwareRegressor': ZoneAwareRegressor,
        'ZoneAwareTwoStage': ZoneAwareTwoStage,
        'ZoneAwareTwoStageJoint': ZoneAwareTwoStageJoint,
        'ZoneClassifier2': ZoneClassifier2,
        'ZoneAwareRegressor2': ZoneAwareRegressor2,
        'ZoneAwareTwoStageJointDeep': ZoneAwareTwoStageJointDeep,
    }

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params