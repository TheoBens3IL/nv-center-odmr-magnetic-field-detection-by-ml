"""
Physics-guided training via differentiable ODMR spectrum synthesis.

Avoids discrete peak extraction: compares measured spectra to a Lorentzian
superposition model derived from the NV Hamiltonian (8 transitions, 4 axes),
with per-MW selectivity weights.

Used by train_physics_guided.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from physics_informed import (
    CURRENT_TO_FIELD_T_PER_A,
    current_to_magnetic_field,
    nv_transitions_4axes,
    nv_axes,
)


# Default lab / synthetic MW setups (field in a.u., phase in radians).
DEFAULT_MW_CONFIGS = [
    {"MW_field": [0.5, 0.0, 0.0], "MW_phase": np.deg2rad(90.0), "name": "mw0"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(30.0), "name": "mw1"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(150.0), "name": "mw2"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(210.0), "name": "mw3"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(330.0), "name": "mw4"},
    {"MW_field": [0.5, 0.5, 0.0], "MW_phase": np.deg2rad(270.0), "name": "mw5"},
]


def lorentzian_dip_torch(freq_hz, f0_hz, gamma_hz, depth, offset):
    """
    Differentiable Lorentzian dip.

    freq_hz: (batch, n_freq)
    f0_hz:   (batch, n_dips)
    gamma_hz, depth, offset: (batch,) scalars per sample
    Returns: (batch, n_dips, n_freq)
    """
    diff = freq_hz.unsqueeze(1) - f0_hz.unsqueeze(2)
    gamma2 = gamma_hz.view(-1, 1, 1) ** 2
    denom = diff ** 2 + gamma2
    off = offset.view(-1, 1, 1)
    dep = depth.unsqueeze(2)
    return off - dep * (gamma2 / denom)


def _zscore_spectrum(spectrum):
    """Per-spectrum z-score along frequency axis."""
    mean = spectrum.mean(dim=-1, keepdim=True)
    std = spectrum.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (spectrum - mean) / std


def compute_mw_selectivity(n_mw, device, mw_configs=None):
    """
    NV-family selectivity per MW channel: |n̂_NV · B̂_MW| summed over MW components.

    Returns (n_mw, 4) tensor, rows normalized to sum to 1.
    """
    axes = torch.tensor(nv_axes(), dtype=torch.float32, device=device)  # (4, 3)
    if mw_configs is None:
        mw_configs = DEFAULT_MW_CONFIGS[:n_mw]
    while len(mw_configs) < n_mw:
        mw_configs = list(mw_configs) + [mw_configs[-1]]

    weights = []
    for k in range(n_mw):
        mw = mw_configs[k]
        raw = np.asarray(mw["MW_field"], dtype=np.float64).reshape(-1)
        if raw.size == 2:
            field = np.array([raw[0], raw[1], 0.0], dtype=np.float64)
        elif raw.size >= 3:
            field = raw[:3]
        else:
            field = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        norm = np.linalg.norm(field)
        if norm < 1e-12:
            field = np.array([1.0, 0.0, 0.0])
        else:
            field = field / norm
        field_t = torch.tensor(field, dtype=torch.float32, device=device)
        w = torch.abs(axes @ field_t)  # (4,)
        w = w / w.sum().clamp_min(1e-6)
        weights.append(w)
    return torch.stack(weights, dim=0)


def synthesize_spectrum_single_mw(
    B_tesla,
    freq_hz,
    mw_weights,
    gamma_hz,
    dip_depth,
    baseline=1.0,
):
    """
    Synthesize one ODMR spectrum channel from B and NV transitions.

    B_tesla: (batch, 3)
    mw_weights: (4,) NV-family weights for this MW config
    Returns: (batch, n_freq)
    """
    transitions = nv_transitions_4axes(B_tesla)  # (batch, 8)
    batch = B_tesla.shape[0]
    n_freq = freq_hz.numel()

    freq = freq_hz.view(1, n_freq).expand(batch, n_freq)
    gamma = torch.full((batch,), float(gamma_hz), device=B_tesla.device, dtype=B_tesla.dtype)
    offset = torch.full((batch,), float(baseline), device=B_tesla.device, dtype=B_tesla.dtype)

    spectrum = torch.zeros(batch, n_freq, device=B_tesla.device, dtype=B_tesla.dtype)
    for nv_i in range(4):
        w_nv = float(mw_weights[nv_i])
        if w_nv <= 0:
            continue
        f0 = transitions[:, 2 * nv_i: 2 * nv_i + 2]  # (batch, 2)
        depth = torch.full((batch, 2), float(dip_depth) * w_nv, device=B_tesla.device, dtype=B_tesla.dtype)
        dips = lorentzian_dip_torch(freq, f0, gamma, depth, offset)
        spectrum = spectrum + dips.sum(dim=1)

    return spectrum / 4.0


def synthesize_multi_mw_spectrum(
    B_tesla,
    freq_hz,
    mw_selectivity,
    gamma_hz,
    dip_depth,
    baseline=1.0,
):
    """
    B_tesla: (batch, 3)
    mw_selectivity: (n_mw, 4)
    Returns: (batch, n_mw, n_freq)
    """
    channels = []
    for k in range(mw_selectivity.shape[0]):
        channels.append(
            synthesize_spectrum_single_mw(
                B_tesla, freq_hz, mw_selectivity[k], gamma_hz, dip_depth, baseline,
            )
        )
    return torch.stack(channels, dim=1)


def spectrum_separability(signals):
    """
    Soft separability score per sample (no peak picking).

    signals: (batch, n_mw, n_freq) or (batch, n_freq)
    Returns: (batch,) scores in [0, 1] — higher = more resolved dips.
    """
    if signals.dim() == 2:
        signals = signals.unsqueeze(1)
    inverted = signals.max(dim=-1, keepdim=True).values - signals
    dynamic = inverted.max(dim=-1).values - inverted.min(dim=-1).values
    noise = inverted.std(dim=-1).clamp_min(1e-6)
    score = (dynamic / noise).mean(dim=-1)
    # Map ~[0, 3] to [0, 1]
    return torch.sigmoid((score - 1.0) / 0.8)


def spectrum_match_loss(measured, predicted, reduction="mean"):
    """
    MSE between z-scored measured and predicted spectra.

    measured, predicted: (batch, n_mw, n_freq)
    """
    m = _zscore_spectrum(measured)
    p = _zscore_spectrum(predicted)
    per = ((m - p) ** 2).mean(dim=-1)  # (batch, n_mw)
    per_sample = per.mean(dim=-1)  # (batch,)
    if reduction == "none":
        return per_sample
    return per_sample.mean()


def multi_mw_consistency_loss(B_per_mw):
    """
    Penalize spread of B inferred independently per MW channel.

    B_per_mw: (batch, n_mw, 3) — optional auxiliary (e.g. from per-channel heads).
    """
    mean_b = B_per_mw.mean(dim=1, keepdim=True)
    return ((B_per_mw - mean_b) ** 2).mean()


def physics_spectrum_loss(
    current_pred,
    signals,
    freq_axis_hz,
    mw_selectivity,
    gamma_hz=50e6,
    dip_depth=0.12,
    gating=True,
    separability_threshold=0.35,
    reduction="mean",
    t_per_a=None,
):
    """
    Main spectrum-level physics loss (differentiable, no peaks).

    current_pred: (batch, 3) in Amperes (denormalized)
    signals: (batch, n_mw, n_freq) measured (normalized as in dataset)
    """
    if t_per_a is None:
        t_per_a = CURRENT_TO_FIELD_T_PER_A

    device = current_pred.device
    dtype = current_pred.dtype
    freq = torch.as_tensor(freq_axis_hz, device=device, dtype=dtype).flatten()

    B = current_to_magnetic_field(current_pred, t_per_a=t_per_a)
    pred_spectra = synthesize_multi_mw_spectrum(
        B, freq, mw_selectivity, gamma_hz, dip_depth,
    )
    pred_spectra = _zscore_spectrum(pred_spectra)

    loss_per = spectrum_match_loss(signals, pred_spectra, reduction="none")

    if gating:
        gate = spectrum_separability(signals)
        gate = (gate >= separability_threshold).to(dtype)
        if gate.sum() > 0:
            loss_per = loss_per * gate
            return loss_per.sum() / gate.sum().clamp_min(1.0)
        return loss_per.mean() * 0.0

    if reduction == "mean":
        return loss_per.mean()
    return loss_per


def refine_current_by_spectrum(
    current_init,
    signals,
    freq_axis_hz,
    mw_selectivity,
    n_steps=8,
    lr=0.05,
    gamma_hz=50e6,
    dip_depth=0.12,
    t_per_a=None,
):
    """
    Post-hoc differentiable refinement: adjust currents to better match spectra.

    current_init: (batch, 3) denormalized, detached from training graph
    Returns: (batch, 3) refined currents
    """
    if n_steps <= 0:
        return current_init

    current = current_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([current], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = physics_spectrum_loss(
            current, signals, freq_axis_hz, mw_selectivity,
            gamma_hz=gamma_hz, dip_depth=dip_depth, gating=False, t_per_a=t_per_a,
        )
        loss.backward()
        optimizer.step()

    return current.detach()


def load_mw_configs_json(path):
    """Load MW metadata from JSON: list of {MW_field, MW_phase (deg or rad), name?}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    configs = []
    for entry in data:
        phase = entry["MW_phase"]
        if abs(float(phase)) > 2 * np.pi:
            phase = np.deg2rad(float(phase))
        configs.append({
            "MW_field": entry["MW_field"],
            "MW_phase": float(phase),
            "name": entry.get("name", ""),
        })
    return configs


def resolve_mw_selectivity(n_mw, device, mw_config_path=None):
    if mw_config_path is not None and Path(mw_config_path).exists():
        configs = load_mw_configs_json(mw_config_path)
    else:
        configs = None
    return compute_mw_selectivity(n_mw, device, mw_configs=configs)


def combined_physics_guided_loss(
    current_pred,
    signals,
    freq_axis_hz,
    mw_selectivity,
    spectrum_weight=1.0,
    consistency_weight=0.0,
    gamma_hz=50e6,
    dip_depth=0.12,
    gating=True,
    separability_threshold=0.35,
    t_per_a=None,
):
    """
    Combined physics loss for training (spectrum + optional consistency).

    consistency_weight is reserved for future per-MW B heads; 0 by default.
    """
    loss = torch.tensor(0.0, device=current_pred.device, dtype=current_pred.dtype)

    if spectrum_weight > 0:
        loss = loss + spectrum_weight * physics_spectrum_loss(
            current_pred, signals, freq_axis_hz, mw_selectivity,
            gamma_hz=gamma_hz, dip_depth=dip_depth,
            gating=gating, separability_threshold=separability_threshold,
            t_per_a=t_per_a,
        )

    if consistency_weight > 0:
        # Same B explains all MW channels — implicit in spectrum term;
        # explicit term would need per-channel B estimates (not used yet).
        pass

    return loss
