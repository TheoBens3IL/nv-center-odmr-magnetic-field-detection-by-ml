import torch
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Lab calibration: coil current (A) -> magnetic field at the NV (mT)
CURRENT_TO_FIELD_MT_PER_A = 0.765
CURRENT_TO_FIELD_T_PER_A = CURRENT_TO_FIELD_MT_PER_A * 1e-3


# Typical ODMR dip HWHM (MHz) for dimensionless frequency error.
ODMR_LINEWIDTH_MHZ = 50.0
NV_TRANSITIONS_PER_SAMPLE = 8


def current_to_magnetic_field(current_A, t_per_a=None):
    """
    Convert lab current components (Ax, Ay, Az) in Amperes to B-field in Tesla.

    Default scale: 1 A -> 0.765 mT.
    """
    if t_per_a is None:
        t_per_a = CURRENT_TO_FIELD_T_PER_A
    return current_A * t_per_a


def _hz_to_mhz(freq_hz):
    return freq_hz * 1e-6


def _normalized_freq_mse_mhz(diff_mhz):
    """Dimensionless MSE: (Δf / linewidth)² averaged over terms."""
    return ((diff_mhz / ODMR_LINEWIDTH_MHZ) ** 2).mean()


def physics_loss_label_transitions(current_pred, current_true, t_per_a=None):
    """
    Hamiltonian consistency: NV transitions from predicted vs true currents (A).

    Compares the 8 slot-ordered transitions (4 NV axes × f⁻/f⁺) in MHz.
    """
    B_pred = current_to_magnetic_field(current_pred, t_per_a=t_per_a)
    B_true = current_to_magnetic_field(current_true.detach(), t_per_a=t_per_a)
    f_pred_mhz = _hz_to_mhz(nv_transitions_4axes(B_pred))
    f_true_mhz = _hz_to_mhz(nv_transitions_4axes(B_true))
    return _normalized_freq_mse_mhz(f_pred_mhz - f_true_mhz)


def physics_loss_peaks_per_slot(f_pred_hz, measured_freqs_hz):
    """
    Match each NV transition slot to its nearest measured ODMR dip (MHz).

    Unlike global sorting, each of the 8 Hamiltonian slots is aligned to the
    closest detected dip frequency, which preserves NV-axis ordering in f_pred.
    """
    f_pred_mhz = _hz_to_mhz(f_pred_hz)
    measured_mhz = _hz_to_mhz(measured_freqs_hz.detach().to(f_pred_hz.device))

    diff = f_pred_mhz.unsqueeze(2) - measured_mhz.unsqueeze(1)  # (B, 8, 8)
    nan_mask = torch.isnan(measured_mhz).unsqueeze(1).expand_as(diff)
    diff = diff.masked_fill(nan_mask, float("inf"))

    min_abs_mhz = diff.abs().amin(dim=2)  # (B, 8)
    valid = torch.isfinite(min_abs_mhz)
    if not valid.any():
        return torch.tensor(0.0, device=f_pred_hz.device, dtype=f_pred_hz.dtype)
    return ((min_abs_mhz[valid] / ODMR_LINEWIDTH_MHZ) ** 2).mean()


def physics_loss_from_current(
    current_pred,
    measured_freqs=None,
    current_true=None,
    peak_weight=0.5,
    label_weight=0.5,
    t_per_a=None,
):
    """
    Combined physics-informed loss (dimensionless, ODMR linewidth-normalized).

    Terms:
        - label: nv_transitions(B_pred) vs nv_transitions(B_true)
        - peaks: each NV slot vs nearest measured dip in the spectrum

    Frequencies internally use MHz (natural ODMR unit); inputs/outputs in Hz/A unchanged.
    """
    if current_true is None and measured_freqs is None:
        return torch.tensor(0.0, device=current_pred.device, dtype=current_pred.dtype)

    loss = torch.tensor(0.0, device=current_pred.device, dtype=current_pred.dtype)
    B_pred = current_to_magnetic_field(current_pred, t_per_a=t_per_a)

    if current_true is not None and label_weight > 0:
        loss = loss + label_weight * physics_loss_label_transitions(
            current_pred, current_true, t_per_a=t_per_a,
        )
    if measured_freqs is not None and peak_weight > 0:
        f_pred_hz = nv_transitions_4axes(B_pred)
        loss = loss + peak_weight * physics_loss_peaks_per_slot(f_pred_hz, measured_freqs)

    return loss


def physics_loss(B_pred, measured_freqs):
    """Legacy entry point — peak term only, NV slot matching."""
    f_pred_hz = nv_transitions_4axes(B_pred)
    return physics_loss_peaks_per_slot(f_pred_hz, measured_freqs)


def extract_measured_peaks_batch(signals, freq_axis_Hz, num_peaks=8):
    """
    Extract ODMR peak frequencies from a batch of spectra.

    Parameters:
        signals: tensor (batch, n_mw, n_freq) or (batch, n_freq)
        freq_axis_Hz: 1D frequency axis in Hz

    Returns:
        (batch, num_peaks) tensor in Hz (NaN for missing peaks)
    """
    signals_np = signals.detach().cpu().numpy()
    measured_freqs = []
    for s in signals_np:
        if s.ndim == 2:
            spectrum = np.mean(s, axis=0)
        else:
            spectrum = s
        measured_freqs.append(extract_odmr_peak_frequencies(spectrum, freq_axis_Hz, num_peaks=num_peaks))
    device = signals.device if torch.is_tensor(signals) else "cpu"
    return torch.tensor(np.array(measured_freqs), dtype=torch.float32, device=device)


def spin_operators(device, dtype=torch.complex64):
    """
    Returns spin-1 operators Sx, Sy, Sz (complex Hermitian).
    """
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=device))

    Sx = torch.tensor(
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]],
        dtype=dtype, device=device
    ) / sqrt2

    Sy = torch.tensor(
        [[0, -1j, 0],
         [1j, 0, -1j],
         [0, 1j, 0]],
        dtype=dtype, device=device
    ) / sqrt2

    Sz = torch.tensor(
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, -1]],
        dtype=dtype, device=device
    )

    return Sx, Sy, Sz


def hamiltonian_matrix(B):
    """
    Calculates Hamiltonian of one NV center, for a given magnetic field B.
    B: tensor of shape (batch, 3) in Tesla
    Returns: Hamiltonian (batch, 3, 3) complex Hermitian
    """
    D = 2.87e9    # Zero-field splitting (Hz)
    E = 0         # Strain parameter (Hz)
    GAMMA = 28e9  # NV gyromagnetic ratio (Hz/T)

    Sx, Sy, Sz = spin_operators(B.device, dtype=torch.complex64)

    Bx, By, Bz = B[:,0].view(-1, 1, 1), B[:,1].view(-1, 1, 1), B[:,2].view(-1, 1, 1)

    H = (
        D * (Sz @ Sz)
        + E * (Sx @ Sx - Sy @ Sy)
        + GAMMA * (Bx * Sx + By * Sy + Bz * Sz)
    )

    return H


def hamiltonian_energy(B):
    """
    Returns sorted eigenvalues (batch, 3)
    """
    H = hamiltonian_matrix(B)
    eigvals = torch.linalg.eigvalsh(H)  # real for Hermitian
    return eigvals


# def nv_transitions(B):
#     """
#     Returns predicted ODMR transition frequencies (f_minus, f_plus) for given B.
#     B: (batch, 3) in Tesla
#     Returns: (batch, 2) [f_minus, f_plus] in Hz
#     """
#     eigvals = hamiltonian_energy(B)
#     f_minus = eigvals[:, 1] - eigvals[:, 0]      # |0> to |-1>
#     f_plus  = eigvals[:, 2] - eigvals[:, 0]      # |0> to |+1>
#     return torch.stack([f_minus, f_plus], dim=1)


# def physics_loss(B_pred, measured_freqs):
#     """
#     B_pred: (batch, 3)
#     measured_freqs: (batch, 2)  [f_minus, f_plus]
#     """
#     f_pred = nv_transitions(B_pred)
#     return torch.mean((f_pred - measured_freqs) ** 2)

# physic_loss with B labels
# def physics_loss(B_pred, B_true):
#     """
#     B_pred: (batch, 3)
#     B_true: (batch, 3)
#     """
#     f_pred = nv_transitions(B_pred) / 1e9
#     f_true = nv_transitions(B_true) / 1e9
#     return torch.mean((f_pred - f_true) ** 2)


def nv_rotation_matrices(device):
    return torch.tensor([
        [[ 0.5,        -0.70710678,  0.5       ],
         [ 0.5,         0.70710678,  0.5       ],
         [-0.70710678,  0.0,         0.70710678]],

        [[-0.5,         0.70710678, -0.5       ],
         [-0.5,        -0.70710678, -0.5       ],
         [-0.70710678,  0.0,         0.70710678]],

        [[-0.5,         0.70710678,  0.5       ],
         [ 0.5,         0.70710678, -0.5       ],
         [-0.70710678,  0.0,        -0.70710678]],

        [[ 0.5,        -0.70710678, -0.5       ],
         [-0.5,        -0.70710678,  0.5       ],
         [-0.70710678,  0.0,        -0.70710678]]
    ], dtype=torch.float32, device=device)


def nv_axes():
    """
    Return the 4 NV axes as unit vectors in Cartesian coordinates.
    These correspond to the <111> directions in the diamond lattice.
    """
    axes = np.array([
        [ 0.5,  0.5,  0.70710678],
        [-0.5, -0.5,  0.70710678],
        [ 0.5, -0.5, -0.70710678],
        [-0.5,  0.5, -0.70710678]
    ], dtype=np.float64)
    return axes / np.linalg.norm(axes, axis=1, keepdims=True)


def nv_transitions_4axes(B_lab):
    """
    B_lab: (batch, 3) in Tesla
    Returns: (batch, 8) transition frequencies in Hz
    """
    T = nv_rotation_matrices(B_lab.device)  # (4,3,3)

    transitions = []
    for i in range(4):
        # Rotate B into NV frame
        B_nv = torch.matmul(B_lab, T[i].T)  # (batch, 3)

        # Compute full Hamiltonian eigenvalues
        eigvals = hamiltonian_energy(B_nv)  # (batch, 3)

        f_minus = eigvals[:, 1] - eigvals[:, 0]  # |0> to |-1>
        f_plus  = eigvals[:, 2] - eigvals[:, 0]  # |0> to |+1>

        transitions.append(torch.stack([f_minus, f_plus], dim=1))

    # Concatenate all 4 NV families → (batch, 8)
    return torch.cat(transitions, dim=1)


# def physics_loss(B_pred, measured_freqs):
#     """
#     B_pred: (batch, 3)
#     measured_freqs: (batch, 8)
#     """
#     f_pred = nv_transitions_4axes(B_pred)
#     return torch.mean((f_pred - measured_freqs) ** 2)

# def physics_loss(B_pred, measured_freqs):
#     """
#     B_pred: (batch, 3)
#     measured_freqs: (batch, 8) → some values can be NaN if missing peaks
#     """
#     f_pred = nv_transitions_4axes(B_pred)  # (batch, 8)

#     # Mask out NaNs
#     if not torch.is_tensor(measured_freqs):
#         measured_freqs = torch.tensor(measured_freqs, device=f_pred.device)
#     else:
#         measured_freqs = measured_freqs.detach().clone().to(f_pred.device)
    
#     mask = ~torch.isnan(measured_freqs)
#     diff = torch.zeros_like(f_pred)
#     diff[mask] = f_pred[mask] - measured_freqs[mask]

#     return torch.mean(diff ** 2)


def lorentzian(f, f0, gamma, depth, offset):
    """
    Lorentzian dip model.
    f0     : center frequency
    gamma  : HWHM
    depth  : dip amplitude
    offset : baseline
    """
    return offset - depth * (gamma**2 / ((f - f0)**2 + gamma**2))


# def extract_odmr_peak_frequencies(spectrum, frequencies, num_peaks=8, distance=5, prominence=0.1):
#     """
#     Extract ODMR peaks positions (frequencies) from a spectrum, using find_peaks and Lorentzian fitting.
#     Parameters:
#         spectrum:    1D array of signal values (N_freq,)
#         frequencies: 1D array of frequency points (Hz or GHz)
#         num_peaks:   number of peaks to extract (default 8 i.e 2 per NV-axis)
#         distance:    minimum distance between peaks (in index units)
#         prominence:  minimum prominence for peak detection
#     Returns:
#         1D numpy array of fitted peak frequencies (Hz)
#     """
#     # Convert to numpy if tensor
#     if torch.is_tensor(spectrum):
#         spectrum = spectrum.cpu().numpy()
#     if torch.is_tensor(frequencies):
#         frequencies = frequencies.cpu().numpy()

#     # Invert spectrum to detect dips as peaks
#     inverted = np.max(spectrum) - spectrum
    
#     peaks, props = find_peaks(inverted, distance=distance, prominence=prominence)
#     # Sort peaks by prominence (strongest first)
#     prominences = props['prominences'] if 'prominences' in props else np.ones_like(peaks)
#     sorted_idx = np.argsort(prominences)[::-1]
#     peaks = peaks[sorted_idx][:num_peaks]
#     peak_freqs = []
#     for idx in peaks:
#         # Fit Lorentzian around each peak
#         window = 7  # points on each side
#         left = max(0, idx - window)
#         right = min(len(frequencies), idx + window + 1)
#         x = frequencies[left:right]
#         y = spectrum[left:right]
#         # Initial guess: x0=peak, gamma=half-width, A=depth, y0=baseline
#         x0 = frequencies[idx]
#         gamma = (frequencies[1] - frequencies[0]) * 3
#         A = spectrum[idx] - np.median(spectrum)
#         y0 = np.median(spectrum)
#         try:
#             popt, _ = curve_fit(lorentzian, x, y, p0=[x0, gamma, A, y0], maxfev=2000)
#             peak_freqs.append(popt[0])
#         except Exception:
#             peak_freqs.append(x0)
#     # Sort by frequency
#     return sorted(peak_freqs)

def extract_odmr_peak_frequencies(spectrum, freqs, num_peaks=8, distance=5, prominence_factor=0.3):
    """
    Extract ODMR peak frequencies from a spectrum using Lorentzian fitting.

    Parameters:
        spectrum: 1D array or tensor of shape (N_freq,)
        freqs: 1D array or tensor of frequency points (Hz)
        num_peaks: number of peaks to extract (default 8)
        distance: minimum distance between peaks (in index units)
        prominence_factor: factor to compute min prominence from spectrum std

    Returns:
        1D numpy array of fitted peak frequencies (Hz), length=num_peaks
    """
    # Convert to numpy if tensor
    if torch.is_tensor(spectrum):
        spectrum = spectrum.cpu().numpy()
    if torch.is_tensor(freqs):
        freqs = freqs.cpu().numpy()

    # Invert spectrum to detect dips
    inverted = np.max(spectrum) - spectrum

    # Compute prominence threshold
    prom_thresh = np.std(inverted) * prominence_factor

    # Rough peak detection
    peaks_idx, props = find_peaks(inverted, distance=distance, prominence=prom_thresh)
    if len(peaks_idx) == 0:
        return np.full(num_peaks, np.nan)

    # Sort peaks by prominence (strongest first)
    prominences = props['prominences'] if 'prominences' in props else inverted[peaks_idx]
    sorted_idx = peaks_idx[np.argsort(prominences)[::-1]]

    # Take top num_peaks
    top_peaks = sorted_idx[:num_peaks]

    fitted_freqs = []
    window = 7  # points on each side for Lorentzian fit

    for idx in top_peaks:
        left = max(0, idx - window)
        right = min(len(freqs), idx + window + 1)
        x_local = freqs[left:right]
        y_local = spectrum[left:right]

        # Initial guess: f0=peak, gamma=3*freq step, A=depth, offset=baseline
        f0_guess = freqs[idx]
        gamma_guess = (freqs[1] - freqs[0]) * 3
        center_idx = idx - left  # index of the peak in the window
        if 0 <= center_idx < len(y_local):
            A_guess = np.median(y_local) - y_local[center_idx]  # approximate dip depth
        else:
            A_guess = np.median(y_local) - np.min(y_local)
        offset_guess = np.max(y_local)

        try:
            popt, _ = curve_fit(lorentzian, x_local, y_local,
                                p0=[f0_guess, gamma_guess, A_guess, offset_guess],
                                maxfev=5000)
            fitted_freqs.append(popt[0])
        except:
            fitted_freqs.append(f0_guess)

    # Sort fitted frequencies ascending
    fitted_freqs = np.sort(fitted_freqs)

    # Pad if less than num_peaks
    if len(fitted_freqs) < num_peaks:
        fitted_freqs = np.pad(fitted_freqs, (0, num_peaks - len(fitted_freqs)), constant_values=np.nan)

    return np.array(fitted_freqs)


def extract_odmr_peak_frequencies(
    spectrum,
    freqs,
    num_peaks=8,
    distance=5,
    prominence_factor=0.3,
    smooth=True,
    smooth_window=7,
    smooth_poly=2,
):
    """
    Extract ODMR peak frequencies from a spectrum using Lorentzian fitting.

    Parameters:
        spectrum: 1D array or tensor of shape (N_freq,)
        freqs: 1D array or tensor of frequency points (Hz)
        num_peaks: number of peaks to extract (default 8)
        distance: minimum distance between peaks (in index units)
        prominence_factor: factor to compute min prominence from spectrum std
        smooth: if True, apply mild smoothing before peak detection
        smooth_window: window length for smoothing (odd integer, in points)
        smooth_poly: polynomial order for Savitzky–Golay smoothing

    Returns:
        1D numpy array of fitted peak frequencies (Hz), length=num_peaks
    """
    from scipy.signal import savgol_filter

    # Convert to numpy if tensor
    if torch.is_tensor(spectrum):
        spectrum = spectrum.cpu().numpy()
    if torch.is_tensor(freqs):
        freqs = freqs.cpu().numpy()

    # Optional mild smoothing (only for peak detection/fitting)
    spec_for_peaks = spectrum.copy()
    if smooth and len(spec_for_peaks) > smooth_window:
        # Ensure odd window length and <= length of spectrum
        if smooth_window % 2 == 0:
            smooth_window += 1
        smooth_window = min(smooth_window, len(spec_for_peaks) - (1 - len(spec_for_peaks) % 2))
        if smooth_window >= smooth_poly + 2:  # basic safety
            try:
                spec_for_peaks = savgol_filter(spec_for_peaks, smooth_window, smooth_poly)
            except Exception:
                # Fallback: keep original spectrum if smoothing fails
                spec_for_peaks = spectrum.copy()

    # Invert spectrum to detect dips (works on z-scored or raw spectra)
    inverted = np.max(spec_for_peaks) - spec_for_peaks
    inv_range = float(inverted.max() - inverted.min())

    # Prominence: combine std-based and dynamic-range-based thresholds (normalized spectra)
    prom_thresh = max(
        np.std(inverted) * prominence_factor,
        inv_range * 0.06,
        1e-8,
    )

    # ---- Rough peak detection (1st pass) ----
    peaks_idx, props = find_peaks(inverted, distance=distance, prominence=prom_thresh)

    # Relax criteria progressively if too few dips are found
    for relax in (0.5, 0.25):
        if len(peaks_idx) >= num_peaks:
            break
        relaxed_prom = prom_thresh * relax
        relaxed_dist = max(1, distance // 2)
        peaks_idx2, props2 = find_peaks(inverted, distance=relaxed_dist, prominence=relaxed_prom)
        if len(peaks_idx2) > 0:
            peaks_idx = np.unique(np.concatenate([peaks_idx, peaks_idx2]))
            if "prominences" in props2:
                prom_map = {int(i): float(p) for i, p in zip(peaks_idx2, props2["prominences"])}
                prominences_full = np.array([prom_map.get(int(i), inverted[int(i)]) for i in peaks_idx])
                props = {"prominences": prominences_full}

    if len(peaks_idx) == 0:
        return np.full(num_peaks, np.nan)

    # Sort peaks by prominence (strongest first)
    prominences = props['prominences'] if 'prominences' in props else inverted[peaks_idx]
    sorted_idx = peaks_idx[np.argsort(prominences)[::-1]]

    # Take top num_peaks
    top_peaks = sorted_idx[:num_peaks]

    fitted_freqs = []
    window = 7  # points on each side around detected peak index

    for idx in top_peaks:
        left = max(0, idx - window)
        right = min(len(freqs), idx + window + 1)

        # Use the original (unsmoothed) spectrum to find the local minimum
        y_local = spectrum[left:right]
        if len(y_local) == 0:
            continue

        # Take the index of the deepest part of the dip inside this local window
        local_min_idx = int(np.argmin(y_local))
        center_idx = left + local_min_idx
        center_idx = np.clip(center_idx, 0, len(freqs) - 1)
        center_freq = freqs[center_idx]

        fitted_freqs.append(center_freq)

    # Keep prominence order (no global sort) — slot matching handles alignment
    fitted_freqs = np.array(fitted_freqs, dtype=np.float64)

    # Pad if less than num_peaks
    if len(fitted_freqs) < num_peaks:
        fitted_freqs = np.pad(
            fitted_freqs, (0, num_peaks - len(fitted_freqs)), constant_values=np.nan,
        )
    else:
        fitted_freqs = fitted_freqs[:num_peaks]

    return fitted_freqs