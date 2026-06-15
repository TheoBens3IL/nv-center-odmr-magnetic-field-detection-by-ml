import torch
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


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

def physics_loss(B_pred, measured_freqs):
    f_pred = nv_transitions_4axes(B_pred) / 1e9 # (batch, 8) converted to GHz

    # Convert measured_freqs to tensor if needed, and move to same device as f_pred
    if not torch.is_tensor(measured_freqs):
        measured_freqs = torch.tensor(measured_freqs, device=f_pred.device)
    else:
        measured_freqs = measured_freqs.detach().clone().to(f_pred.device) # ensure it's on the same device and not part of the graph
        
    measured_freqs = measured_freqs / 1e9  # convert to GHz

    # Align with peak extraction: measured peaks are sorted by frequency, while f_pred
    # follows (NV axis k, f_minus, f_plus). Compare sorted sequences so slot i matches
    # the i-th smallest transition frequency (NaNs from partial detection sort to the end).
    # keep only valid entries (non-NaN) for loss calculation
    f_sorted = torch.sort(f_pred, dim=1).values
    m_sorted = torch.sort(measured_freqs, dim=1).values
    mask = ~torch.isnan(m_sorted)
    diff = (f_sorted - m_sorted)[mask]

    # If no valid entries, return zero loss
    if diff.numel() == 0:
        return torch.tensor(0.0, device=f_pred.device, dtype=f_pred.dtype)

    return torch.mean(diff ** 2)


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

    # Invert spectrum to detect dips
    inverted = np.max(spec_for_peaks) - spec_for_peaks

    # Compute prominence threshold
    prom_thresh = np.std(inverted) * prominence_factor

    # ---- Rough peak detection (1st pass) ----
    peaks_idx, props = find_peaks(inverted, distance=distance, prominence=prom_thresh)

    # If we detect too few peaks, relax the criteria and retry once
    if len(peaks_idx) < num_peaks:
        relaxed_prom = prom_thresh * 0.5
        relaxed_dist = max(1, distance // 2)
        peaks_idx2, props2 = find_peaks(inverted, distance=relaxed_dist, prominence=relaxed_prom)
        # Merge unique indices from both passes
        if len(peaks_idx2) > 0:
            peaks_idx = np.unique(np.concatenate([peaks_idx, peaks_idx2]))
            # Use prominences from the more permissive pass when available
            if 'prominences' in props2:
                # Build a simple prominence array aligned with peaks_idx (fallback to inverted value)
                prom_map = {int(i): float(p) for i, p in zip(peaks_idx2, props2['prominences'])}
                prominences_full = np.array([prom_map.get(int(i), inverted[int(i)]) for i in peaks_idx])
                props = {'prominences': prominences_full}

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

    # Sort fitted frequencies ascending
    fitted_freqs = np.sort(fitted_freqs)

    # Pad if less than num_peaks
    if len(fitted_freqs) < num_peaks:
        fitted_freqs = np.pad(fitted_freqs, (0, num_peaks - len(fitted_freqs)), constant_values=np.nan)

    return np.array(fitted_freqs)