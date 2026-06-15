import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Global parameters
# Wstart = 2.7e3
# Wend = 3.05e3
Wstart = 2.77e3
Wend = 2.97e3
Wnr = 201
D = 2.87e3
gamma_e = 28.024e3  # MHz/T

# Spin operators S=1
Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128)
Sz2 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.complex128)
S0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.complex128)

# Principal Hamiltonian
def H0_e(Bz=0, W_mw=2.87e3):
    return D * Sz2 + gamma_e * Bz * Sz - W_mw * Sz2

# Interaction Hamiltonian (simplified, vectorized)
def Hint_e(phase, Ox, coef_x, Oy, coef_y):
    Ax = Ox * coef_x[0] / (2 * np.sqrt(2))
    Ay = Ox * coef_x[1] / (2 * np.sqrt(2))
    Bx = Oy * coef_y[0] / (2 * np.sqrt(2))
    By = Oy * coef_y[1] / (2 * np.sqrt(2))
    H = np.zeros((3, 3), dtype=np.complex128)
    H[0, 1] = Ax - 1j * Ay + (By + 1j * Bx) * np.exp(-1j * phase)
    H[1, 0] = Ax + 1j * Ay + (By - 1j * Bx) * np.exp(1j * phase)
    H[1, 2] = Ax - 1j * Ay + (-By - 1j * Bx) * np.exp(1j * phase)
    H[2, 1] = Ax + 1j * Ay + (-By + 1j * Bx) * np.exp(-1j * phase)
    return H


# Mean population (state 0)
def mean_pop_e(t=2, Bz=1e-3, W_mw=2.87e3, phase=0, Ox=1, coef_x=None, Oy=1, coef_y=None, n_time_points=10):
    H = H0_e(Bz, W_mw) + Hint_e(phase, Ox, coef_x, Oy, coef_y)
    tlist = np.linspace(0, t, n_time_points)
    U = np.array([expm(-1j * H * tt) for tt in tlist])
    proj0 = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.complex128)
    states = np.array([U_t @ proj0 @ U_t.conj().T for U_t in U])
    PiPulsePop0 = np.mean(np.real(states[:, 1, 1]))
    return PiPulsePop0

# Generation of the ODMR spectrum for a single NV
def single_nv_spectrum(Bz, phase, Ox, coef_x, Oy, coef_y, freq_list=None):
    if freq_list is None:
        freq_list = np.linspace(Wstart, Wend, Wnr)
    pops = [mean_pop_e(t=5, Bz=Bz, W_mw=w, phase=phase, Ox=Ox, coef_x=coef_x, Oy=Oy, coef_y=coef_y, n_time_points=10) for w in freq_list]
    pops = np.array(pops)
    return pops / np.max(pops)

# --- Rotation and Transformation Functions ---
def rot_x(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(beta):
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def extract_numeric_coefficients(alpha, beta):
    TlabNV = [
        np.array([[1/np.sqrt(3), 0,  np.sqrt(2/3)], [0, 1, 0], [-np.sqrt(2/3), 0, 1/np.sqrt(3)]]),
        np.array([[1/np.sqrt(3), 0, -np.sqrt(2/3)], [0, 1, 0], [ np.sqrt(2/3), 0, 1/np.sqrt(3)]]),
        np.array([[1, 0, 0], [0, 1/np.sqrt(3), -np.sqrt(2/3)], [0, np.sqrt(2/3),  1/np.sqrt(3)]]),
        np.array([[1, 0, 0], [0, 1/np.sqrt(3),  np.sqrt(2/3)], [0, -np.sqrt(2/3), 1/np.sqrt(3)]])
    ]
    vx = (1/np.sqrt(2)) * np.array([1, -1, 0])
    vy = (1/np.sqrt(2)) * np.array([1,  1, 0])
    Rtot = rot_x(alpha) @ rot_y(beta)
    R_T = Rtot.T
    results = []
    for T in TlabNV:
        coeff_x = T @ R_T @ vx
        coeff_y = T @ R_T @ vy
        results.append((coeff_x, coeff_y))
    return results

def transform_field_toNV(B, alpha=0, beta=0):
    Bx_NV = np.sqrt(1/2)*(B[0]+B[1])
    By_NV = np.sqrt(1/2)*(-B[0]+B[1])
    B_column = np.array([[Bx_NV], [By_NV], [B[2]]])
    Rtot = rot_x(alpha) @ rot_y(beta)
    TlabNV = [
        np.array([[1/np.sqrt(3), 0, np.sqrt(2/3)], [0,  1,  0], [-np.sqrt(2/3), 0,  1/np.sqrt(3)]]),
        np.array([[1/np.sqrt(3), 0, -np.sqrt(2/3)], [0,  1,  0], [np.sqrt(2/3), 0,  1/np.sqrt(3)]]),
        np.array([[1, 0, 0], [0, 1/np.sqrt(3), -np.sqrt(2/3)], [0, np.sqrt(2/3), 1/np.sqrt(3)]]),
        np.array([[1, 0, 0], [0, 1/np.sqrt(3), np.sqrt(2/3)], [0, -np.sqrt(2/3), 1/np.sqrt(3)]])
    ]
    B_in_NV_basis = []
    for T in TlabNV:
        T_rot = T @ Rtot.T
        B_nv = T_rot @ B_column
        B_in_NV_basis.append(B_nv[:,0])
    return B_in_NV_basis

# Ensemble spectrum for the 4 NV orientations
def ensemble_spectrum(B, MW_field, MW_phase, freq_list=None, tilt_x=0, tilt_y=0):
    if freq_list is None:
        freq_list = np.linspace(Wstart, Wend, Wnr)
    B_inNV = transform_field_toNV(B, tilt_x, tilt_y)
    Omx, Omy = MW_field[0], MW_field[1]
    num_coeffs = extract_numeric_coefficients(tilt_x, tilt_y)
    EnsCont = np.zeros(len(freq_list))
    for idx, B_nv in enumerate(B_inNV):
        Bz = B_nv[2]
        coef_x, coef_y = num_coeffs[idx]
        Contrast = single_nv_spectrum(Bz, MW_phase, Omx, coef_x, Omy, coef_y, freq_list)
        EnsCont += Contrast
    EnsCont /= 4
    return EnsCont


# Test and plot
if __name__ == "__main__":
    import time
    freqList = np.linspace(Wstart, Wend, Wnr)
    B_vec = [0.00145, 0.00145, 0.00291] # highest field
    # B_vec = [0.003, 0.000, 0.000]
    MW_vec = [0.9, 0.5, 0.0]
    MW_phase = np.deg2rad(30)
    tilt_x = np.deg2rad(10.0)
    tilt_y = np.deg2rad(5.0)
    t0 = time.time()
    spectrum = ensemble_spectrum(B_vec, MW_vec, MW_phase, freqList, tilt_x, tilt_y)
    t1 = time.time()
    print(f"Execution time (optimized): {t1-t0:.3f} s")
    plt.plot(freqList, spectrum)
    plt.show()
