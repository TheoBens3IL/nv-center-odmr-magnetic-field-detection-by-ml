
"""
Created on Wed Sep 10 15:30 2025

@author: julia
"""
#%%
import numpy as np
import pandas as pd
from itertools import permutations, product
import os
import sys


# Establece el nuevo directorio de trabajo
directorio_cfm = r'C:\Users\jberterod\Documents\GitHub\PhD-NVcenters-QNanoLab-magnetometry\NV_magnetometry_fexp_to_Blab\Code'
directorio_lenovo = r'C:\Users\julia\Desktop\Juli\PhD QNanoLab\PhD-NVcenters-QNanoLab-magnetometry\PhD-NVcenters-QNanoLab-magnetometry\NV_magnetometry_fexp_to_Blab\Code'
nuevo_directorio = directorio_cfm
os.chdir(nuevo_directorio)  # Cambia el directorio actual

import simulate_TStrength_NV_ensemble_modificado as simnv
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#%%
# constants
gamma = 2.8*1e6 #Hz / Gauss
E = 3.5*1e6   # Hz
D = 2871.5*1e6 # Hz
theta = np.pi/4

# MW frequency
nfreq = 2000 #Hz
freqi = 2860*1e6 #Hz
freqf = 2890*1e6 #Hz
MWfreq = np.linspace(freqi, freqf, nfreq)
# Linewidth
Linewidth = 3e6 #Hz, equiv a 3 MHz

# Define MW directions in lab frame
thetaMW = np.pi/3.
phiMW = np.pi/4.

MWvec = simnv.transform_all_frames(5, thetaMW, phiMW)

def get_vector_spherical(Avec):
    """ Compute spherical coordinates of a vector fron its cartesian
    coordinates """
    A0 = np.sqrt(np.dot(Avec, Avec))
    theta = np.arccos(Avec[2] / A0)
    try:
        phi = np.arctan(Avec[1] / Avec[0])
    except ZeroDivisionError:
        phi = 0.
    if np.isnan(phi):
        phi = 0.
    if Avec[0] < 0:
        phi += np.pi
    return A0, theta, phi

def get_rotation_matrix(idx_nv): #config <111> del NV
    """ Returns the transformation matrix from lab frame to the desired 
    NV frame, identified by idx_nv (can be 1, 2, 3 or 4) """
    if idx_nv==4:
        RNV = np.array([[1/np.sqrt(6), -1/np.sqrt(6), -2/np.sqrt(6)],
                        [1/np.sqrt(2),  1/np.sqrt(2),  0],
                        [1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3)]])
    elif idx_nv==1:
        RNV = np.array([[-1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
                        [-1/np.sqrt(2), -1/np.sqrt(2),  0],
                        [-1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)]])
    
    elif idx_nv==3:
        RNV = np.array([[-1/np.sqrt(6), -1/np.sqrt(6),  2/np.sqrt(6)],
                        [-1/np.sqrt(2),  1/np.sqrt(2),  0],
                        [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]])
    elif idx_nv==2:
        RNV = np.array([[1/np.sqrt(6),  1/np.sqrt(6),  2/np.sqrt(6)],
                        [1/np.sqrt(2), -1/np.sqrt(2),  0],
                        [1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3)]])
    else:
        raise ValueError('Invalid index of NV orientation')
    
    return RNV

K_matrix = np.array([
    get_rotation_matrix(1)[-1],
    get_rotation_matrix(2)[-1],
    get_rotation_matrix(3)[-1],
    get_rotation_matrix(4)[-1]
])
#calculo num de picos significativos que espero ver segun la aplitud de absorcion
def lorentzian(x, x0, gamma):
    """Lorentzian lineshape centered at x0 with FWHM = gamma"""
    return (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2)


def Bnv_to_Blab_qr(Bnv):
    """
    Transform the magnetic field vector from the NV frame to the laboratory frame
    using a QR decomposition.

    Parameters
    ----------
    Bnv : array-like
        Magnetic field vector expressed in the NV frame.
    theta : float
        Angle in radians used to build the transformation matrix.

    Returns
    -------
    Bx, By, Bz : floats
        Components of the magnetic field vector in the lab frame.
    """
    
    # Perform QR decomposition of the transformation matrix K(theta)
    Q, R = np.linalg.qr(K_matrix, mode='reduced')
    
    # Solve for Blab using the relation K * Blab = Bnv
    Bx, By, Bz = np.linalg.inv(R) @ (Q.T @ Bnv)
    
    return Bx, By, Bz  # return components of Blab


"funcion que calcula |B| dado Bi"

def Bi_to_Bmag(B):
    #separo en casos depende de si es Blab o Bnv
    if len(B)==4: #caso B = Bnv
        B1, B2, B3, B4 = B
        Bmag = (3/4) * np.sqrt(B1**2 + B2**2 + B3**2 + B4**2) 
    
    elif len(B) == 3: #caso B = Blab
        Bx, By, Bz = B
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        
    return(Bmag)



"funcion que calcula |B| dado fi"
def fi_to_Bmag(fi):
    #separo en casos
    f_ordered = np.sort(fi*1e-6)#ordeno las freq y las escribo en MHz
    df = f_ordered[-4:][::-1] - f_ordered[:4]
    Bmag = (3/4) * np.sqrt( np.sum(df**2)/4 - E**2)/ gamma
    
    return(Bmag)



#%%
# pasos para passar de fi_exp -> Bnv -> Blab -> fi_simu -> graph
#cada flecha es una funcion que empieza como step...

def step1_fi_to_Bnv(fi, order, signs):
    
    """
    calculates |Bi| components in nv frame (not ordered and no sign)
    given the resonace freq obtained from measurement

    fi en Hz
    """

    #ordeno freq
    f_ordered = np.sort(fi)
    # f_ordered = fi
    
    #calculo dOmega
    domega_1 = f_ordered[-1]-f_ordered[0]
    domega_2 = f_ordered[-2]-f_ordered[1]
    domega_3 = f_ordered[-3]-f_ordered[2]
    domega_4 = f_ordered[-4]-f_ordered[3]
    domega = np.array([domega_1,domega_2,domega_3,domega_4]) #Hz

    #calculo Bnv componenete de B en los ejes del NV
    Bnv = np.sqrt((domega/2)**2-E**2)/gamma 
    
    #ordeno segun order y signs
    Bnv_ordered = np.array([Bnv[i-1] for i in order]) #reordeno
    Bnv_signed_ordered = Bnv_ordered* np.array(signs) #elijo signos
    
    return(Bnv_signed_ordered)
    

def step2_Bnv_to_Blab(Bnv):
    """
    use Bnv (B in nv frame) to calculate Blab (B in lab frame) by solving
    Bnv = K(theta) @ Blab where theta = pi/4 because we are in <100> config of NV
    """
    
    
    # Transform to lab frame
    Bx, By, Bz = Bnv_to_Blab_qr(Bnv)

    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2) #calculo |Blab|
    
    return(Bx, By, Bz, B_mag)

def step3_Blab_to_fi_Simu_3state (Blab): #calculo solo para 3 estados
    """
    Parameters
    ----------
    Blab : for each row in Blab corresponding to a possible Blab = Bx, By, Bz 
    it uses the simulation to calculate fi and Ta

    Returns
    -------
    f_i, TA, f_i_filtered, TA_filtered
    f_i : list of transition frequencies (differences between eigenvalues of the Hamiltonian).
    TA  : list of transition amplitudes (|<f|Hint|i>|^2), which quantify the transition strength
          induced by the microwave probe.
    """
    Bx, By, Bz = Blab[:3]
    # B_mag = Blab[-1]
    
    Bvec_lab = ([Bx,By,Bz])
    B0, thetaB, phiB = simnv.get_vector_spherical(Bvec_lab)
    
    #calculo lista para todas las transf
    Bvector_list = simnv.transform_all_frames(B0,thetaB, phiB) #lista de 4 elementos, donde cada elemento es un vector 3D (Bx, By, Bz) pero en el sistema de referencia de ese NV.
    nu_list = []
    TA_list = []
    
    for j in range(4): #calculo las freq para cada orientacion del NV
        # print(j)
        Bvec = Bvector_list[j]
        
        #considering only 3 states:
        E_I, vec_I, H_total = simnv.NV_transitionsElevels_noE_noNuc_3state(Bvec) #toma campo en gauss y devuelve en MHz.
        Hint = simnv.NV_GS_Hamiltonian_MWprobe_noNuc_3state(MWvec)
        
        for i in range(3):
            for f in range(i+1, 3):
                
                nu = (E_I[f] - E_I[i])              # transition frequency 
                TME = (vec_I[:, f].conj().T @ Hint @ vec_I[:, i])
                TA  = abs(TME)**2                  # transition amplitude
                if i == 0:#pido que solo guarde transicion 1->2 (nu1) y 1->3 (nu2)
                    # print('i = '+str(i))
                    # print('f = '+str(f)+'\n')
                    nu_list.append(nu*1e6)
                    TA_list.append(TA)
                
    nu_tot = np.array(nu_list) 
    TA_tot = np.array(TA_list)         
    return (nu_tot, TA_tot) 


nv_coords = {
    "NV1": (1, -1, 1),
    "NV2": (-1, 1, 1),
    "NV3": (-1, -1, -1),
    "NV4": (1, 1, -1)
}


#%% Grafico ambos: 1) 3D con B proyectado sobre los NVs y 2) ESR spectrum

def step4_plot_Bnv_frame_and_ESR(Blab_exp, f0s, order, signs, nv_graph, esr_simu, Linewidth_data):
    """
    Plot 
    1) projection of B vectors in the NV coordinate frame.
    2) NV simulated absorption spectrum

    Parameters:
    - f0s: list or array of 8 scalars: freqs measured
    - order: list of indices to order the B vectors (e.g., [1, 2, 3, 4])
    - signs: list of +1/-1 to indicate sign conventions for Blab (e.g., [1, -1, -1, 1])
    """
    #-------------------- calculo nu y TA usando f0 ----------------------#
    #Step1: Calculate Bnv using f0s
    Bnv = step1_fi_to_Bnv(f0s, order, signs) 
    B1, B2, B3, B4 = Bnv

    # Step 2: Calculate Blab from Bnv, order, and signs
    Blab = step2_Bnv_to_Blab(Bnv)[:3]
    Bx, By, Bz = Blab[:3]
    # Step 3: Simulate expected resonance frequencies and transition amplitudes
    nu_tot, TA_tot = step3_Blab_to_fi_Simu_3state(np.array(Blab))

    # Step 4: Simulate absorption spectrum "fluo"
    f_min, f_max = min(nu_tot) - 50e6, max(nu_tot) + 50e6   # ±50 MHz margin
    freqs = np.linspace(f_min, f_max, int(1e6))
    # Build absorption spectrum
    spectrum = np.zeros_like(freqs)
    for nu, TA_val, Linewidth in zip(nu_tot, TA_tot, Linewidth_data):
        spectrum += TA_val * lorentzian(freqs, nu, Linewidth)

    # Normalize
    fluo = 1 - (spectrum / np.max(spectrum))

    
    #-------------------- 1) grafico NV ----------------------#
    if nv_graph ==1:

        # Colors and labels for plotting
        colors = ['r', 'darkviolet', 'b', 'green', 'k']
        labels = ['B1', 'B2', 'B3', 'B4', 'Blab'] 
        
        # Create 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Plot vectors from the origin in the direction of each NV axis
        origin = np.array([0, 0, 0])
        
        # Plot Blab vector in black from the origin
        ax.quiver(*origin, *Blab[:3], color='k', label='Blab', linewidth=2)
        
        # Plot Blab experimental vector in black from the origin
         
        ax.quiver(*origin, *Blab_exp[:3], color='gray', label='Blab exp', linewidth=2)
        
        for nv, coord in nv_coords.items():
            pos = int(nv[-1]) - 1  # Extract NV number (1-based) and convert to index (0-based)
            Bnv_i = Bnv[pos]
            direction = np.array(coord) * np.abs(Bnv_i) * signs[pos]
            ax.quiver(*origin, *direction, color=colors[pos], label=labels[pos])
    
        # Draw coordinate axes
        ax.plot([20, 20], [0, 0], [0, 0], color='gray', linewidth=2)   # X axis
        ax.plot([0, 0], [-20, 20], [0, 0], color='gray', linewidth=2)  # Y axis
        ax.plot([0, 0], [0, 0], [-20, 20], color='gray', linewidth=2)  # Z axis
    
        # Set axis labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Projection of B onto the NV frame axes")
        ax.legend(loc='lower left')
    
    
        # Add text annotations in 3D space
        text_order = f'order = {order}'
        text_signs = f'signs = {signs}'
        
        # Adjust these coordinates based on how the plot looks
        ax.text(-23, -25, 50, text_order, 
                verticalalignment='top', horizontalalignment='left', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5))
    
        ax.text(-23, -25, 40, text_signs, 
                verticalalignment='top', horizontalalignment='left', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5))
    
    
        # Equal scale for all axes
        max_range = 25
        for axis in 'xyz':
            getattr(ax, f'set_{axis}lim')([-max_range, max_range])
        # if save_NV3D_graph != 0:
        #     filename = f'{save_NV3D_graph}.png'
        #     plt.savefig(filename, dpi=300, bbox_inches='tight')  # ajusta calidad y bordes
            
        plt.show()
    
    #-------------------- 2) grafico espectro de absorcion ----------------------#
    # Plot
    if esr_simu ==1:
        plt.figure(figsize=(8,5))
        plt.plot(freqs/1e6, fluo, 'k-')  # GHz
        plt.xlim(2750, 2950) 
            
        for freq in (nu_tot[0],nu_tot[1]):
            plt.axvline(x=freq*1e-6, color = 'red')
        for freq in (nu_tot[2],nu_tot[3]):
            plt.axvline(x=freq*1e-6, color = 'darkviolet')
        for freq in (nu_tot[4],nu_tot[5]):
            plt.axvline(x=freq*1e-6, color = 'blue')
        for freq in (nu_tot[6],nu_tot[7]):
            plt.axvline(x=freq*1e-6, color = 'green')

        textstr = f'Bx = {Bx:.2f}\nBy = {By:.2f}\nBz = {Bz:.2f}'
        plt.text(2765, 0.8, textstr, verticalalignment='top',
                  horizontalalignment='left', fontsize=12,
                  bbox=dict(facecolor='white', alpha=0.5))
        
        textstr = f'order = {order}'
        plt.text(2765, 0.4, textstr, verticalalignment='top',
                  horizontalalignment='left', fontsize=12,
                  bbox=dict(facecolor='white', alpha=0.5))
        
        textstr = f'signs = {signs}'
        plt.text(2765, 0.3, textstr, verticalalignment='top',
                  horizontalalignment='left', fontsize=12,
                  bbox=dict(facecolor='white', alpha=0.5))
    
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Fluorescence absorption (arb. units)")
        plt.title("Simulated ESR Fluorescence Spectrum no Nuclear F")
        plt.grid(True)
        plt.show()

# # %%
# # Example with 1 order and signs chosen
# #data
# order = ([3,1,4,2]) # rojo / vio / azul / verde
# signs = ([-1,1,1,-1])
# f0s = np.array([
#     2.886183,  # Dip 1
#     2.815595,  # Dip 2
#     2.859650,  # Dip 3
#     2.926755,  # Dip 4
#     2.803049,  # Dip 5
#     2.847305,  # Dip 6
#     2.898322,  # Dip 7
#     2.9390   # Dip 8
# ])*1e3 #MHz

# f0s_sorted = np.sort(f0s)*1e6
# fexp_ordered = np.sort(f0s)*1e6 #ordeno freq medidas experimentalmente y paso a Hz.

# Blab_exp = ([-31,10,5])

# Bnv = step1_fi_to_Bnv(f0s, order, signs) #la funcion ordena las f0 asi que pueden estar en cualquier orden
# Blab = step2_Bnv_to_Blab(Bnv)[:3]
# nu_tot, TA_tot = step3_Blab_to_fi_Simu_3state(np.array(Blab))
# step4_plot_Bnv_frame_and_ESR(Blab_exp, f0s, order, signs, nv_graph = 1, esr_simu = 1)


# %%
