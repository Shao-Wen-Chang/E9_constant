import sys
from pathlib import Path

# User defined modules
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if str(E9path) not in sys.path:
    sys.path.insert(1, str(E9path))
from E9_fn.E9_constants import *
import numpy as np
import matplotlib.pyplot as plt
from E9_fn.util import quadrupole_Bfield_vec

def BFieldAtZSlice(z, rad_axis, coil_coeff, I):
    """Returns (3 x n) array of B field values at a fixed z over n points defined by rad_axis."""
    pos = np.vstack((rad_axis, np.zeros_like(rad_axis), z * np.ones_like(rad_axis)))
    return quadrupole_Bfield_vec(pos, coil_coeff * I)

def BVecGradAtPos(pos, vec, coil_coeff, I):
    """Returns [(vec grad) pos] as a 3-element array at pos.
    
    Note that this implementation assumes that the field is smooth within 1 um."""
    vecum = 0.5e-6 * vec / np.linalg.norm(vec)
    B1 = quadrupole_Bfield_vec(pos + vecum, coil_coeff * I)
    B2 = quadrupole_Bfield_vec(pos - vecum, coil_coeff * I)
    return (B1 - B2) / 1e-6

def sB(a_lat, B_offset, B_grad, B_var):
    """Figure of merit for B field stability for single layer / site selection, see [Dydiowa].
    
    a_lat: lattice spacing
    B_offset: strength of offset field at trap center
    B_grad: B field gradient, assumed to be constant
    B_var: unwanted spatial variation of B field strength within a single layer due to curvature etc."""
    return (a_lat * B_grad - B_var) / B_offset

def BVar(coil_coeff, I_FB, cloud_size, B_offset):
    """Returns the variation of |B| within a layer.
    
    Note that practically cloud_size should be a funcion of I_FB.
    It is assumed that the cloud is at the trap center ([0, 0, 0])."""
    return abs( abs(np.linalg.norm(quadrupole_Bfield_vec(np.array([0, 0, 0]), coil_coeff * I_FB) + np.array([0, 0, B_offset]))) \
        - abs(np.linalg.norm(quadrupole_Bfield_vec(np.array([cloud_size, 0, 0]), coil_coeff * I_FB) + np.array([0, 0, B_offset]))) )

def sBandBo_Bgrads(a_lat, I_FBs, coil_coeff, cloud_size):
    """Returns B_offset's that maximizes sB, and the resulting sB's, for gradients given in B_grads.
    
    a_lat: lattice spacing
    I_FBs: a list of FB coil currents"""
    B_offsets = np.arange(10, 100, 0.25) * 1e-4
    sB_best = np.zeros_like(I_FBs, dtype = np.float64)
    Bo_best = np.zeros_like(I_FBs, dtype = np.float64)
    for i, I_now in enumerate(I_FBs):
        B_zGrad = abs(BVecGradAtPos(np.array([0, 0, 0]), np.array([0, 0, 1]), coil_coeff, I_now)[2])
        B_rvars = np.zeros_like(B_offsets)
        sB_now = np.zeros_like(B_offsets)
        for j, Bo in enumerate(B_offsets):
            B_rvars[j] = BVar(coil_coeff, I_now, cloud_size, Bo)
            sB_now[j] = sB(a_lat, Bo, B_zGrad, B_rvars[j])
        sB_best[i] = sB_now.max()
        Bo_best[i] = B_offsets[sB_now.argmax()]
    return sB_best, Bo_best

#%% find sB
plt.close('all')
I_FB = 120
B_offsets = np.arange(10, 100) * 1e-4 # range of offset field to test
cloud_size = 100e-6 # (radial) radius of the cloud
B_zGrad = abs(BVecGradAtPos(np.array([0, 0, 0]), np.array([0, 0, 1]), FBcoil_coeff, I_FB)[2])
B_rvars = np.zeros_like(B_offsets)
sB_result = np.zeros_like(B_offsets)
for i, Bo in enumerate(B_offsets):
    B_rvars[i] = BVar(FBcoil_coeff, I_FB, cloud_size, Bo)
    sB_result[i] = sB(a_vert, Bo, B_zGrad, B_rvars[i])

fig_sB = plt.figure(1)
fig_sB.clf()
ax_Bvar = fig_sB.add_subplot(121)
ax_Bvar.plot(B_offsets * 1e4, B_rvars)
ax_Bvar.set_xlabel('$B_{offset}$ [G]')
ax_Bvar.set_ylabel('$B_{var}$')

ax_sB = fig_sB.add_subplot(122)
ax_sB.plot(B_offsets * 1e4, sB_result)
ax_sB.set_xlabel('$B_{offset}$ [G]')
ax_sB.set_ylabel('$s_B$')
fig_sB.tight_layout()

#%% Plot B field for a few layers
B_offset = B_offsets[sB_result.argmax()]
zs = np.arange(-1, 2) * a_vert # layer (-1, 0, 1) for calculating B_grad; target layer is layer 0
rad_axis = np.arange(-cloud_size, cloud_size, 5 * 1e-6)
all_B = np.zeros((len(zs), len(rad_axis)))
for i, z in enumerate(zs):
    Bs = BFieldAtZSlice(z, rad_axis, FBcoil_coeff, I_FB)
    Bs[2, :] += B_offset
    all_B[i, :] = np.linalg.norm(Bs ,axis = 0)
B_rfmin = all_B[1, :].min() * 1.1 + all_B[1, :].max() * (-0.1)
B_rfmax = all_B[1, :].min() * (-0.1) + all_B[1, :].max() * 1.1

fig_B = plt.figure(2)
fig_B.clf()
ax_B = fig_B.add_subplot(111)
for i, coor in enumerate(zs):
    ax_B.plot(rad_axis / 1e-6, all_B[i, :] * 1e4, label = str(coor))
ax_B.fill_between(rad_axis / 1e-6, B_rfmin * 1e4, B_rfmax * 1e4, color = [0, 0, 0], alpha = 0.3)
ax_B.set_xlabel('radial position [um]')
ax_B.set_ylabel('B [G]')
ax_B.set_title('I = {}, '.format(I_FB) + '$B_{offset}$' + ' = {} G'.format(B_offset * 1e4))
ax_B.legend()
fig_B.tight_layout()

#%%
I_all = np.arange(40, 120, 2)
sB_all, Bo_all = sBandBo_Bgrads(a_vert, I_all, FBcoil_coeff, cloud_size)

fig_I = plt.figure(3)
fig_I.clf()
ax_IsB = fig_I.add_subplot(121)
ax_IsB.plot(I_all, sB_all)
ax_IsB.set_xlabel('I [A]')
ax_IsB.set_ylabel('sB')

ax_IBo = fig_I.add_subplot(122)
ax_IBo.plot(I_all, Bo_all * 1e4)
ax_IBo.set_xlabel('I [A]')
ax_IBo.set_ylabel('$B_{offset}$ [G]')
fig_I.tight_layout()