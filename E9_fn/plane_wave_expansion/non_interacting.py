import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import sys

sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn.plane_wave_expansion import blochstate_class as bsc
import E9_fn.E9_constants as E9c
from E9_fn import util

species = "Rb87" # "Rb87", "K40"
#%% Unit
if species == "Rb87":
    all_units_dict = E9c.all_lat_unit_Rb87
elif species == "K40":
    all_units_dict = E9c.all_lat_unit_K40
else:
    raise ValueError("Unknown species: {}".format(species))
m_unit = all_units_dict["m_unit"]
l_unit = all_units_dict["l_unit"]
E_unit = all_units_dict["E_unit"]
f_unit = all_units_dict["f_unit"]
t_unit = all_units_dict["t_unit"]

#%% Experiment parameters
V532nom = 25 # in kHz (i.e. V_SI / hbar / 1e3 / 2pi)
V1064nom = 15
n0nom = 0               # peak density
phi12, phi23 = 0., 0.   # The superlattice phase that determines the relative position between 1064 and 532 lattice
ABoffset1064nom = 0
B1_rel_int_1064 = 1     # relative intensity (field is sqrt of that) of 1064 B1
B1_rel_int_532  = 0.64     # relative intensity (field is sqrt of that) of 532 B1
B3_rel_int_1064 = 1     # relative intensity (field is sqrt of that) of 1064 B3
B3_rel_int_532  = 1     # relative intensity (field is sqrt of that) of 532 B3
# ABoffset1064nom = 0.011585 * V1064nom / 9 / np.sqrt(3)
# Simulation related constants (e.g. # of points used in calculation) are scattered around

#%% Initialization
V532 = 2 * np.pi * V532nom * 1e3 / f_unit   # 2 * np.pi because I have f = E/hbar instead of E/h as normally defined
V1064 = 2 * np.pi * V1064nom * 1e3 / f_unit
ABoffset1064 = 2 * np.pi * ABoffset1064nom * 1e3 / f_unit
n0 = n0nom * l_unit**3
Exp_lib = {"species": species, "units_dict": all_units_dict
        , 'V532nom': V532nom, 'V1064nom': V1064nom, 'V532': V532, 'V1064': V1064
        , 'B1_rel_int_532': B1_rel_int_532, 'B1_rel_int_1064': B1_rel_int_1064 , 'B3_rel_int_532': B3_rel_int_532, 'B3_rel_int_1064': B3_rel_int_1064
        , 'n0nom': n0nom, 'n0': n0
        , 'ABoffset1064nom': ABoffset1064nom, 'ABoffset1064': ABoffset1064
        , 'phi12': phi12, 'phi23': phi23}

#%% Functions
def MinimumGap(e_values, band1, band2):
    """Prints the position and energy of the minimum gap.
    
    Example: MinimumGap(e_values,2,3)
    Should be slightly enhanced to return something
    """
    gaps = e_values[:, band2] - e_values[:, band1]
    print('The minimum band gap between band{} and band{} is {} kHz, happening at the {} (python index) q evaluated' \
          .format(band1, band2, gaps.min() * f_unit / 1e3 / (2 * np.pi), gaps.argmin()))
# deleted a bunch of things from the original code

#%% Finding Bloch states and band energies
start_time = time.time()

# Basic simulation parameters
num = 5 # size of q-momentum space we consider: (-num, num) (usually 5)
size = 2 * num + 1
k_center = (0, 0)
bandstart = 0 # starting from 0, inclusive
bandend = 2 # inclusive
bandnum = bandend - bandstart + 1 # number of bands interested in
Qp_str = '(Kp/K + 0.4 * np.array([np.cos(pi/3), np.sin(pi/3)]))'
qverts_str = 'E9c.Kp4/E9c.k_lw, E9c.Gp/E9c.k_lw, E9c.Kp/E9c.k_lw, E9c.Mp/E9c.k_lw, E9c.Gp/E9c.k_lw' #'(Kp/K + 0.477 * np.array([np.cos(pi/3), np.sin(pi/3)])), (Kp/K + 0.677 * np.array([np.cos(pi/3), np.sin(pi/3)]))' #1.5*Kp/K  #
x_ticklist = ["K'", '$\Gamma$', 'K', 'M', '$\Gamma$']
qverts_arr = eval(qverts_str)
qverts_type = 1 # "What qset defines": 1 - line; 2 - area (see PlotBZSubplot)
save_results = False

# Generate qset
if qverts_type == 1:
    num_points = np.array([100, 100, 60, 40]) # number of points sampled between two points (can be an array specifying each path, or just one number for all)
    index_points = np.hstack((np.array([0]), np.cumsum(num_points))) - np.arange(len(num_points) + 1)
    qsets = bsc.FindqSets(num_points, qverts_arr)#bsc.FindqSets(points, Gammap/k, kp/k, mp/k, Gammap/k)
    PlotBZinput = qverts_str
elif qverts_type == 2:
    dq = 0.01
    dq2BZfrac = dq**2 * (2/(bandend + 1)) / (np.sqrt(3)/8) # (ONLY CORRECT FOR Kp > Gp > Mp > Kp) how much of a BZ does a point roughly correspond to
    qsets = bsc.FindqArea(qverts_arr, dqx = dq, dqy = dq)
    PlotBZinput = (qverts_str, qsets)

e_values = np.zeros((len(qsets), bandnum), dtype = np.cdouble)
e_states = np.zeros((len(qsets), size**2, bandnum), dtype = np.cdouble)
e_states_ni = [[] for _ in range(bandnum)]

# find non-interacting solution
print("Total number of points = {0}".format(len(qsets)))
ax_BZ = bsc.PlotBZ(qset = PlotBZinput)
xrun = np.arange(len(qsets))
Hq_mmat, Hq_nmat, H_lat = bsc.find_H_components(num, Exp_lib, center = k_center)
for i in range(len(qsets)):
    # H = bsc.FindH(qsets[i], num, Exp_lib, center = k_center)
    # H_old = bsc.FindH(qsets[i], num, Exp_lib, center = k_center)
    H = bsc.find_H(qsets[i], Exp_lib, Hq_mmat, Hq_nmat, H_lat)
    # print(i, np.allclose(H, H_old))
    e_values[i,:], e_states[i,:,:] = bsc.FindEigenStuff(H, (bandstart, bandend), num = num)
    for j, bandN in enumerate(range(bandstart, bandend + 1)):
        e_states_ni[j].append(bsc.blochstate(e_states[i,:,j], q = qsets[i], center = k_center, N = bandN, E = e_values[i,j], param = Exp_lib))

print("--- {0} seconds ---".format((time.time() - start_time)))

# If the eigenvalues are not real, raise a warning
if not np.all(np.isreal(e_values)):
    print('Warning: complex eigenvalue detected. Imaginary parts are discarded.')
else:
    e_values = e_values.astype(np.double)

if save_results:
    path_str = "simulation_results\\" #IF NOT WORKING: CHANGE TO FULL PATH NAME // TRY ADDING FULL PATH TO "PYTHONPATH manager"
    suffix_str = "_honeycomb_NearG_20210826"
    bsc.SaveStateList(path_str + "e_states_ni_{0}_{1}_{2}_k{3}{3}_{4}{5}".format(V532nom, V1064nom, n0nom, num, k_center[0], k_center[1]) + suffix_str, e_states_ni)

#%% Plot band structure
f2kHz = f_unit / 1e3 / (2 * np.pi) # conversion factor from natural units to kHz
E_lowest = np.min(e_values)
E_kHz_offset = (e_values - E_lowest) * f2kHz
E_kHz_highest = np.max(E_kHz_offset)
fig_E = plt.figure(0, figsize=(10,7))
fig_E.clf()
fig_E.suptitle(qverts_str)

if qverts_type == 1:
    ax_E = fig_E.add_subplot(111)
    #xq = [np.linalg.norm(qpt) for qpt in qsets] # won't work for plots with winding q paths
    xq = bsc.FindQAxis(num_points, qverts_arr)
    ax_E.set_ylabel('E/h (kHz)', fontsize = 15)
    for i in range(bandnum):
        ax_E.plot(xq, (e_values.transpose()[i] - E_lowest) * f2kHz, '-', label = 'Non-interacting' + str(i + bandstart))
    plt.xticks(xq[index_points], x_ticklist, fontsize = 14) # why does ax.set_xticks not work?
    # plt.yticks([0, 10, 20, 30], ['0', '10', '20', '30'], fontsize = 14)
    # ax_E.set_ylim([0, 6])
    # ax_E.legend()
elif qverts_type == 2:
    ax_E = fig_E.add_subplot(111, projection = '3d')
    # Add BZ boundary
    bz1qx, bz1qy = [q[0] for q in E9c.BZ1_vertices], [q[1] for q in E9c.BZ1_vertices]
    for i in range(3):
        ax_E.plot(bz1qx, bz1qy, np.ones_like(bz1qx) * E_kHz_highest * i / 2, '-k', alpha = 0.5)
    for i in range(bandnum):
        ax_E.plot_trisurf(qsets[:, 0], qsets[:, 1], E_kHz_offset[:, i])
    ax_E.set_xlabel('q_x', fontsize = 15)
    ax_E.set_ylabel('q_y', fontsize = 15)
    ax_E.set_zlabel('E/h (kHz)', fontsize = 15)

# ax_E.set_title('V532 = {} kHz, V1064 = {} kHz; AB offset = {:.4} kHz\nn0 = {}; (-{},{})_{}; phi = ({:.4},{:.4})'.format(V532nom, V1064nom, float(ABoffset1064nom), n0nom, num, num, k_center, phi12, phi23))

#%% Plot DoS

if qverts_type == 2:
    E_DoS, DoS, DoS_int = bsc.FindDoS(e_values, cellsize = dq2BZfrac)
    fig_DoS = plt.figure(6, figsize=(11,4))
    fig_DoS.clf()
    ax_DoS = fig_DoS.add_subplot(121)
    ax_DoS.plot((E_DoS - E_lowest) * f2kHz, DoS)
    ax_DoS.set_xlabel('E/h (kHz)', fontsize = 15)
    ax_DoS.set_ylabel('DoS', fontsize = 15)
    ax_DoS_int = fig_DoS.add_subplot(122)
    ax_DoS_int.plot((E_DoS - E_lowest) * f2kHz, DoS_int)
    ax_DoS_int.set_xlabel('E/h (kHz)', fontsize = 15)
    ax_DoS_int.set_ylabel('DoS_int', fontsize = 15)

#%% Plot group velocity & other stuff
# fig_gv = plt.figure(1, figsize=(10,7))
# fig_gv.clf()
# plt.title('GV: V532 = {0} kHz, V1064 = {1} kHz; n0 = {2}; (-{3},{3})_{4}'.format(V532nom, V1064nom, n0nom, num, k_center))
# plt.xlabel('runnum') # plt.xlabel('q (/q_Kp)')
# plt.ylabel('v (/(hbar*k/m))')
# # This is assuming we are on the y axis
# for i, bandN in enumerate(range(bandstart, bandend + 1)):
#     gvs_ni = [np.sqrt(nistate.findvx()**2 + nistate.findvy()**2) for nistate in e_states_ni[i]]
#     plt.plot(xrun, gvs_ni, '-', label = 'Non-interacting{}'.format(bandN))
#     plt.legend()

# band_low = 0
# band_high = 1
# fig_gap = plt.figure(22, figsize = (10,7))
# fig_gap.clf()
# plt.title('gap between band {} and {}'.format(band_low, band_high))
# plt.xlabel('|q| (/q_Kp)')  #plt.xlabel('|q| (/q_Kp)')
# plt.ylabel('E/h (kHz)')
# plt.plot(xq, (e_values.transpose()[band_high] - e_values.transpose()[band_low]) * f_unit / 1e3 / (2 * np.pi), 'o-')

# band_pairs = [[2, 7], [2, 8], [2, 9], [2, 10]]
# fig_gap = plt.figure(22, figsize = (10,7))
# fig_gap.clf()
# ax_gap = fig_gap.add_subplot(111)
# ax_gap.set_title('gaps between selected band pairs @ V1064 = {}'.format(Exp_lib['V1064nom']))
# ax_gap.set_xlabel('xq')  #plt.xlabel('q (/q_Kp)')
# ax_gap.set_ylabel('E/h (kHz)')
# for band_low, band_high in band_pairs:
#     ax_gap.plot(xq, (e_values.transpose()[band_high] - e_values.transpose()[band_low]) * f_unit / 1e3 / (2 * np.pi), \
#               'o-', label = 'gap{}_{}'.format(band_low, band_high))
# freq_K = 9e3 # Don't remember
# ax_gap.plot(xq, 2 * xq * freq_K / 1e3, 'o-', label = '2 * lat_detune')
# plt.legend()