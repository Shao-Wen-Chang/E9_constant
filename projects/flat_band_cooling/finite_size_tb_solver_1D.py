import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pathlib import Path

import sys
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
sys.path.insert(1, str(E9path))
from E9_fn import util
from E9_fn.tight_binding import E9tb

logpath = '' # '' if not logging to a file
loglevel = logging.INFO
logroot = logging.getLogger()
list(map(logroot.removeHandler, logroot.handlers))
list(map(logroot.removeFilter, logroot.filters))
logging.basicConfig(filename = logpath, level = loglevel)

save_folder = Path(E9path, "projects", "flat_band_cooling", "eigvals_library")
save_data = False
file_name = ""  # This will overwrite the default file name

#%% Define the model and solve it
lattice_str = "sawtooth"
lattice_len = 10
lattice_dim = (lattice_len, 1)
tb_params = E9tb.get_model_params(lattice_str)#, overwrite_param = {"lat_bc": (1, 0)})
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
H_bare = my_tb_model.H

# Add offset to the bare model
sys_len = 5
sys_range = ((lattice_len - sys_len) // 2, 1)
n_sys = sys_len
V_rsv_offset = -2
# Find what unit cells are in the reservoir by excluding the unit cells in the system
sys_natural_uc_ind = np.array([(ii, 0) for ii in range(sys_len)])
rsv_natural_uc_ind = np.array([(my_tb_model.lat_dim[0] - 1 - ii, 0)
                               for ii in range(my_tb_model.lat_dim[0] - sys_len)])
rsv_ind = np.hstack(
    [my_tb_model.get_reduced_index(rsv_natural_uc_ind[:,0], rsv_natural_uc_ind[:,1], k)
        for k in range(my_tb_model.n_basis)])
H_offset = np.zeros_like(H_bare)
H_offset[rsv_ind, rsv_ind] = V_rsv_offset

H_total = H_bare + H_offset
eigvals, eigvecs = eigh(H_total)

#%% Post processing
# calculate the ratio of the density in the system region for each state
density_sys = np.zeros_like(eigvals)
sys_reduced_uc_ind = [my_tb_model.get_reduced_index(ii, jj, k) for (ii, jj) in sys_natural_uc_ind
                                                               for k in range(my_tb_model.n_basis)]
for i in range(len(density_sys)):
    eigvec = eigvecs[:, i]
    density_sys[i] = sum(abs(eigvec[sys_reduced_uc_ind]**2))

# When I care enough, calculate the ratio of the density on the edge for each state
pass

#%% Plots
plot_real_space = True
plot_state_list = []

# if np.all(np.isreal(H_total)):
#     fig_H, ax_H = util.make_simple_axes(fignum = 100)
#     ax_H.matshow(H_total.real)
# else:
#     print("H_total is not all real, not plotting H for now")

fig_E = plt.figure(figsize = (8, 8))
fig_E.suptitle("{} (total {}, system {}, reservoir offset = {})".format(
                lattice_str, lattice_dim, (sys_len, sys_len), V_rsv_offset))
ax_E = fig_E.add_subplot(221)
ax_DoS = fig_E.add_subplot(222)
ax_nu = fig_E.add_subplot(223)
ax_E.scatter(np.arange(len(eigvals)), eigvals)
ax_E.set_title("Energy of all states")
ax_E.scatter(plot_state_list, eigvals[plot_state_list], color = "red", label = "selected states")
ax_E.legend()
E_bins = np.linspace(eigvals[0], eigvals[-1], my_tb_model.n_orbs // 20)
ax_DoS.hist(eigvals, bins = E_bins, orientation = "horizontal")
ax_DoS.set_title("DoS")
ax_nu.plot(density_sys)
ax_nu.set_title(r"$\nu_{sys}$")
fig_E.tight_layout()

if plot_real_space:
    for st in plot_state_list:
        fig_lat, ax_lat = util.make_simple_axes(fig_kwarg = {"figsize": (12, 6)})
        my_tb_model.plot_H(ax = ax_lat, H = H_total)
        my_tb_model.plot_state(eigvecs[:, st], ax_lat)
        ax_lat.set_title("state {}, E = {:.4f}".format(st, eigvals[st]))
    if plot_state_list == []:
        fig_lat, ax_lat = util.make_simple_axes(fig_kwarg = {"figsize": (12, 6)})
        my_tb_model.plot_H(ax = ax_lat, H = H_total)

#%% Save eigenvalues
if save_data:
    if V_rsv_offset == 0:
        str_offset_config = "no_offset"
    else:
        str_offset_config = "sys{}x{}_Vrsv{}".format(sys_len, sys_len, V_rsv_offset)
    if not file_name:
        file_name = "{}_lat{}x{}_{}".format(lattice_str, lattice_dim[0], lattice_dim[1], str_offset_config).replace(".", "p") + ".npz"
    
    full_path = Path(save_folder, file_name)
    if full_path.exists():
        logging.info("File {} already exists; not doing anything for now".format(file_name))
    else:
        np.savez(full_path,
                 eigvals = eigvals,
                 eigvecs = eigvecs,
                 n_sys = n_sys,
                 density_sys = density_sys,)
        logging.info("File {} saved".format(file_name))