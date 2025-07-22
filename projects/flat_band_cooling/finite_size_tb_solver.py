import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pathlib import Path
from secrets import randbits

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
bool_save_results = False

rng_seed = randbits(128)
rng1 = np.random.default_rng(rng_seed)

#%% Define the model and solve it
lattice_str = "kagome"
lattice_len = 10
tnnn = -0.02
lattice_dim = (lattice_len, lattice_len)    # 2D lattices
# lattice_dim = (lattice_len, 1)              # 1D lattices
overwrite_param = {}
# overwrite_param = {"sublat_offsets": [0., 0., 0., 15.]}
# overwrite_param = {"tnnn": tnnn, "lat_bc": (1, 1)}
tb_params = E9tb.get_model_params(lattice_str, overwrite_param = overwrite_param)
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
H_bare = my_tb_model.H

# Add offset to the bare model
sys_len = 6
sys_range = ((lattice_len - sys_len) // 2, (lattice_len + sys_len) // 2)
n_sys = sys_len**2
V_rsv_offset = -2
V_std_random = 0.2
# Find what unit cells are in the reservoir by excluding the unit cells in the system
# 2D lattices:
sys_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1]) if sys_range[0] <= jj and jj < sys_range[1]
                                    for ii in range(my_tb_model.lat_dim[0]) if sys_range[0] <= ii and ii < sys_range[1]])
# 1D lattices:
# sys_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1])
#                                     for ii in range(my_tb_model.lat_dim[0]) if sys_range[0] <= ii and ii < sys_range[1]])
rsv_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1])
                                    for ii in range(my_tb_model.lat_dim[0])])
rsv_natural_uc_ind -= sys_natural_uc_ind
rsv_natural_uc_ind = np.array(list(rsv_natural_uc_ind))
logging.debug(rsv_natural_uc_ind)
rsv_ind = np.hstack(
    [my_tb_model.get_reduced_index(rsv_natural_uc_ind[:,0], rsv_natural_uc_ind[:,1], k)
        for k in range(my_tb_model.n_basis)])

H_offset = np.zeros_like(H_bare)
H_offset[rsv_ind, rsv_ind] = V_rsv_offset
H_offset += V_std_random * np.diag(rng1.standard_normal(my_tb_model.n_orbs))

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
plot_state_list = [22, 33]

# fig_H, ax_H = util.make_simple_axes(fignum = 100)
# ax_H.matshow(H_total)

fig_E = plt.figure(figsize = (8, 8))
fig_E.suptitle("{} (total {}, system {}, reservoir offset = {}, V_std_random = {})".format(
                lattice_str, lattice_dim, (sys_len, sys_len), V_rsv_offset, V_std_random))
ax_E = fig_E.add_subplot(221)
ax_DoS = fig_E.add_subplot(222)
ax_nu = fig_E.add_subplot(223)
ax_E.scatter(np.arange(len(eigvals)), eigvals)
ax_E.set_title("Energy of all states")
ax_E.scatter(plot_state_list, eigvals[plot_state_list], color = "red", label = "selected states")
ax_E.legend()
E_bins = np.linspace(eigvals[0], eigvals[-1], my_tb_model.n_orbs // 10)
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

#%% Save results
def get_model_str():
    """Return a string that describes the model."""
    other_params_str = ""
    if lattice_str == "kagome_withD":
        other_params_str += f"_tnnn{tnnn:.4f}"
    return (f"{lattice_str}_lat{lattice_dim[0]}x{lattice_dim[1]}"
            f"_sys{sys_len}x{sys_len}_Vrsv{V_rsv_offset}{other_params_str}").replace(".", "p")

def save_arr_data(file_path, arr_str_list):
    """Save specified arrays using np.savez."""
    arr_dict = {arr_str: eval(arr_str) for arr_str in arr_str_list}
    np.savez(file_path, **arr_dict)

if bool_save_results:
    folder_name = get_model_str()
    save_folder_path = Path(save_folder, folder_name)
    if save_folder_path.exists():
        logging.info(f"Folder {folder_name} already exists; overwriting")
    else:
        save_arr_data(save_folder_path, ["eigvals", "eigvecs", "n_sys", "density_sys"])
        
        logging.info(f"Files saved to {folder_name}")