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
import E9_fn.thermodynamics as thmdy
from E9_fn import util
from E9_fn.tight_binding import E9tb
from projects.flat_band_cooling import helper_fns as hpfn

logpath = '' # '' if not logging to a file
loglevel = logging.INFO
logroot = logging.getLogger()
list(map(logroot.removeHandler, logroot.handlers))
list(map(logroot.removeFilter, logroot.filters))
logging.basicConfig(filename = logpath, level = loglevel)

rng_seed = randbits(128)
rng1 = np.random.default_rng(rng_seed)

#%% Define the model
lattice_str = "sawtooth"
parent_folder_name = lattice_str
lattice_len = 10
tnnn = -0.02
# lattice_dim = (lattice_len, lattice_len)    # 2D lattices
lattice_dim = (lattice_len, 1)              # 1D lattices
overwrite_param = {}
# overwrite_param = {"sublat_offsets": [0., 0., 0., 15.]}
# overwrite_param = {"tnnn": tnnn, "lat_bc": (1, 1)}
tb_params = E9tb.get_model_params(lattice_str, overwrite_param = overwrite_param)
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
H_bare = my_tb_model.H

# Add offset to the bare model
sys_len = 0
sys_range = ((lattice_len - sys_len) // 2, (lattice_len + sys_len) // 2)
n_sys = sys_len**2
V_rsv_offset = 0.
l_res = 0.   # Resolution of the box potential (in units of lattice cell size)
V_std_random = 0.
# Find what unit cells are in the reservoir by excluding the unit cells in the system
# 2D lattices:
# sys_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1]) if sys_range[0] <= jj and jj < sys_range[1]
#                                     for ii in range(my_tb_model.lat_dim[0]) if sys_range[0] <= ii and ii < sys_range[1]])
# 1D lattices:
sys_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1])
                                    for ii in range(my_tb_model.lat_dim[0]) if sys_range[0] <= ii and ii < sys_range[1]])
rsv_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1])
                                    for ii in range(my_tb_model.lat_dim[0])])
rsv_natural_uc_ind -= sys_natural_uc_ind
rsv_natural_uc_ind = np.array(list(rsv_natural_uc_ind))
logging.debug(rsv_natural_uc_ind)
rsv_ind = np.hstack(
    [my_tb_model.get_reduced_index(rsv_natural_uc_ind[:,0], rsv_natural_uc_ind[:,1], k)
        for k in range(my_tb_model.n_basis)])

#%% save related configs
data_folder = Path(E9path, "projects", "flat_band_cooling", "eigvals_library")
bool_save_results = False
bool_overwrite = False      # always overwrite existing results if True, check existing results if False
save_new_upto = 1           # if overwrite = False and there is a existing folder, save another result up to the n-th folder
# arr_str_list_to_save = ["eigvals", "eigvecs", "density_sys"]
arr_str_list_to_save = ["eigvals", "density_sys"]

param_dict = dict()
if V_std_random != 0:
    param_dict["Vran"] = V_std_random
if tnnn != 0 and lattice_str in {"kagome_nnn", "bilayer_kagome"}:
    param_dict["tnnn"] = tnnn
if l_res != 0:
    param_dict["lres"] = l_res

H_offset = np.zeros_like(H_bare)
if l_res == 0:
    H_offset[rsv_ind, rsv_ind] = V_rsv_offset
else:
    H_offset = V_rsv_offset * hpfn.get_finite_res_box(lattice_dim, sys_range, my_tb_model, l_res)
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

# von Neumann entropy in the system
S_sys = np.zeros_like(eigvals)
for i in range(my_tb_model.n_orbs):
    eigvec = eigvecs[:, i]
    rho = np.outer(eigvec.conj().T, eigvec)
    rho_sys = util.get_red_den_mat(rho, sys_reduced_uc_ind)
    S_sys[i] = thmdy.find_SvN(rho_sys)

# When I care enough, calculate the ratio of the density on the edge for each state
pass

#%% Plots
plot_real_space = True
plot_state_list = [25, 35]

# fig_H, ax_H = util.make_simple_axes(fignum = 100)
# ax_H.matshow(H_total)

fig_E = plt.figure(figsize = (8, 8))
fig_E.suptitle("{} (total {}, system {}, reservoir offset = {}, V_std_random = {})".format(
                lattice_str, lattice_dim, (sys_len, sys_len), V_rsv_offset, V_std_random))
ax_E = fig_E.add_subplot(221)
ax_DoS = fig_E.add_subplot(222)
ax_nu = fig_E.add_subplot(223)
ax_S_sys = fig_E.add_subplot(224)
ax_E.scatter(np.arange(len(eigvals)), eigvals)
ax_E.set_title("Energy of all states")
ax_E.scatter(plot_state_list, eigvals[plot_state_list], color = "red", label = "selected states")
ax_E.legend()
E_bins = np.linspace(eigvals[0], eigvals[-1], my_tb_model.n_orbs // 10)
ax_DoS.hist(eigvals, bins = E_bins, orientation = "horizontal")
ax_DoS.set_title("DoS")
ax_nu.plot(density_sys)
ax_nu.set_title(r"$\nu_{sys}$")
ax_S_sys.plot(S_sys, label = r"$S^{vN}_{sys}$")
ax_S_sys.plot(density_sys * (1 - density_sys), label = r"$\nu_{sys} (1 - \nu_{sys})$")
ax_S_sys.set_title("Entropy wannabes")
ax_S_sys.legend()
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
if not bool_save_results: logging.info("Results are not saved")
if bool_save_results:
    if bool_overwrite:
        raise(Exception("I haven't coded this yet")) # account for run number
    bool_all_in_npz = True
    for cnt in range(1, save_new_upto + 1):
        folder_name = hpfn.get_model_str(lattice_str, lattice_dim, sys_len, V_rsv_offset
                                                , runnum = cnt, param_dict = param_dict)
        save_folder_path = Path(data_folder, parent_folder_name, folder_name)
        bool_all_in_npz, flag_str = hpfn.check_npz_contents(save_folder_path, arr_str_list_to_save)
        if bool_all_in_npz:     # Folder exists and has all arrays, check the next cnt
            pass
        elif flag_str == "":    # Folder does not exist for cnt
            logging.info(f"Saving to {folder_name}")
            pass
        else:                   # Folder exist for cnt, but does not have all arrays
            logging.info(f"Array {flag_str} not found in {folder_name}; will overwrite")
            pass
    if bool_all_in_npz:         # Folders for all cnt exist and have all arrays, skip this offset
        logging.info(f"Folder {folder_name} already exists up to {cnt} copies; skip this one")
        pass

    # Save results
    dict_save = my_tb_model.to_dict()
    if V_std_random != 0:
        dict_save["rng_seed"] = rng_seed
    util.save_arr_data(Path(save_folder_path, "np_arrays"), arr_str_list_to_save, [eval(a) for a in arr_str_list_to_save])
    util.save_dict(Path(save_folder_path, "tb_model_params"), dict_save)
    logging.debug(f"Files saved to {save_folder_path}")