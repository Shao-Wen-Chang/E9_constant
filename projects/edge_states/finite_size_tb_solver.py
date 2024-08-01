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

save_folder = Path(E9path, "projects", "edge_states", "eigvals_library")
save_data = False
file_name = ""  # This will overwrite the default file name

#%% Define the model and solve it
lattice_str = "kagome_str_spk"
lattice_len = 8
lattice_dim = (2, lattice_len)
# lattice_dim = (lattice_len, 2)
tnnn = 0.
n_ky = 51

# _temp_params = E9tb.get_model_params(lattice_str, overwrite_param = {"lat_bc": (0, 1)})
# _temp_model = E9tb.tbmodel_2D(lat_dim = lattice_dim, **_temp_params)
# n_orbs = _temp_model.n_orbs
# ky_list = np.linspace(-np.pi, np.pi, n_ky)
# all_models = [None for _ in ky_list]
# all_Hs = np.zeros((n_ky, n_orbs, n_orbs), dtype = complex)
# all_eigvals = np.zeros((n_ky, n_orbs))
# all_eigvecs = np.zeros((n_ky, n_orbs, n_orbs))

# for i, ky in enumerate(ky_list):
#     tb_params = E9tb.get_model_params(lattice_str, ky = ky, overwrite_param = {"lat_bc": (1, 1)})
#     my_tb_model = E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
#     H_bare = my_tb_model.H
#     eigvals, eigvecs = eigh(H_bare)

#     all_models[i] = my_tb_model
#     all_Hs[i, :, :] = H_bare
#     all_eigvals[i, :] = eigvals
#     all_eigvecs[i, :, :] = eigvecs

tb_params = E9tb.get_model_params(lattice_str, ky = 0, overwrite_param = {"lat_bc": (1, 1)})
my_tb_model = E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
H_bare = my_tb_model.H
eigvals, eigvecs = eigh(H_bare)
#%% Post processing
# When I care enough, calculate the ratio of the density on the edge for each state
pass

#%% Plots
# fig_band, ax_band = util.make_simple_axes(fignum = 200)
# for band_ind in range(n_orbs):
#     ax_band.plot(ky_list, all_eigvals[:, band_ind], color = str(0.5 - 0.5 * band_ind / n_orbs))
plot_real_space = True
plot_state_list = []

# fig_H, ax_H = util.make_simple_axes(fignum = 100)
# ax_H.matshow(H_total)

fig_E = plt.figure(figsize = (8, 8))
fig_E.suptitle("{} (total {})".format(lattice_str, lattice_dim))
ax_E = fig_E.add_subplot(121)
ax_DoS = fig_E.add_subplot(122)
ax_E.scatter(np.arange(len(eigvals)), eigvals)
ax_E.set_title("Energy of all states")
ax_E.scatter(plot_state_list, eigvals[plot_state_list], color = "red", label = "selected states")
ax_E.legend()
E_bins = np.linspace(eigvals[0], eigvals[-1], my_tb_model.n_orbs // 20)
ax_DoS.hist(eigvals, bins = E_bins, orientation = "horizontal")
ax_DoS.set_title("DoS")
fig_E.tight_layout()

if plot_real_space:
    for st in plot_state_list:
        fig_lat, ax_lat = util.make_simple_axes(fig_kwarg = {"figsize": (8, 5)})
        my_tb_model.plot_H(ax = ax_lat, H = H_bare)
        my_tb_model.plot_state(eigvecs[:, st], ax_lat)
        ax_lat.set_title("state {}, E = {:.4f}".format(st, eigvals[st]))
    if plot_state_list == []:
        fig_lat, ax_lat = util.make_simple_axes(fig_kwarg = {"figsize": (8, 5)})
        my_tb_model.plot_H(ax = ax_lat, H = H_bare)

#%% Save eigenvalues
# if save_data:
#     if not file_name:
#         if lattice_str == "kagome_nnn":
#             lattice_str = lattice_str + str(tnnn)
#         file_name = "{}_lat{}x{}_{}".format(lattice_str, lattice_dim[0], lattice_dim[1]).replace(".", "p") + ".npz"
    
#     full_path = Path(save_folder, file_name)
#     if full_path.exists():
#         logging.info("File {} already exists; not doing anything for now".format(file_name))
#     else:
#         np.savez(full_path,
#                  eigvals = eigvals,
#                  eigvecs = eigvecs,)
#         logging.info("File {} saved".format(file_name))