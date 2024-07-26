import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pathlib import Path

import sys
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
import equilibrium_finder as eqfind # For some reason removing this line gives an error (ModuleNotFoundError: No module named 'E9_fn')
sys.path.insert(1, E9path)
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
lattice_str = "kagome"
lattice_len = 10
tnnn = -0.02
lattice_dim = (lattice_len, lattice_len)
tb_params = E9tb.get_model_params(lattice_str, tnnn = tnnn)
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
H_bare = my_tb_model.H

# Add a quadratic confinement to the bare model
V_quad_center = ((lattice_len - 1) / 2.) * (my_tb_model.lat_vec[0] + my_tb_model.lat_vec[1])
A_quad = 0.03  # The on-site offset is given by A_quad * norm(pos - V_quad_center)**2 
H_offset = np.zeros_like(H_bare)
for ri in np.arange(my_tb_model.n_orbs):
    H_offset[ri, ri] = A_quad * np.linalg.norm(my_tb_model.get_lat_pos(ri) - V_quad_center)**2

H_total = H_bare + H_offset
eigvals, eigvecs = eigh(H_total)

#%% Post processing
### calculate the ratio of the density in the system region for each state
# Define "system" (an arbitrary region centered at the potential minimum)
sys_radii = [2, 3]
sys_colors = ["red", "orange", "yellow"]
sys_natural_uc_ind = [[] for _ in sys_radii]
for ii in range(my_tb_model.lat_dim[0]):
    for jj in range(my_tb_model.lat_dim[1]):
        for kk in range(my_tb_model.n_basis):
            for l, r in enumerate(sys_radii):
                if np.linalg.norm(my_tb_model.get_lat_pos((ii, jj, kk)) - V_quad_center) <= r:
                    sys_natural_uc_ind[l].append((ii, jj, kk))
n_sys = [len(ls) for ls in sys_natural_uc_ind]
density_sys = [np.zeros_like(eigvals) for _ in sys_radii]
sys_reduced_uc_ind = [[my_tb_model.get_reduced_index(ii, jj, kk)
                       for (ii, jj, kk) in ls] for ls in sys_natural_uc_ind]
for i in range(len(eigvals)):
    eigvec = eigvecs[:, i]
    for l in range(len(sys_radii)):
        density_sys[l][i] = sum(abs(eigvec[sys_reduced_uc_ind[l]]**2))

#%% Plots
plot_real_space = True
plot_state_list = []

# fig_H, ax_H = util.make_simple_axes(fignum = 100)
# ax_H.matshow(H_total)

fig_E = plt.figure(figsize = (8, 8))
fig_E.suptitle("{} (total {}, A_quad = {})".format(
    lattice_str, lattice_dim, A_quad))
ax_E = fig_E.add_subplot(221)
ax_DoS = fig_E.add_subplot(222)
ax_nsys = fig_E.add_subplot(223)
ax_nusys = fig_E.add_subplot(224)
ax_E.scatter(np.arange(len(eigvals)), eigvals)
ax_E.set_title("Energy of all states")
ax_E.scatter(plot_state_list, eigvals[plot_state_list], color = "red", label = "selected states")
ax_E.legend()
E_bins = np.linspace(eigvals[0], eigvals[-1], my_tb_model.n_orbs // 20)
ax_DoS.hist(eigvals, bins = E_bins, orientation = "horizontal")
ax_DoS.set_title("DoS")
for l in range(len(sys_radii)):
    ax_nsys.plot(density_sys[l], color = sys_colors[l], label = "system{}".format(l))
    ax_nusys.plot(density_sys[l] / n_sys[l], color = sys_colors[l], label = "system{}".format(l))
ax_nsys.set_title(r"$n_{sys}$")
ax_nsys.legend()
ax_nusys.set_title(r"$\nu_{sys}$")
ax_nusys.legend()
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
    thetas = np.linspace(0, np.pi * 2)
    for l, r in enumerate(sys_radii):
        x_sys = V_quad_center[0] + r * np.cos(thetas)
        y_sys = V_quad_center[1] + r * np.sin(thetas)
        ax_lat.plot(x_sys, y_sys, color = sys_colors[l], linestyle = "--", label = "system{}".format(l))
        ax_lat.legend()

#%% Save eigenvalues
if save_data:
    str_offset_config = "A_quad{}".format(A_quad)
    if not file_name:
        if lattice_str == "kagome_nnn":
            lattice_str = lattice_str + str(tnnn)
        file_name = "{}_lat{}x{}_{}".format(lattice_str, lattice_dim[0], lattice_dim[1], str_offset_config).replace(".", "p") + ".npz"
    
    full_path = Path(save_folder, file_name)
    if full_path.exists():
        logging.info("File {} already exists; not doing anything for now".format(file_name))
    else:
        np.savez(full_path,
                 eigvals = eigvals,
                 eigvecs = eigvecs,
                 A_quad = A_quad,)
        logging.info("File {} saved".format(file_name))