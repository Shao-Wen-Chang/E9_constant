import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pathlib import Path

import sys
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
from E9_fn.tight_binding import E9tb

logpath = '' # '' if not logging to a file
loglevel = logging.INFO
logroot = logging.getLogger()
list(map(logroot.removeHandler, logroot.handlers))
list(map(logroot.removeFilter, logroot.filters))
logging.basicConfig(filename = logpath, level = loglevel)

save_folder = "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations\\projects\\flat_band_cooling\\eigvals_library"
save_eigvals = False

#%% Define the model and solve it
lattice_str = "kagome"
lattice_halfdim = 6
lattice_dim = (lattice_halfdim * 2, lattice_halfdim * 2)
tb_params = E9tb.model_dictionary[lattice_str]
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
H_bare = my_tb_model.H

# Add offset to the bare model
sys_halfdim = 0
sys_dim = (sys_halfdim * 2, sys_halfdim * 2)
sys_range = (lattice_halfdim - sys_halfdim, lattice_halfdim + sys_halfdim)
V_rsv_offset = -0
# Find what unit cells are in the reservoir by excluding the unit cells in the system
sys_natural_uc_ind = set([(ii, jj) for jj in range(my_tb_model.lat_dim[1]) if sys_range[0] <= jj and jj < sys_range[1]
                                    for ii in range(my_tb_model.lat_dim[0]) if sys_range[0] <= ii and ii < sys_range[1]])
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

H_total = H_bare + H_offset
eigvals, eigvecs = eigh(H_total)

#%% Plots
plot_state_list = []

# fig_H, ax_H = util.make_simple_axes(fignum = 100)
# ax_H.matshow(H_total)

fig_E, ax_E = util.make_simple_axes()
ax_E.scatter(np.arange(len(eigvals)), eigvals)
ax_E.set_title("{} ({} unit cells), all states".format(lattice_str, lattice_dim))
ax_E.scatter(plot_state_list, eigvals[plot_state_list], color = "red", label = "selected states")
ax_E.legend()

for st in plot_state_list:
    fig_lat, ax_lat = util.make_simple_axes(fig_kwarg = {"figsize": (12, 6)})
    my_tb_model.plot_H(ax = ax_lat, H = H_total)
    my_tb_model.plot_state(eigvecs[:, st], ax_lat)
    ax_lat.set_title("state {}, E = {:.4f}".format(st, eigvals[st]))
if plot_state_list == []:
    fig_lat, ax_lat = util.make_simple_axes(fig_kwarg = {"figsize": (12, 6)})
    my_tb_model.plot_H(ax = ax_lat, H = H_total)

#%% Save eigenvalues
if save_eigvals:
    if V_rsv_offset == 0:
        str_offset_config = "no_offset"
    else:
        str_offset_config = "sys{}x{}_Vrsv{}".format(sys_dim[0], sys_dim[1], V_rsv_offset)
    file_name = "{}_lat{}x{}_{}.npy".format(lattice_str, lattice_dim[0], lattice_dim[1], str_offset_config)
    full_path = Path(save_folder, file_name)
    if full_path.exists():
        print("file already exists; not doing anything for now")
    else:
        np.save(full_path, eigvals)