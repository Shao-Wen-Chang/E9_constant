import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pathlib import Path

import sys
E9path = Path("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if str(E9path) not in sys.path:
    sys.path.insert(1, str(E9path))
from E9_fn import util
from E9_fn.tight_binding import E9tb
from projects.flat_band_cooling import helper_fns as hpfn

logpath = '' # '' if not logging to a file
loglevel = logging.INFO
logroot = logging.getLogger()
list(map(logroot.removeHandler, logroot.handlers))
list(map(logroot.removeFilter, logroot.filters))
logging.basicConfig(filename = logpath, level = loglevel)

data_folder = Path(E9path, "projects", "flat_band_cooling", "eigvals_library")
bool_save_results = True
bool_overwrite = False          # overwrite existing results if True, skip if False

#%% Define the model and solve it
lattice_str = "kagome"
parent_folder_name = lattice_str
lattice_len = 20
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
sys_len = 12
sys_range = ((lattice_len - sys_len) // 2, (lattice_len + sys_len) // 2)
n_sys = sys_len**2
V_rsv_offsets = np.linspace(0.1, 2, 191)

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

H_offset_ones = np.zeros_like(H_bare)
H_offset_ones[rsv_ind, rsv_ind] = 1.

#%% Solve for each reservoir offset
if not bool_save_results: logging.info("Results are not saved")
for V_rsv_offset in V_rsv_offsets:
    if bool_save_results:
        folder_name = hpfn.get_model_str(lattice_str, lattice_dim, sys_len, V_rsv_offset)
        save_folder_path = Path(data_folder, parent_folder_name, folder_name)
        arr_str_list_to_save = ["eigvals", "density_sys"]
        if save_folder_path.exists():
            if bool_overwrite:
                logging.info(f"Folder {folder_name} already exists; overwriting")
            else:
                logging.info(f"Folder {folder_name} already exists; skip this one")
                continue
        else:
            logging.info(f"working on V = {V_rsv_offset:.4f} ...")

    H_total = H_bare + H_offset_ones * V_rsv_offset
    eigvals, eigvecs = eigh(H_total)

    # calculate the ratio of the density in the system region for each state
    density_sys = np.zeros_like(eigvals)
    sys_reduced_uc_ind = [my_tb_model.get_reduced_index(ii, jj, k) for (ii, jj) in sys_natural_uc_ind
                                                                for k in range(my_tb_model.n_basis)]
    for i in range(len(density_sys)):
        eigvec = eigvecs[:, i]
        density_sys[i] = sum(abs(eigvec[sys_reduced_uc_ind]**2))

    # Save results
        util.save_arr_data(Path(save_folder_path, "np_arrays"), arr_str_list_to_save, [eval(a) for a in arr_str_list_to_save])
        util.save_dict(Path(save_folder_path, "tb_model_params"), my_tb_model.to_dict())
        logging.debug(f"Files saved to {folder_name}")