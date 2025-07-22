import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pathlib import Path
from secrets import randbits

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

#%% Define the model and solve it
lattice_str = "kagome_nnn"
parent_folder_name = lattice_str
lattice_len = 20
tnnn = -0.01
lattice_dim = (lattice_len, lattice_len)    # 2D lattices
# lattice_dim = (lattice_len, 1)              # 1D lattices
overwrite_param = {}
# overwrite_param = {"sublat_offsets": [0., 0., 0., 15.]}
# overwrite_param = {"tnnn": tnnn, "lat_bc": (1, 1)}
tb_params = E9tb.get_model_params(lattice_str, tnnn = tnnn, overwrite_param = overwrite_param)
my_tb_model= E9tb.tbmodel_2D(lat_dim = lattice_dim, **tb_params)
H_bare = my_tb_model.H

# Add (fixed) offset to the bare model
sys_len = 12
sys_range = ((lattice_len - sys_len) // 2, (lattice_len + sys_len) // 2)
n_sys = sys_len**2
V_rsv_offsets = np.linspace(-3.5, 2, 56)
V_std_random = 0

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

#%% save related configs
data_folder = Path(E9path, "projects", "flat_band_cooling", "eigvals_library")
bool_save_results = True
bool_overwrite = False          # overwrite existing results if True, skip if False
save_new_upto = 1           # if overwrite = False and there is a existing folder, save another result up to the n-th folder
arr_str_list_to_save = ["eigvals", "density_sys"]

param_dict = dict()
if V_std_random != 0:
    param_dict["Vran"] = V_std_random
if tnnn != 0:
    param_dict["tnnn"] = tnnn

#%% Solve for each reservoir offset
if not bool_save_results: logging.info("Results are not saved")
for V_rsv_offset in V_rsv_offsets:
    rng_seed = randbits(128)                # Use a different seed for each offset
    rng1 = np.random.default_rng(rng_seed)
    
    if bool_save_results:
        folder_name = hpfn.get_model_str(lattice_str, lattice_dim, sys_len, V_rsv_offset, param_dict = param_dict)
        save_folder_path = Path(data_folder, parent_folder_name, folder_name)
        if save_folder_path.exists():
            if bool_overwrite:
                logging.info(f"Folder {folder_name} already exists; overwriting")
            else:
                if save_new_upto > 1:
                    cnt = 2
                    # new_folder_name = f"{folder_name}_{cnt:03d}"
                    new_folder_name = hpfn.get_model_str(lattice_str, lattice_dim, sys_len, V_rsv_offset
                                                         , runnum = cnt, param_dict = param_dict)
                    new_save_folder_path = Path(data_folder, parent_folder_name, new_folder_name)
                    while new_save_folder_path.exists() and cnt < save_new_upto:
                        cnt += 1
                        # new_folder_name = f"{folder_name}_{cnt:03d}"
                        new_folder_name = hpfn.get_model_str(lattice_str, lattice_dim, sys_len, V_rsv_offset
                                                             , runnum = cnt, param_dict = param_dict)
                        new_save_folder_path = Path(data_folder, parent_folder_name, new_folder_name)
                    if new_save_folder_path.exists():
                        logging.info(f"Folder {folder_name} already exists up to {cnt} copies; skip this one")
                        continue
                    else:
                        logging.info(f"Folder {folder_name} already exists; append _{cnt:03d}")
                        save_folder_path = new_save_folder_path
                else:
                    logging.info(f"Folder {folder_name} already exists; skip this one")
                    continue
        else:
            logging.info(f"working on V = {V_rsv_offset:.4f} ...")

    H_total = H_bare + H_offset_ones * V_rsv_offset + V_std_random * np.diag(rng1.standard_normal(my_tb_model.n_orbs))
    eigvals, eigvecs = eigh(H_total)

    # calculate the ratio of the density in the system region for each state
    density_sys = np.zeros_like(eigvals)
    sys_reduced_uc_ind = [my_tb_model.get_reduced_index(ii, jj, k) for (ii, jj) in sys_natural_uc_ind
                                                                for k in range(my_tb_model.n_basis)]
    for i in range(len(density_sys)):
        eigvec = eigvecs[:, i]
        density_sys[i] = sum(abs(eigvec[sys_reduced_uc_ind]**2))

    # Save results
        dict_save = my_tb_model.to_dict()
        if V_std_random != 0:
            dict_save["rng_seed"] = rng_seed
        util.save_arr_data(Path(save_folder_path, "np_arrays"), arr_str_list_to_save, [eval(a) for a in arr_str_list_to_save])
        util.save_dict(Path(save_folder_path, "tb_model_params"), dict_save)
        logging.debug(f"Files saved to {save_folder_path}")