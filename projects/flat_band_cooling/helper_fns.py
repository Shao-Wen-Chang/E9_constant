import sys
from pathlib import Path as fPath
E9path = fPath("C:/", "Users", "ken92", "Documents", "Studies", "E5", "simulation", "E9_simulations")
if str(E9path) not in sys.path:
    sys.path.insert(1, str(E9path))
from E9_fn import util
import numpy as np
from matplotlib.path import Path as plt_Path
from scipy.signal import convolve

def get_model_str(lat_str, lat_dim, sys_len, V_rsv, runnum: int = 1, param_dict = dict()):
    """Construct a standardized string that uniquely describes a model configuration.

    This function encodes key model parameters into a single string for use in file naming,
    logging, or version tracking. Additional parameters are passed via `param_dict` and are
    appended in alphabetical order of their keys for consistency.

    Args:
        lat_str: 
            A short string identifying the lattice type (e.g., "kagome").
        lat_dim: 
            A tuple or list of two integers specifying the lattice dimensions (e.g., (20, 20)).
        sys_len: 
            An integer specifying the linear system size.
        V_rsv: 
            A float representing the value of the `V_rsv` parameter in the model.
        runnum: 
            Optional integer run number; appended to the string if greater than 1.
        param_dict: 
            Optional dictionary of additional parameter names (as strings) and their float values.
            These will be sorted alphabetically by key and appended as `_key<val>` with 3-digit precision.

    Returns:
        A string uniquely describing the model configuration, with all numeric periods
        replaced by `'p'` for filename compatibility.
    """
    runnum_str = f"_{runnum:03d}" if runnum > 1 else ""
    sorted_dict = dict(sorted(param_dict.items()))
    other_params_str = "_".join([""] + [f"{k}{v:.3f}" for k, v in sorted_dict.items()])
    return (f"{lat_str}_lat{lat_dim[0]}x{lat_dim[1]}"
            f"_sys{sys_len}x{sys_len}_Vrsv{V_rsv:.4f}{other_params_str}{runnum_str}").replace(".", "p")

def get_finite_res_box(lattice_dim, sys_range, tbmodel, l_res):
    """Return the box potential (with amplitude 1) convolved with a Gaussian with width given by l_res.
    
    tbmodel is required to get the real space position of each lattice site.
    I assume 2D lattice for now.
    """
    # Find the size of the whole system
    lv1, lv2 = tbmodel.lat_vec
    padding = 5
    padded_lat_origin = -padding * lv1 - padding * lv2
    lat_size = lattice_dim[0] * lv1 + lattice_dim[1] * lv2  # This is the position of the farthest unit cell
    padded_lat_size = (lattice_dim[0] + padding) * lv1 + (lattice_dim[1] + padding) * lv2  # This is the position of the farthest unit cell
    s1, s2 = sys_range
    sv1 = s1 * lv1 + s1 * lv2
    sv2 = s1 * lv1 + s2 * lv2
    sv3 = s2 * lv1 + s2 * lv2
    sv4 = s2 * lv1 + s1 * lv2
    sys_region = util.get_closed_polygon([sv1, sv2, sv3, sv4, sv1])

    # Get a fine-grained 2D matrix with the sharp potential
    dx = 0.05
    xi, xf, yi, yf = padded_lat_origin[0], padded_lat_size[0], padded_lat_origin[1], padded_lat_size[1]
    x = np.arange(xi, xf, dx * np.sign(xf - xi))
    y = np.arange(yi, yf, dx * np.sign(yf - yi))
    xx, yy = np.meshgrid(x, y)  # shape (grain_num, grain_num)

    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    mask_flat = sys_region.contains_points(grid_points) 
    mask = mask_flat.reshape(xx.shape)
    V_box_sharp = np.where(mask, 0, 1)  # 0 inside polygon, 1 outside

    # Convolve with a 2D Gaussian
    w_conv = 5 * l_res
    xg = np.arange(-w_conv, w_conv, dx)
    yg = np.arange(-w_conv, w_conv, dx)
    grid_gauss = np.meshgrid(xg, yg)
    gauss_res = dx**2 * util.gaussian_2D(grid_gauss, 0, 0, l_res, l_res)
    V_box = convolve(V_box_sharp, gauss_res, mode = "same")

    # Sample the fine-grained 2D matrix at the position of mat_in
    all_reduced_ind = np.arange(tbmodel.n_orbs)
    H_box = np.zeros(tbmodel.n_orbs)
    for i in all_reduced_ind:
        pos = tbmodel.get_lat_pos(i)
        x_ind = util.find_nearest(x, pos[0])[1] #int(np.round(pos[0] / lat_size[0] * (grain_num - 1)))
        y_ind = util.find_nearest(y, pos[1])[1] #int(np.round(pos[1] / lat_size[1] * (grain_num - 1)))
        H_box[i] = V_box[y_ind, x_ind]
    return np.diag(H_box)