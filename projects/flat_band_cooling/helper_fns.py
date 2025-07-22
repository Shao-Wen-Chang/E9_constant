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

def add_finite_res(lattice_dim, sys_range, tbmodel, l_res, window_hw = 5):
    """Return mat_in convolved with a Gaussian with width given by l_res.
    
    tbmodel is required to get the real space position of each lattice site.
    I assume 2D lattice for now.
    """
    # Find the size of the whole system
    lv1, lv2 = tbmodel.lat_vec
    lat_size = lattice_dim[0] * lv1 + lattice_dim[1] * lv2  # This is the position of the farthest unit cell
    s1, s2 = sys_range
    sv1 = s1 * lv1 + s1 * lv2
    sv2 = s1 * lv1 + s2 * lv2
    sv3 = s2 * lv1 + s2 * lv2
    sv4 = s2 * lv1 + s1 * lv2
    sys_region = util.get_closed_polygon([sv1, sv2, sv3, sv4])

    # Get a fine-grained 2D matrix with the sharp potential
    

    # Convolve with a 2D Gaussian

    # Sample the fine-grained 2D matrix at the position of mat_in
    # ix, iy = np.arange(-window_hw, window_hw + 1), np.arange(-window_hw, window_hw + 1)
    # ixx, iyy = np.meshgrid(ix, iy)
    # gauss_res = util.gaussian_2D