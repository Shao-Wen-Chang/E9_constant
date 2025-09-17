import E9_fn.E9_constants as E9c
import logging
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.path import Path as plt_Path
from pathlib import Path as fPath
import gftool as gt
from typing import Iterable
from sympy.physics.wigner import wigner_6j
from sympy.core.numbers import Rational
import pickle
import json

#%% File manipulations
def save_dict(save_path: fPath, dict_in: dict, use_pickle: bool = True, use_json: bool = True):
    """Save dict_in in both pickle and JSON format.

    Loading pickle: e.g.
    with open("data/kagome_model.pkl", "rb") as f:
        model_dict = pickle.load(f)

    Loading JSON: e.g.
    with open("data/kagome_model.json", "r") as f:
        model_dict = json.load(f)
    
    Parameters:
        dict_in:    dictionary to be saved.
        save_path:  pathlib.Path object (no extension), e.g. Path("data/kagome_model")
        use_pickle: if True, save as a pickle file.
        use_json:   if True, save as a JSON file.
    """
    save_path.parent.mkdir(parents = True, exist_ok = True)

    if use_pickle:
        with open(save_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(dict_in, f)

    if use_json:
        def make_json_safe(obj):
            if isinstance(obj, np.ndarray):
                if np.iscomplexobj(obj):
                    return {
                        "__complex_array__": True,
                        "real": make_json_safe(obj.real),
                        "imag": make_json_safe(obj.imag)
                    }
                else:
                    return obj.tolist()
            elif isinstance(obj, complex):
                return {
                    "__complex__": True,
                    "real": obj.real,
                    "imag": obj.imag
                }
            elif isinstance(obj, (np.complexfloating,)):
                return {
                    "__complex__": True,
                    "real": obj.real.item(),
                    "imag": obj.imag.item()
                }
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (tuple, set)):
                return [make_json_safe(x) for x in obj]
            elif isinstance(obj, list):
                return [make_json_safe(x) for x in obj]
            elif isinstance(obj, dict):
                return {str(k): make_json_safe(v) for k, v in obj.items()}
            else:
                return obj
        safe_dict = make_json_safe(dict_in)

        with open(save_path.with_suffix(".json"), "w") as f:
            json.dump(safe_dict, f, indent = 4)

def save_arr_data(file_path, arr_str_list, arr_list):
    """Save specified arrays using np.savez."""
    if len(arr_list) != len(arr_str_list):
        raise("Length of arr_str_list and arr_list doesn't match!")
    
    file_path.parent.mkdir(parents = True, exist_ok = True)
    arr_dict = {arr_str: arr for arr_str, arr in zip(arr_str_list, arr_list)}
    np.savez(file_path, **arr_dict)

#%% dictionary manipulation
def all_values_from_key(dict_iter: Iterable[dict], key, default = 'raise_error'):
    """Given an iterable of dictionaries, return the value in each that correspond to key.
    
    The default behavior is to raise error if a key value doesn't exist in one of the
    dictionary. The output is of the same type as dict_iter.
        default: if == 'raise_error', then raise KeyError. Otherwise, it sets the value
                 to use if a dictionary doesn't have the key value given."""
    iter_type = type(dict_iter)
    if default == 'raise_error':
        return iter_type([x[key] for x in dict_iter])
    else:
        return iter_type([x.get(key, default) for x in dict_iter])

#%% np.array manipulation
def find_nearest(arr, val):
    """Find the nearest value to val in arr, returned as a tuple of (value, idx)"""
    idx = (np.abs(arr - val)).argmin()
    return (arr[idx], idx)

def find_y_by_nearest_x(xarr, yarr, val):
    """For a dataset (x, y), find the value of y, whose corresponding x is closest to val, and return as a tuple of (x, y)
    
    xarr and yarr should have the same shape."""
    xnear, idx = find_nearest(xarr, val)
    return (xnear, yarr[idx])

def find_derivative(arr, dx = 1, assume_C3_conti = True):
    """Find the derivative of an array, assuming uniform spacing dx.
    
    Returns an array of the same shape as arr, with the last element set to the same
    value as the last derivative.
    """
    if arr.ndim != 1:
        raise Exception("find_derivative only works for 1D arrays")
    
    if assume_C3_conti:
        return np.gradient(arr, dx)
    else:    # If not assuming C3 continuity, use the finite difference method
        deriv = np.zeros_like(arr)
        deriv[:-1] = (arr[1:] - arr[:-1]) / dx
        deriv[-1] = deriv[-2]
        return deriv

def find_sign_change(arr):
    """Find the indices where the sign of arr changes.
    
    Returns an array of indices where arr[i] * arr[i + 1] < 0.
    """
    if arr.ndim != 1:
        raise Exception("find_sign_change only works for 1D arrays")
    node_finder = arr[:-1] * arr[1:]
    return np.where(node_finder < 0)[0]

def arr_insert_sorted(arr: np.ndarray, values):
    """Insert one or more values into a sorted 1D NumPy array while keeping it sorted.
    Duplicate values already in arr will be skipped.

    Args:
        arr:
            A 1D NumPy array, assumed to be sorted in increasing order.
        values:
            A scalar or iterable of values to insert into arr. Can be in arbitrary order.

    Returns:
        A new NumPy array with the values inserted in the correct positions,
        without duplicating existing entries.
    """
    arr = np.asarray(arr)
    values = np.atleast_1d(values)
    values = np.sort(values)
    values = np.setdiff1d(values, arr, assume_unique = False)   # remove duplicates

    if values.size == 0:
        return arr  # nothing to insert
    idxs = np.searchsorted(arr, values)
    return np.insert(arr, idxs, values)

#%% Linear algebra
def dagger(mat, axis = None):
    """Return the conjugate transpose of the input matrix.
    
    I guess ndarray doesn't incorporate .H for arrays because it can be multidimensional."""
    if mat.ndim <= 2:
        return mat.conj().T
    else:
        if axis is None:
            raise(Exception(("The axes to be daggered must be provided for arrays with"
                             " dimensions larger than 2, e.g. axis = (1, 2).")))
        elif len(axis) != 2:
            raise(Exception("axis must be a list with two elements"))
        ind_order = np.arange(mat.ndim)
        ind_order[axis] = np.array(axis)[::-1]
        return np.einsum(mat, ind_order).conj()

def is_density_matrix(mat):
    """Determine if the input matrix is a valid density matrix."""
    if not IsHermitian(mat):
        return False
    elif not np.allclose(mat.diagonal().sum(), 1):
        return False
    elif not is_p_semidef(mat):
        return False
    return True

def IsHermitian(mat):
    """Determine if the input matrix is Hermitian."""
    return np.allclose(mat, np.asmatrix(mat).H)

def is_p_semidef(mat, tol = 1e-8):
    """Determine if the input matrix is positive semidefinite, tolerating for rounding error."""
    if not IsHermitian(mat):
        raise(Exception("not accepting non-Hermitian matrices for now!"))
    e_vals = eigh(mat, eigvals_only = True)
    return np.all(e_vals > -tol)

def is_unitary(mat):
    """Determine if the input matrix is unitary."""
    return np.allclose(np.eye(mat.shape[0]), np.asmatrix(mat).H * mat)

def IsReal(mat):
    """Determine if the input matrix is purely real."""
    return np.allclose(mat, np.conj(mat))

def Normalize(vec, norm = 1):
    """Return the normalized input vector."""
    norm0 = np.linalg.norm(vec)
    return vec / norm0 * np.sqrt(norm)

def VecProj(a, b):
    """Return the projection of a on b (with C.C.)"""
    return np.inner(a.conj(), b) * b / np.linalg.norm(b)

def VecNormal(a, b):
    """Return the vector component of a that's normal to b"""
    return a - VecProj(a, b)

def VecTheta(theta):
    """Return the unit vector with polar angle theta."""
    return np.array([np.cos(theta), np.sin(theta)])

def get_red_den_mat(rho, ind_sub):
    """Get the reduced density matrix of some subspace of the full Hilbert space.
    
    This is not a partial trace, since the subspace does not necessarily partition the original Hilbert
    space into cosets. One considers instead writing HS_tot = HS_sub (tensorsum) HS_else, and "trace" out
    HS_else by assigning the vacuum state to them. It seems a bit arbitrary.

    In the current implementation, HS_tot is assumed to be a single particle Fock space, and HS_sub is
    the single & vacuum particle Fock space of the subsystem.

    Args:
        rho:        The input density matrix of dimension (n, n).
        ind_sub:    A (m < n)-dimensional 1D array that specifies the indices belonging to the subspace.
    
    return:
        A (m + 1, m + 1)-dimensional density matrix. The first m indicies are entries in ind_sub, in that
        order, and the last index is the vacuum space of the subsystem.
    """
    if not is_density_matrix(rho):
        raise(Exception("The input density matrix is not valid!"))
    
    dim_rho = rho.shape[0]
    mask = np.full(dim_rho, True)
    mask[ind_sub] = False
    ind_vac = np.arange(dim_rho)[mask]  # indices that don't count towards the system (, to be summed to vacuum state)
    rho_sub = rho[:,ind_sub][ind_sub,:]
    rho_sub = np.pad(rho_sub, ((0, 1), (0, 1)), "constant", constant_values = 0.)
    rho_sub[-1, -1] = rho[ind_vac, ind_vac].sum()
    return rho_sub

#%% Simple physics stuff
# Basic thermodynamical stuff
def lambda_de_broglie(m, T):
    return E9c.hnobar / np.sqrt(2 * np.pi * m * E9c.k_B * T)

# Gaussian beam related
def I_from_power(P0, w0x, w0y = None):
    """[W/m^2] Return peak intensity of a gaussian beam with power P and beam waist w0.
    
    Also, I = (c_light * n * epsilon_0 / 2) * |E|**2 .

    Args:
        P0:  [W] Power
        w0x: [m] beam WAIST (the RADIUS of the beam at 1/e^2 INTENSITY)
        w0y: [m] beam WAIST in the y direction (if not given, use w0x).
    """
    if w0y is None: w0y = w0x
    return 2 * P0 / np.pi / (w0x * w0y)

def rayleigh_range(lamb_in, w0):
    """Returns the Rayleigh range of a Gaussian beam.
    
    This (and functions with only w0 input below) is only considering one direction.
    An elliptical beam has different Rayleigh ranges in the x and y directions.
    """
    return np.pi * w0**2 / lamb_in

def w_gaussian_beam(z, lamb_in, w0):
    """Returns the beam waist at z of a Gaussian beam."""
    return w0 * np.sqrt(1 + (z / rayleigh_range(lamb_in, w0))**2)

def I_gaussian_beam_3D(r, z, lamb_in, w0x, w0y = None, theta = 0):
    """Returns the intensity of a Gaussian beam at (r, z, theta).
    
    z-axis is taken to be the axis of beam propagation. Note that this experssion
        i)   is for intensity, not electric field.
        ii)  doesn't include phases.
        iii) is different from the normal distribution (no 1/2 in the exponent).
    Beam intensity is normalized to be 1 at the maximum (so that it plays well with I_from_power).
    """
    if w0y is None: w0y = w0x
    wzx = w_gaussian_beam(z, lamb_in, w0x)
    wzy = w_gaussian_beam(z, lamb_in, w0y)
    x, y = r * np.cos(theta), r * np.sin(theta)
    return (w0x / wzx) * np.exp(-(x / wzx)**2 / 2) * (w0y / wzy) * np.exp(-(y / wzy)**2 / 2)

def x2approx_gaussian_beam_3D(lamb_in, w0, theta = 0):
    """Returns the harmonic oscillator approximation (i.e. taylor expansion up to x^2) of a gaussian beam.
    
    For a gaussian beam given by I(r, z) = I_max * I_gaussian(r = l * cos(theta), z = l * sin(theta), ...)
    , the expansion around l = 0 is
        I(r, z) = I_max * (1 - (return value of this function) * l^2 + O(l^4))
    , and to approximate trap frequency, use: (V_max = I_max * polarizibility)
        V(r, z) = V_max * (return) = (1 / 2) * m * w_eff^2 (up to 2nd order)
    to get, for example,
        w_eff(theta = np.pi/2) = np.sqrt(4 * V_max / m / w0**2)
    
    One gets the result for arbitrary theta easily by "adding up" the contribution from the (w0/w(z)) term
    and the gaussian term.

    Args:
        theta:  angle with respect to the z (optical) axis."""
    z_part = 1 / rayleigh_range(lamb_in, w0)**2
    r_part = 2 / w0**2
    return np.cos(theta)**2 * z_part + np.sin(theta)**2 * r_part

# Others
def quadrupole_Bfield_vec(pos, Bz_grad):
    """Returns the B field at pos (relative to coil center) when the coil pair is configured to generate a quadrupole field.
    
    This is only accurate near the center. For off-center fields, consider modelling with magpylib instead.
    TODO: bad function, why should n be 1-dim?
    
    Args:
        pos:        a (3 x n) array, where pos[:,i] is the i-th spatial point
        Bz_grad:    Magnetic field gradient in the z-direction (the most tightly confined direction).
    """
    M = np.diag([0.5, 0.5, -1])
    B = M @ (Bz_grad * pos)
    return B

#%% Special functions not defined in scipy
def gaussian_2D(xy,
                x0: float,
                y0: float, 
                sx: float, 
                sy: float,
                amp: float = 1.,
                angle: float = 0.,
                ):
    """2D Gaussian curve.

    Args:
        xy:             xy-data. The x-data is xy[0] and the y data is xy[1].
        x0:             The 2D Gaussian center location in x.
        y0:             The 2D Gaussian center location in y.
        sx:             The 2D Gaussian standard deviation in x (68% within +-1 sigma).
                        If an angle is provided, this is the standard deviation in the polar direction of angle.
        sy:             The 2D Gaussian standard deviation in y (68% within +-1 sigma).
        amp:            The amplitude of the Gaussian such that it integrates to amp (when offset is 0).
        angle:          [DEGREES] The angle of rotation of the Gaussian.
    
    Returns:
        A 2D array.
    """
    x, y = xy[0], xy[1]
    angle_rad = np.radians(angle)
    rx = np.cos(angle_rad) * (x - x0) + np.sin(angle_rad) * (y - y0)
    ry = - np.sin(angle_rad) * (x - x0) + np.cos(angle_rad) * (y - y0)
    
    return amp / 2 / np.pi / sx / sy * np.exp(-(1 / 2) * ((rx / sx)**2 + (ry / sy)**2))


def wigner_6j_safe(j1, j2, j3, j4, j5, j6):
    """Pitfall-free version of Wigner 6j symbol."""
    # Check that the inputs are half-integers - note that inputs such as 9/2 or 4.5 are not necessarily converted
    # to half integers. See e.g. https://github.com/sympy/sympy/issues/26219
    original_js = (j1, j2, j3, j4, j5, j6)
    new_js = [Rational(j).limit_denominator(3) for j in original_js]
    for nj, oj in zip(new_js, original_js):
        if abs(float(nj) * 2 - oj * 2) > 1e-7:
            raise ValueError("invalid j in input: j = {}".format(oj))
    
    return wigner_6j(*new_js)

# particle statistics
def part_stat(E, tau, mu, xi, replace_inf = "Don't"):
    """Fermi (xi = +1) or Bose (xi = -1) statistics function.
    
    Useful for coding. bose_stat and fermi_stat are the two possible cases for
    part_stat, and are derived from part_stat for readability where it helps.
    Be careful about units!
        E: energy of the orbital
        tau: fundamental temperature, \tau = k_B * T
        mu: chemical potential, if considered
        xi: 1 for fermions, -1 for bosons
        replace_inf: the value used to replace any inf. If "Don't" (default),
                     then raise an error."""
    if xi != 1 and xi != -1:
        logging.error("xi = {}".format(xi))
        raise Exception("xi must be 1 or -1 in part_stat")
    
    output = 1/(np.exp((E - mu) / tau) + xi)
    if np.isinf(output).any():
        if replace_inf == "Don't":
            logging.error("inf encountered in part_stat")
            raise Exception("no replacement value given")
        else:
            output[np.isinf(output)] = replace_inf
            logging.info("inf encountered in part_stat; replaced with {}".format(replace_inf))
    return output

def bose_stat(E, tau, mu = 0.):
    """The Bose statistics function."""
    return part_stat(E, tau, mu, xi = -1)

def fermi_stat(E, tau, mu = 0):
    """The Fermi statistics function."""
    return part_stat(E, tau, mu, xi = 1)

# Density of states (lowest energy is 0 by convention)
def kagome_DoS(E, hbw: float = 1., fhbw: float = 0.):
    """The density of state of kagome lattice, possibly including a 'flat-ish' band.
    
    The dispersive part integrates to 2/3. If fhbw != 0, then there is also a "flat band"
    that integrates to 1/3.
    I redefined the zero to be at the bottom of the band structure.
    Args:
        hbw:    half bandwidth ( = tight-binding t x 3).
        fhbw:   half bandwidth of the flat band, typically a small finite number
                (if zero, the flat band is omitted, and the function only integrates to 2/3.)
    See gftool.lattice.kagome.dos for more detail. Some notes of the original dos:
        - This function integrates to 2/3 since the flat band is not included.
        - It returns 0 at the VHS of the second band.
    """
    return gt.lattice.kagome.dos(E - hbw * (2 / 3), half_bandwidth = hbw) + (1/3) * dirac_delta(E, x0 = 2 * hbw, hw = fhbw)

# General stuff
def is_int(num):
    """Check if num is an integer or not."""
    return np.isclose(num % 1, 0)

def dirac_delta(x, x0 = 0, hw = 1e-6):
    """An approximation of the Dirac delta function with norm 1.
    
    I am just using a narrow rectangle with width 2 * hw for now.
        x0: position of the delta function.
        hw: halfwidth of the rectangle.
    """
    if hw == 0:
        return 0
    else:
        return rect_fn(x, x0 = x0 - hw, x1 = x0 + hw) / (2 * hw)

def Gaussian_1D(x, s = 1, mu = 0):
    """The Gaussian normal distribution (i.e. integrates to 1)."""
    return np.exp(-(1/2) * (x - mu)**2 / s**2) / (s * np.sqrt(2 * np.pi))

def gaussian_1D_integral(s):
    """The integral of a gaussian with width s and amplitude 1."""
    return np.sqrt(2 * np.pi) * s

def two_gaussian_1D_max_product(s1, s2, mu):
    """The maximum value of two Gaussians multiplied together.
    
    Both Gaussians are normalized such that they individually integrate to 1.

    Args:
        s1, s2: The standard deviations of the two Gaussians.
        mu:     The difference in the center of the two Gaussians.
    """
    return np.exp(-(1/2) * mu**2 / (s1**2 + s2**2)) / (s1 * s2 * 2 * np.pi)

def two_gaussian_1D_max_product_pos(s1, s2, mu):
    """The position at which two Gaussians multiplied together has the maximum value."""
    return mu * s1**2 / (s1**2 + s2**2)

def two_gaussian_1D_integral(s1, s2, mu):
    """The integral of two normal gaussian distributions multiplied together."""
    return two_gaussian_1D_power_integral(s1, s2, mu, 1, 1)

def two_gaussian_1D_power_integral(s1, s2, mu, p1, p2):
    """The integral of two normal gaussian distributions raised to some power and then multiplied together.
    
    See my notes on Gaussian integrals.
    """
    s1p, s2p = s1 / np.sqrt(p1), s2 / np.sqrt(p2)
    S = 1 / np.sqrt(1 / s1p**2 + 1 / s2p**2)
    c1, c2, cS = gaussian_1D_integral(s1), gaussian_1D_integral(s2), gaussian_1D_integral(S)
    return cS / (c1**p1 * c2**p2) * np.exp(-mu**2 / 2 / (s1p**2 + s2p**2))

def LogisticFn(x, x0 = 0, k = 1):
    """Returns the logistic function, 1/(1 + exp(- k * (x - x0)))."""
    return 1/(1 + np.exp(- k * (x - x0)))

def rect_fn(x, x0: float = 0, x1: float = 1.):
    """Returns the rectangular function, f = 1 for x1 >= x >= x0, f = 0 otherwise."""
    return step_fn(x, x0) * step_fn(-x, -x1)

def step_fn(x, x0: float = 0):
    """Returns the step function, f = 1 for x >= x0, f = 0 otherwise.
    
    Can easily extend to different behaviors at x0 if needed.
    """
    return np.array(x >= x0).astype(float)

#%% Helper plotting functions
#%% Plot related
def set_custom_plot_style(activate: bool = True, overwrite: dict = {}):
    """Use a set of rcParams that I prefer.
    
    To restore to default values, use plt.rcdefaults().
        overwrite: a dictionary with rcParams settings. This overwrites the
                   original custom values."""
    # The default custom parameters
    custom_rcParams = {"font.size": 18,                     # Default is 10
                       "axes.labelsize": "large",
                       "axes.formatter.limits": (-3, 3),    # Default is [-5, 6]
                       "xtick.minor.visible": True,
                       "ytick.minor.visible": True,
                       "figure.autolayout": True,           # Use tight layout
                       } | overwrite

    if activate:
        for key, value in custom_rcParams.items():
            plt.rcParams[key] = value
    else:
        plt.rcdefaults()

def get_color(var, var_list: np.ndarray, cmap, assignment = "index", 
              crange = (0.0, 1.0), cfn = lambda x: x):
    """Get a color from a colormap based on a variable's index or value, 
    with optional range and scaling adjustments.

    Args:
        var:
            The variable whose color is to be determined. 
            Should either be a member of var_list (if assignment = "index")
            or a numeric value comparable to elements in var_list (if assignment = "value").
        var_list:
            A NumPy array (or list) of reference values. Determines how colors are mapped.
        cmap:
            A Matplotlib colormap object (e.g., plt.cm.viridis).
        assignment:
            Specifies how to map the variable to a color:
            - "index": Color is chosen based on the index of var in var_list.
            - "value": Color is chosen based on var's relative numeric value in var_list.
        crange:
            A tuple (cmin, cmax) defining the normalized range of the colormap to use.
            Default is (0, 1), meaning the full range of cmap.
        cfn:
            A strictly increasing function that rescales the normalized values before
            mapping to colors. It should satisfy cfn(0) = 0 and cfn(1) = 1.
            Example: lambda x: x**3 compresses midrange values.

    Returns:
        A color from the given colormap, represented as an RGB or RGBA tuple.
    """
    if isinstance(var_list, list):
        var_list = np.array(var_list)

    if assignment == "index":
        try:
            arg = np.where(var_list == var)[0][0]
        except ValueError:
            logging.error(f"Variable {var} not found in the list.")
            raise
        frac = arg / (len(var_list) - 1) if len(var_list) > 1 else 0.5

    elif assignment == "value":
        vmin, vmax = np.min(var_list), np.max(var_list)
        span = vmax - vmin
        frac = (var - vmin) / span if span != 0 else 0.5

    else:
        raise ValueError("Invalid assignment type. Use 'index' or 'value'.")

    # Apply colormap range scaling
    cmin, cmax = crange
    if not (0.0 <= cmin < cmax <= 1.0):
        raise ValueError("crange must satisfy 0.0 <= cmin < cmax <= 1.0")
    scaled_frac = cmin + (cmax - cmin) * frac

    # Apply custom scaling function
    scaled_frac = cfn(scaled_frac)

    return cmap(scaled_frac)

def fix_clabel_orientation(labels):
    """
    Ensure contour labels are always upright (never upside-down).

    Parameters
    ----------
    labels : list of matplotlib.text.Text
        The list returned by ax.clabel().
    """
    for lbl in labels:
        angle = lbl.get_rotation()
        # Wrap angle to [-90, 90] so text is never upside down
        if angle > 90:
            lbl.set_rotation(angle - 180)
        elif angle < -90:
            lbl.set_rotation(angle + 180)

# I want to remove this function
def make_simple_axes(ax = None, fignum = None, fig_kwarg = {}):
    """Make a figure with one single Axes if ax is None, and return (figure, axes).
    
    My favorite thing to have in the beginning of a plotting function."""
    if ax is None:
        f = plt.figure(num = fignum, **fig_kwarg)
        f.clf()
        ax = f.add_axes(111)
        return (f, ax)
    else:
        return (ax.get_figure(), ax)
        
def plot_delta_fn(ax,
                  x0: float = 0,
                  a_plt: float = 1,
                  text: str = "",
                  text_height: float = 0.,
                  axis = 'x',
                  **kwargs):
    """Plot the delta function, a0 * delta(x - x0), as an arrow, and put a0 next to it.
    
    The arrow must be plotted after limits have changed.
        ax: the Axes object to be plotted on.
        x0: position of the delta function peak.
        text: What to display next to the arrow.
        a_plt: length of the arrow.
        text_height: moves the position of the text up or down by a certain amount.
        axis: which axis is used as the variable. Default is x (horizontal axis).
        *kwargs: must be plot-related arguments accepted by arrow()."""
    # Initialize arguments to arrow()
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    xc, xr = (xlims[0] + xlims[1]) / 2., xlims[1] - xlims[0]
    yc, yr = (ylims[0] + ylims[1]) / 2., ylims[1] - ylims[0]
    if axis == 'x':
        (xi, yi, dx, dy) = (x0, 0, 0, a_plt)
        tx, ty = xi + dx + 0.05 * xr, yi + dy + text_height
    elif axis == 'y':
        (xi, yi, dx, dy) = (0, x0, a_plt, 0)
        tx, ty = xi + dx, yi + dy + 0.05 * yr + text_height
    else:
        raise Exception("axis must be \'x\' or \'y\'")
    
    arr = ax.arrow(xi, yi, dx, dy, head_width = 0.5, head_length = 0.1, **kwargs)
    ax.text(tx, ty, text)
    return arr

def get_closed_polygon(vert):
    '''Generate the plt_Path defined by a set of vertices that define a closed polygon.'''
    path_code = [plt_Path.MOVETO] + [plt_Path.LINETO for _ in vert[:-2]] + [plt_Path.CLOSEPOLY]
    return plt_Path(vert, path_code)