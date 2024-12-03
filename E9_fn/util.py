import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import gftool as gt
from typing import Iterable
from sympy.physics.wigner import wigner_6j
from sympy.core.numbers import Rational

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

#%% Linear algebra
def IsHermitian(mat):
    """Determine if input matrix is Hermitian."""
    return np.allclose(mat, np.asmatrix(mat).H)

def IsReal(mat):
    """Determine if input matrix is purely real."""
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

#%% Simple physics stuff
def I_from_power(P0, w0):
    """[W/m^2] Return peak intensity of a gaussian beam with power P and beam waist w0.
    
    Also, I = (c_light * n * epsilon_0 / 2) * |E|**2 .
    P0: [W] Power
    w0: [m] beam WAIST (the RADIUS of the beam at 1/e^2 intensity)"""
    return 2 * P0 / np.pi / w0**2

def rayleigh_range(w0, lamb_in):
    return np.pi * w0**2 / lamb_in

def quadrupole_Bfield(pos, coil_coeff, I):
    """Returns the B field at pos (relative to coil center) when the coil pair is configured to generate a quadrupole field.
    
    pos: a (3 x n) array, where pos[:,i] is the i-th spatial point
    This is only accurate near the center. For off-center fields, consider modelling with magpylib instead."""
    M = np.diag([0.5, 0.5, -1])
    B = M @ (I * coil_coeff * pos)
    return B
#%% Special functions not defined in scipy
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
        - It returns 0 at the VHS of the second band."""
    return gt.lattice.kagome.dos(E - hbw * (2 / 3), half_bandwidth = hbw) + (1/3) * dirac_delta(E, x0 = 2 * hbw, hw = fhbw)

# General stuff
def dirac_delta(x, x0 = 0, hw = 1e-6):
    """An approximation of the Dirac delta function with norm 1.
    
    I am just using a narrow rectangle with width 2 * hw for now.
        x0: position of the delta function.
        hw: halfwidth of the rectangle."""
    if hw == 0:
        return 0
    else:
        return rect_fn(x, x0 = x0 - hw, x1 = x0 + hw) / (2 * hw)

def Gaussian_1D(x, s = 1, mu = 0):
    """The Gaussian normal distribution (i.e. integrates to 1)."""
    return np.exp(-(1/2) * (x - mu)**2 / s**2) / (s * np.sqrt(2 * np.pi))

def LogisticFn(x, x0 = 0, k = 1):
    """Returns the logistic function, 1/(1 + exp(- k * (x - x0)))."""
    return 1/(1 + np.exp(- k * (x - x0)))

def rect_fn(x, x0: float = 0, x1: float = 1.):
    """Returns the rectangular function, f = 1 for x1 >= x >= x0, f = 0 otherwise."""
    return step_fn(x, x0) * step_fn(-x, -x1)

def step_fn(x, x0: float = 0):
    """Returns the step function, f = 1 for x >= x0, f = 0 otherwise.
    
    Can easily extend to different behaviors at x0 if needed."""
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

def GetClosedPolygon(vert):
    '''Generate the Path defined by a set of vertices that define a closed polygon.'''
    path_code = [Path.MOVETO] + [Path.LINETO for _ in vert[:-2]] + [Path.CLOSEPOLY]
    return Path(vert, path_code)