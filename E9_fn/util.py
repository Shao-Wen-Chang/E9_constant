import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable

#%% dictionary manipulation
def all_values_from_key(dict_iter: Iterable[dict], key, default = 'raise_error'):
    '''Given an iterable of dictionaries, return the value in each that correspond to key.
    
    The default behavior is to raise error if a key value doesn't exist in one of the
    dictionary. The output is of the same type as dict_iter.
        default: if == 'raise_error', then raise KeyError. Otherwise, it sets the value
                 to use if a dictionary doesn't have the key value given.'''
    iter_type = type(dict_iter)
    if default == 'raise_error':
        return iter_type([x[key] for x in dict_iter])
    else:
        return iter_type([x.get(key, default) for x in dict_iter])

#%% np.array manipulation
def find_nearest(arr, val):
    '''Find the nearest value to val in arr, returned as a tuple of (value, idx)'''
    idx = (np.abs(arr - val)).argmin()
    return (arr[idx], idx)

def find_y_by_nearest_x(xarr, yarr, val):
    '''For a dataset (x, y), find the value of y, whose corresponding x is closest to val, and return as a tuple of (x, y)
    
    xarr and yarr should have the same shape.'''
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

#%% Special functions not defined in scipy
def bose_stat(E, tau, mu = 0):
    """The Bose statistics function.
    
    Be careful about units!
        E: energy of the orbital
        tau: fundamental temperature, \tau = k_B * T
        mu: chemical potential, if considered"""
    return 1/(np.exp((E - mu) / tau) - 1)

def Gaussian_1D(x, s = 1, mu = 0):
    """The Gaussian normal distribution (i.e. integrates to 1)."""
    return np.exp(-(1/2) * (x - mu)**2 / s**2) / (s * np.sqrt(2 * np.pi))

def fermi_stat(E, tau, mu = 0):
    """The Fermi statistics function."""
    return 1/(np.exp((E - mu) / tau) + 1)

def LogisticFn(x, x0 = 0, k = 1):
    """Returns the logistic function, 1/(1 + exp(- k * (x - x0)))."""
    return 1/(1 + np.exp(- k * (x - x0)))

def step_fn(x, x0: float = 0):
    '''Returns the step function, f = 1 for x >= x0, f = 0 otherwise.
    
    Can easily extend to different behaviors at x0 if needed.'''
    return np.array(x >= x0).astype(float)

def rect_fn(x, x0: float = 0, x1: float = 0):
    '''Returns the rectangular function, f = 1 for x1 >= x >= x0, f = 0 otherwise.'''
    return step_fn(x, x0) * step_fn(-x, -x1)

#%% Helper plotting functions
def plot_delta_fn(ax,
                  x0: float = 0,
                  a0: float = 1,
                  a_plt: float = 1,
                  text: str = None,
                  axis = 'x',
                  *kwargs):
    '''Plot the delta function, a0 * delta(x - x0), as an arrow.
    
    The arrow makes it clear that the delta function is not just a sharply peaked function.
        ax: the Axes object to be plotted on.
        x0: position of the delta function peak.
        a0: actual value that the delta function integrates to. Only affects the text.
        a_plt: the length of the arrow on the plot, normalized to y_lim (or x_lim).
        text: the text to be added next to the arrow to indicate the actual value of
              a0. Default to a0 if None.
        axis: which axis is used as the variable. Default is x (horizontal axis).
        *kwargs: must be plot-related arguments accepted by arrow().'''
    # Initialize arguments to arrow()
    if text is None: text = str(a0)
    if axis == 'x':
        (xi, yi, dx, dy) = (x0, 0, 0, a0)
    elif axis == 'y':
        (xi, yi, dx, dy) = (0, x0, a0, 0)
    else:
        raise("axis must be \'x\' or \'y\'")
    
    arr = ax.arrow(xi, yi, dx, dy, *kwargs)
    return arr