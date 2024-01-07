import numpy as np
import matplotlib.pyplot as plt

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
def Gaussian_1D(x, s = 1, mu = 0):
    """The Gaussian normal distribution (i.e. integrates to 1)."""
    return np.exp(-(1/2) * (x - mu)**2 / s**2) / (s * np.sqrt(2 * np.pi))

def fermi_stat(E, tau, mu = 0):
    """The Fermi statistics function.
    
    Be careful about units!
        E: energy of the orbital
        tau: fundamental temperature, \tau = k_B * T
        mu: chemical potential, if considered"""
    return 1/(np.exp((E - mu) / tau) + 1)

def LogisticFn(x, x0 = 0, k = 1):
    """Returns the logistic function, 1/(1 + exp(- k * (x - x0)))."""
    return 1/(1 + np.exp(- k * (x - x0)))