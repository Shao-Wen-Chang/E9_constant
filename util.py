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
def LogisticFn(x, x0 = 0, k = 1):
    """Returns the logistic function, 1/(1 + exp(- k * (x - x0)))."""
    return 1/(1 + np.exp(- k * (x - x0)))