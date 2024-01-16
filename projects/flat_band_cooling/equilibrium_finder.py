# Recommended import call: import equilibrium_finder as eqfind
import sys
from copy import deepcopy
import numpy as np
from scipy import root_scalar
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
import E9_fn.E9_models as E9M
# Algorithms for finding equilibrium conditions under different circumstances

def isentropic_solver(S0: float, exp0: E9M.DoS_exp, max_step: int = 50, tol: float = 1e-6):
    '''Find thermal equlibrium config, in particular T, given some total entropy.
    
    See my personal notes on 2024/01/15 for some physical considerations.
    Since I am using root_scalar, only the temperature of the system is returned.
    This is a bit wasteful, but find_outputs() runs pretty fast.
    Arguments:
        S0: total entropy of the system. Physically this is given by e.g. how cold
            we can get after sympathetic evaporation.
        exp0: DoS_exp that specifies initial conditions. This object is left
              unmodified. Some notes about the defining parameters:
            T: This is used as an initial guess for the solver. The final value of
               T will be whatever gives the correct entropy.
        max_setp: number of steps to try before the algorithm is terminated
        tol: acceptable relative tolerance in the final entropy.
    Return:
        rrst.root: temperature of the equilibrated system.
        rrst: the root_result object returned by root_scalar for full information.
        # exp_eq: equilibrium configuration that satisfies the initial condition.'''
    def S_err(T_in, exp_in, S_in):
        '''Try to use scipy's root finding function to find S.'''
        exp_eq = deepcopy(exp_in)
        exp_eq.T = T_in
        # print something to make sure that deep copy is doing its job
        print("[{}] from exp_in: T = {}; from exp_eq: T = {}".format(i, exp_in.T, exp_eq.T))
        exp_eq.find_outputs()
        return exp_in["S"] - S_in

    rrst = root_scalar(S_err, arg = (exp0, S0), x0 = exp0.T,
                       rtol = tol, maxiter = max_step) # outputs a root_result object
    if not rrst.converged: print("Algorithm failed to converge!")
    return rrst.root, rrst