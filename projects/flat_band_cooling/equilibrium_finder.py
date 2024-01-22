# Recommended import call: import equilibrium_finder as eqfind
import sys
from copy import deepcopy
import logging
import numpy as np
from scipy.optimize import root_scalar, RootResults
# User defined modules
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
import E9_fn.E9_models as E9M
# Algorithms for finding equilibrium conditions under different circumstances

def isentropic_solver(S0: float,
                      exp0: E9M.DoS_exp,
                      max_step: int = 50,
                      tol: float = 1e-4) -> (float, RootResults):
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
        rrst: the RootResults object returned by root_scalar for full information.
        # exp_eq: equilibrium configuration that satisfies the initial condition.'''
    def S_err(T_in, exp_in, S_in):
        '''Deviation in entropy. (want to find zeros)'''
        exp_eq = deepcopy(exp_in)
        exp_eq.T = T_in
        exp_eq.find_outputs()
        return sum(exp_eq.Ss)- S_in

    rrst = root_scalar(S_err, args = (exp0, S0), x0 = exp0.T, method = "secant",
                       rtol = tol, maxiter = max_step) # outputs a root_result object
    if not rrst.converged: logging.warning("Algorithm failed to converge!")
    return rrst.root, rrst