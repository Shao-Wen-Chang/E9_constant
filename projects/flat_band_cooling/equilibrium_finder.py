# Recommended import call: import equilibrium_finder as eqfind
import sys
from copy import deepcopy
import logging
import numpy as np
from scipy.optimize import root_scalar, root, RootResults
# User defined modules
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
import E9_fn.E9_models as E9M
# Algorithms for finding equilibrium conditions under different circumstances

def isentropic_fix_filling_solver(S_tar: float,
                      exp0: E9M.DoS_exp,
                      max_step: int = 50,
                      tol: float = 1e-4) -> (float, RootResults):
    """Find thermal equlibrium config, in particular T, given some total entropy
    and target filling in systems.
    
    See my personal notes on 2024/01/15 for some physical considerations. This function
    is useful for finding the target particle number and temperature / total entropy.
    Arguments:
        S_tar: total entropy of the system. Physically this is given by e.g. how cold
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
        # exp_eq: equilibrium configuration that satisfies the initial condition.
    Since I am using root_scalar, only the temperature of the system is returned.
    This is a bit wasteful, but find_outputs() runs pretty fast."""
    def S_err(T_in, exp_in, S_in):
        """Deviation in entropy. (want S = S_in)"""
        exp_eq = deepcopy(exp_in)
        exp_eq.T = T_in
        exp_eq.find_outputs()
        return sum(exp_eq.Ss)- S_in

    rrst = root_scalar(S_err, args = (exp0, S_tar), x0 = exp0.T, method = "secant",
                       rtol = tol, maxiter = max_step) # outputs a root_result object
    if not rrst.converged: logging.warning("Algorithm failed to converge!")
    return rrst.root, rrst

def isentropic_canonical_solver(N_tot_tar: dict,
                      S_tar: float,
                      exp0: E9M.DoS_exp,
                      N_tot_fn = None,
                      max_step: int = 100,
                      tol: float = 1e-3) -> (float, RootResults):
    # Doesn't work - input is [T, *N], but output is error of [S, *N]
    """Find thermal equlibrium config, in particular T, given some total entropy
    and total particle number.
    
    In actual experiments, we are often given a fixed number of particle, and
    filling is whatever that results from that number.
    This is useful for simulating what would actually happen in experiments.
    Arguments:
        N_tot_tar: the number of particles for each particle type, expressed as e.g.:
               {"fermi1": 3000, "fermi2": 5000}.
        N_tot_fn: function that returns N_tot: N_tot_fn(exp0) = N_tot_now.
        exp0: in this function, Np of each species in exp0 is also considered as an
              initial guess."""
    def default_N_tot_fn(exp):
        """A N_tot_fn that should work in most cases."""
        N_tot = dict.fromkeys(N_tot_tar.keys(), 0.)
        for k in N_tot_tar.keys():
            for sp in exp.species_list: # add particles from all species of the given particle type
                if sp["name"] == k or sp["reservoir"] == k:
                    N_tot[k] += sp["Np"]
        return N_tot
    
    def NS_err(TN_in, exp_in, N_in, S_in, N_tot_fn):
        """Deviation in N_tot and S. (want N_tot = N_in and S = S_in)
        
        Arguments:
            TN_in: a list of [T_now, Np1_now, Np2_now, ...]"""
        exp_eq = deepcopy(exp_in)
        logging.debug("guess: T = {}, N = {}".format(TN_in[0], TN_in[1:]))
        exp_eq.T = TN_in[0]
        for k, i in enumerate(N_in.keys()):
            for sp in exp_eq.species_list:
                if sp["name"] == k:
                    sp["Np"] = TN_in[i + 1]
        exp_eq.find_outputs()
        N_out = N_tot_fn(exp_eq)
        S_err = abs(sum(exp_eq.Ss)- S_in)
        N_err = [N_out[k] - N_in[k] for k in N_in.keys()]
        logging.debug("S_err = {}, N_err = {}".format(S_err, N_err))
        return [S_err, *N_err]
    
    if N_tot_fn is None: N_tot_fn = default_N_tot_fn
    Np0 = [sp["Np"] for sp in exp0.species_list if sp["name"] in N_tot_tar.keys()]
    # Didn't work: hybr
    rrst = root(NS_err, args = (exp0, N_tot_tar, S_tar, N_tot_fn), x0 = [exp0.T, *Np0],
                method = "hybr") # outputs a root_result object
    if not rrst.success: logging.warning("Algorithm failed to converge!")
    return rrst.x, rrst