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
        max_step: number of steps to try before the algorithm is terminated
        tol: acceptable relative tolerance in the final entropy.
    Return:
        rrst.root: temperature of the equilibrated system.
        rrst: the RootResults object returned by root_scalar for full information.
        # exp_eq: equilibrium configuration that satisfies the initial condition.
    Since I am using root_scalar, only the temperature of the system is returned.
    This is a bit wasteful, but find_outputs() runs pretty fast."""
    def S_err(T_in, exp_in, S_in):
        """Deviation in entropy. (want S = S_in)
        
        T_in: current guess of T"""
        exp_eq = deepcopy(exp_in)
        exp_eq.T = T_in
        exp_eq.find_outputs()
        return sum(exp_eq.Ss)- S_in

    rrst = root_scalar(S_err, args = (exp0, S_tar), x0 = exp0.T, method = "secant",
                       rtol = tol, maxiter = max_step) # outputs a root_result object
    if not rrst.converged: logging.warning("Algorithm failed to converge!")
    return rrst.root, rrst

#%% Method that don't work yet
def isentropic_fix_Ntot_solver(N_tot_tar: dict,
                    S_tar: float,
                    exp0: E9M.DoS_exp,
                    max_step_S: int = 50,
                    tol_S: float = 1e-4) -> (list[float], RootResults):
    # Many syntax errors
    """Find thermal equlibrium config, in particular T and *Np_sys, given some total entropy
    and total number of particles in (system + reservoirs).
    
    The algorithm is pretty rough:
        while (N_tot not close to N_target):
            guess a new filling in the system
            Use isentropic_fix_filling_solver to find Np in reservoirs
            add Np in reservoirs and the sytem to find N_tot
    So it is basically two single-variable solvers stacked together.
    Arguments:
        S_tar: total entropy of the system.
        N_tot_tar: the number of particles for each particle type, expressed as e.g.:
                   {"fermi1": 3000, "fermi2": 5000}. Includes reserviors.
        exp0: DoS_exp that specifies initial conditions. This object is left
              unmodified. Some notes about the defining parameters:
              T: This is used as an initial guess for the solver. The final value of
                 T will be whatever gives the correct entropy.
        max_step_S: max_step passed to isentropic_fix_Ntot_solver.
        tol_S: tol passed to isentropic_fix_Ntot_solver.
    Return:
        rrst.root: a list of [T, Np1_sys, Np2,sys, ...].
        rrst: the RootResults object returned by root_scalar for full information."""
    def update_exp(exp_in):
        """Use isentropic_fix_filling_solver to find the new experient condition."""
        T_now, _ = isentropic_fix_filling_solver(S_tar, exp_in, max_step_S, tol_S)
        exp_now = deepcopy(exp_in)
        exp_now.T = T_now
        exp_now.find_outputs()
        return exp_now
    
    def N_err(N_sys_guess, exp_in):
        """Given N_sys_guess, use update_exp to find the new N_tot, and calculate deviation from N_tot_tar.
        
        Output: sum of |N_tot_tar - N_tot_now| for all species."""
        all_sp = N_tot_tar.keys()
        for sp_name in all_sp:
            for sp in exp_in.species_list:
                if sp["name"] == sp_name:
                    sp["Np"] = N_sys_guess[sp_name]
                    continue
            raise Exception("{} not found in the experiment".format(sp_name))
        exp_in = update_exp(exp_in)
        return sum([abs(N_tot_tar[k] - exp_in.find_N_tot()[k]) for k in all_sp])
    
    N_sys_init = []
    for k in N_tot_tar.keys():
        for sp in exp0.species_list:
            if sp["name"] == k:
                N_sys_init.append(sp["Np"])
    rrst = root(N_err, args = exp0, x0 = N_sys_init) # outputs a root_result object
    if not rrst.converged: logging.warning("Algorithm failed to converge!")
    # Unfortunately for now I have to run isentropic_fix_filling_solver again
    # to re-obtain T
    for i, k in enumerate(N_tot_tar.keys()):
        for sp in exp0:
            if sp["name"] == k:
                sp["Np"] = rrst.x[i]
    T_now, _ = isentropic_fix_filling_solver(S_tar, exp0, max_step_S, tol_S)
    return np.hstack(T_now, rrst.x), rrst

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
               {"fermi1": 3000, "fermi2": 5000}. Includes reserviors.
        N_tot_fn: function that returns N_tot: N_tot_fn(exp0) = N_tot_now.
        exp0: in this function, Np of each species in exp0 is also considered as an
              initial guess."""
    def default_N_tot_fn(exp):
        """A N_tot_fn that should work in most cases.
        
        For each species specified (as keys) in N_tot_tar, this function looks for that
        the species and reservoirs of that species, and add their total atom number."""
        N_tot = dict.fromkeys(N_tot_tar.keys(), 0.)
        for k in N_tot_tar.keys():
            for sp in exp.species_list: # add particles from all species of the given particle type
                if sp["name"] == k or sp["reservoir"] == k:
                    N_tot[k] += sp["Np"]
        return N_tot
    
    def NS_err(TN_in, exp_in, N_in, S_in, N_tot_fn):
        """Deviation in N_tot and S. (want N_tot = N_in and S = S_in)
        
        Arguments:
            TN_in: a list of current guesses [T_now, Np1_now, Np2_now, ...]
        Output: a list of [S_err, Np1_err, Np2_err, ...]"""
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
    rrst = root(NS_err, x0 = [exp0.T, *Np0], args = (exp0, N_tot_tar, S_tar, N_tot_fn),
                method = "krylov") # outputs a root_result object
    if not rrst.success: logging.warning("Algorithm failed to converge!")
    return rrst.x, rrst