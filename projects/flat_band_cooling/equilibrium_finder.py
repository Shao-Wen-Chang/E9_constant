# Recommended import call: import equilibrium_finder as eqfind
import sys
from copy import deepcopy
import logging
import numpy as np
from scipy.optimize import root_scalar, root, RootResults, minimize
# User defined modules
sys.path.insert(1,
    "C:\\Users\\ken92\\Documents\\Studies\\E5\\simulation\\E9_simulations")
from E9_fn import util
import E9_fn.E9_models as E9M
# Algorithms for finding equilibrium conditions under different circumstances

#%% One API to rule them all
# TODO: probably want to rewrite the old solvers
def get_eqn_of_state_solver(from_vars: str, to_vars: str):
    """Get a solver function that solves the equation of state."""
    possible_EoSs = {
        "NVT":{
            "muVT": muVT_from_NVT_solver,
            "muVS": None,
            "NVS": None,
        },
        "NVS":{
            "muVT": muVT_from_NVS_solver,
            "muVS": None,
            "NVT":  NVT_from_NVS_solver,
        },
        "NVE":{
            "muVT": None,
        }
    }
    
    solver = possible_EoSs[from_vars][to_vars]
    if solver is None:
        raise NotImplementedError(
            f"Equation of state solver from {from_vars} to {to_vars} is not implemented yet.")
    return solver

def muVT_from_NVT_solver(N: float,
                         T: float,
                         E_orbs: np.ndarray) -> tuple[float, RootResults]:
    """Find chemical potential mu given N and T."""
    def N_err(mu):
        N_from_mu = sum(util.fermi_stat(E_orbs, T, mu))
        return abs(N - N_from_mu)
    
    # Get a reasonable guess for mu based on the number of particles and energy levels.
    mu_helper = np.linspace(E_orbs[0], E_orbs[-1], 1000)
    N_helper = np.array([sum(util.fermi_stat(E_orbs, T, mu)) for mu in mu_helper])
    mu_guess = mu_helper[abs(N_helper - N).argmin()]
    
    rrst = root_scalar(N_err, x0 = mu_guess, method = "secant")
    if not rrst.converged:
        logging.warning("muVT_from_NVT_solver failed to converge! Try loosening xtol")
        rrst = root_scalar(N_err, x0 = mu_guess, method = "secant", xtol = 1e-2)
        if not rrst.converged:
            logging.warning("muVT_from_NVT_solver still failed to converge at xtol = 1e-2")
    return rrst.root, rrst

def muVT_from_NVE_solver(N: float,
                         E: float,
                         E_orbs: np.ndarray,
                         T_guess: float = None) -> tuple[float, float, RootResults]:
    """Find a muVT system that give the right N and E.
    
    For each T, this function finds a mu such that the number of particles matches N,
    and then checks if the energy matches E.
    
    If this doesn't work, then try to change T such that mu is increased (decreased)
    for E too small (large).

    The algorithm makes use of the fact that both the particle number and total energy
    are monotonically increasing functions of mu at a given T. Assuming that for a fixed
    N, mu increases with T at large T

    Args:
        N: total number of particles in the system.
        E: total energy of the system.
        E_orbs: energy levels of the system.
        T_guess: initial guess for T. If None, it will be set to 1 or -1.
    """
    E_avg_T_infty = E_orbs.mean()
    E_avg_input = E / N
    if E_avg_input > E_orbs[-1] or E_avg_input < E_orbs[0]:
        logging.warning("Illegal average energy input")
        return None, None
    elif E_avg_input > E_avg_T_infty:
        logging.info("E / N is larger than the average energy at T = inf, try T < 0")
        T_guess = -abs(T_guess)
    elif E_avg_input < E_avg_T_infty:
        logging.debug("E / N is smaller than the average energy at T = inf, try T > 0")
        T_guess = abs(T_guess)
    
    def E_err(T):
        mu, _ = muVT_from_NVT_solver(N, T_guess, E_orbs)
        E_from_mu_and_T = sum(E_orbs * util.fermi_stat(E_orbs, T, mu))
        return abs(E - E_from_mu_and_T)
    
    rrst = root_scalar(E_err, x0 = T_guess, method = "secant")
    if not rrst.converged:
        logging.warning("muVT_from_NVE_solver failed to converge! Try loosening xtol")
        rrst = root_scalar(E_err, x0 = T_guess, method = "secant", xtol = 1e-2)
        if not rrst.converged:
            logging.warning("muVT_from_NVE_solver still failed to converge at xtol = 1e-2")
    
    T_out = rrst.root
    mu, _ = muVT_from_NVT_solver(N, T_out, E_orbs)
    return mu, T_out, rrst

def NVT_from_NVS_solver(S_tar: float,
                      exp0: E9M.NVT_exp,
                      max_step: int = 50,
                      tol: float = 1e-4) -> tuple[float, RootResults]:
    """Find thermal equlibrium config, in particular T, given some total entropy
    and target filling in systems.
    
    TODO: renamed from isentropic_fix_filling_solver 20250531, check if this is actually
          what it does
    See my personal notes on 2024/01/15 for some physical considerations. This function
    is useful for finding the target particle number and temperature / total entropy.
    Args:
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
        return sum(exp_eq.Ss) - S_in

    rrst = root_scalar(S_err, args = (exp0, S_tar), x0 = exp0.T, method = "secant",
                       rtol = tol, maxiter = max_step) # outputs a root_result object
    if not rrst.converged: logging.warning("Algorithm failed to converge!")
    return rrst.root, rrst

def muVT_from_NVS_solver(S_tar: float,
                       N_tar: float,
                       subregion_list: list[E9M.muVT_subregion],
                       T0: float,
                       mu0: float,
                       Tbounds: tuple = (0, 5),
                       mubounds: tuple = (0, 6),
                       method: str = "Nelder-Mead",
                       options_dict: dict = None) -> (np.ndarray[float], RootResults):
    """Solve for mu and T given S and N (V is held constant).
    
    Returns:
        rrst.root: an array of [T, mu] that will give the correct S and N.
        orst: the OptimizeResult object returned by root_scalar for full information."""
    # Assumes only one species for now; generalize if this works
    sp_name = subregion_list[0].species
    def err_fn(Tmu_guess):
        T_guess = Tmu_guess[0]
        mu_guess = Tmu_guess[1]
        exp_guess = E9M.muVT_exp(T_guess, subregion_list, {sp_name: mu_guess})
        S, T, N = exp_guess.S, exp_guess.T, exp_guess.N_dict[sp_name]
        return abs(S - S_tar) + abs(N - N_tar)

    orst = minimize(err_fn, x0 = [T0, mu0], bounds = [Tbounds, mubounds], method = method,
                    options = options_dict)
    if not orst.success: logging.warning("Algorithm failed to converge!")
    return orst.x, orst

#%% Method that don't work yet
def isentropic_fix_Ntot_solver(N_tot_tar: dict,
                    S_tar: float,
                    exp0: E9M.NVT_exp,
                    max_step_S: int = 50,
                    tol_S: float = 1e-4) -> (list[float], RootResults):
    # Many syntax errors
    """Find thermal equlibrium config, in particular T and *Np_sys, given some total entropy
    and total number of particles in (system + reservoirs).
    
    The algorithm is pretty rough:
        while (N_tot not close to N_target):
            guess a new filling in the system
            Use NVT_from_NVS_solver to find Np in reservoirs
            add Np in reservoirs and the sytem to find N_tot
    So it is basically two single-variable solvers stacked together.
    Args:
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
        """Use NVT_from_NVS_solver to find the new experient condition."""
        T_now, _ = NVT_from_NVS_solver(S_tar, exp_in, max_step_S, tol_S)
        exp_now = deepcopy(exp_in)
        exp_now.T = T_now
        exp_now.find_outputs()
        return exp_now
    
    def N_err(N_sys_guess, exp_in):
        """Given N_sys_guess, use update_exp to find the new N_tot, and calculate deviation from N_tot_tar.
        
        Returns: sum of |N_tot_tar - N_tot_now| for all species."""
        all_sp = N_tot_tar.keys()
        for sp_name in all_sp:
            for sp in exp_in.subregion_list:
                if sp["name"] == sp_name:
                    sp["Np"] = N_sys_guess[sp_name]
                    continue
            raise Exception("{} not found in the experiment".format(sp_name))
        exp_in = update_exp(exp_in)
        return sum([abs(N_tot_tar[k] - exp_in.find_N_tot()[k]) for k in all_sp])
    
    N_sys_init = []
    for k in N_tot_tar.keys():
        for sp in exp0.subregion_list:
            if sp["name"] == k:
                N_sys_init.append(sp["Np"])
    rrst = root(N_err, args = exp0, x0 = N_sys_init) # outputs a root_result object
    if not rrst.success: logging.warning("Algorithm failed to converge!")
    # Unfortunately for now I have to run NVT_from_NVS_solver again
    # to re-obtain T
    for i, k in enumerate(N_tot_tar.keys()):
        for sp in exp0:
            if sp["name"] == k:
                sp["Np"] = rrst.x[i]
    T_now, _ = NVT_from_NVS_solver(S_tar, exp0, max_step_S, tol_S)
    return np.hstack(T_now, rrst.x), rrst

def isentropic_canonical_solver(N_tot_tar: dict,
                      S_tar: float,
                      exp0: E9M.NVT_exp,
                      N_tot_fn = None,
                      max_step: int = 100,
                      tol: float = 1e-3) -> tuple[float, RootResults]:
    # Doesn't work - input is [T, *N], but output is error of [S, *N]
    """Find thermal equlibrium config, in particular T, given some total entropy
    and total particle number.
    
    In actual experiments, we are often given a fixed number of particle, and
    filling is whatever that results from that number.
    This is useful for simulating what would actually happen in experiments.
    Args:
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
            for sp in exp.subregion_list: # add particles from all species of the given particle type
                if sp["name"] == k or sp["reservoir"] == k:
                    N_tot[k] += sp["Np"]
        return N_tot
    
    def NS_err(TN_in, exp_in, N_in, S_in, N_tot_fn):
        """Deviation in N_tot and S. (want N_tot = N_in and S = S_in)
        
        Args:
            TN_in: a list of current guesses [T_now, Np1_now, Np2_now, ...]
        Returns: a list of [S_err, Np1_err, Np2_err, ...]"""
        exp_eq = deepcopy(exp_in)
        logging.debug("guess: T = {}, N = {}".format(TN_in[0], TN_in[1:]))
        exp_eq.T = TN_in[0]
        for k, i in enumerate(N_in.keys()):
            for sp in exp_eq.subregion_list:
                if sp["name"] == k:
                    sp["Np"] = TN_in[i + 1]
        exp_eq.find_outputs()
        N_out = N_tot_fn(exp_eq)
        S_err = abs(sum(exp_eq.Ss)- S_in)
        N_err = [N_out[k] - N_in[k] for k in N_in.keys()]
        logging.debug("S_err = {}, N_err = {}".format(S_err, N_err))
        return [S_err, *N_err]
    
    if N_tot_fn is None: N_tot_fn = default_N_tot_fn
    Np0 = [sp["Np"] for sp in exp0.subregion_list if sp["name"] in N_tot_tar.keys()]
    # Didn't work: hybr
    rrst = root(NS_err, x0 = [exp0.T, *Np0], args = (exp0, N_tot_tar, S_tar, N_tot_fn),
                method = "krylov") # outputs a root_result object
    if not rrst.success: logging.warning("Algorithm failed to converge!")
    return rrst.x, rrst