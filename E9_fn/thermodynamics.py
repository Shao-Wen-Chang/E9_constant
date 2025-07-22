import logging
import numpy as np
from scipy.linalg import eigh

from E9_fn import util
# simplest version (spinless fermions in s-bands of, say, a kagome lattice)
# Units: t_hubbard = 1

### Definitions ###
#%% Functions for generating the list of orbital energies E_orbs
def E_orbs_from_DoS(DoS, E_range, sample_num: int, bin_num: int = 500):
    """Given a density of state DoS, return a list of energies sampled from this DoS.
    
    Assumes continuous DoS. E_range is separated into bin_num bins, and energies in each
    bin are evenly spread out.
        DoS: callable; density of states. Normalization doesn't matter.
        E_range: (E_min, E_max); range of energies considered.
        sample_num: total number of points in E_range. Should be system size.
        bin_num: number of bins in E_range.
        
        E_orbs: ndarray; energies of orbitals."""
    # Condition the bins
    bin_E_ends = np.linspace(E_range[0], E_range[1], bin_num + 1)
    bin_weights = DoS(bin_E_ends[1:])
    bin_samples = (sample_num * (bin_weights / bin_weights.sum())).astype(int) # number of points in each bin
    rounding_error = bin_samples.sum() - sample_num
    if rounding_error < 0: # not enough points; add to the highest energy bin
        bin_samples[-1] += -rounding_error
    elif rounding_error > 0: # too many points; remove from highest energy bins
        ind = -1
        while rounding_error > 0:
            rounding_error = rounding_error - bin_samples[ind]
            bin_samples[ind] = max(0, -rounding_error)
            ind -= 1
    bin_samples_cum = bin_samples.cumsum()
    bin_ends = np.concatenate((np.array([0], dtype = int), bin_samples_cum))

    # generate E_orbs
    E_orbs = np.zeros(sample_num)
    for i in range(bin_num):
        i1, i2, E1, E2, s = bin_ends[i], bin_ends[i + 1], bin_E_ends[i], bin_E_ends[i + 1], bin_samples[i]
        E_orbs[i1:i2] = np.linspace(E1, E2, s, endpoint = False)
    return E_orbs

def E_orbs_with_deg(DoS, E_range, sample_num: int, dgn_list: list[tuple] = [], bin_num: int = 500):
    """Generate E_orbs with a list of degenaracies added to the dispersive E_orbs.
    
    sample_num is the number of orbitals sampled in DoS, so the total number of states will be
    sample_num + sum([dl[1] for dl in dgn_list])
    Args:
        dgn_list: a list of tuple(energy: float, num_of_degenerate_orbitals: int)."""
    E_orbs = E_orbs_from_DoS(DoS, E_range, sample_num, bin_num = bin_num)
    for dgn in dgn_list: 
        E_orbs = np.hstack((E_orbs, np.ones(dgn[1]) * dgn[0]))
    return E_orbs

#%% Find thermodynamic values
def find_Np(E_orbs, T, mu, xi) -> float:
    """(Still returning float!) Find the number of (non-condensed) particle of a system.
    
    For fermions, this is the total number of particles in the system. For bosons, this
    is the total number of particles less the fraction that forms a BEC.
        T: (fundamental) temperature (i.e. multiplied by k_B)
        xi: 1 for fermions, -1 for bosons"""
    return np.sum(util.part_stat(E_orbs, T, mu, xi, replace_inf = 0))

def find_E(E_orbs, T, mu, xi, N_BEC: int = 0):
    """Find the total energy of a system.
    
        N_BEC: number of bose-condensed particles, if any (should be 0 for fermions)"""
    return sum(E_orbs * util.part_stat(E_orbs, T, mu, xi, replace_inf = 0)) + N_BEC * E_orbs[0]

# TODO: This should be replaced by muVT_from_NVT_solver
def find_mu(E_orbs, T, Np, xi, max_step: int = 10000, tolerance = 1e-6):
    """Find the chemical potential $\mu$ of a system, and (if any) the number of bose-
    condensed particles.
    
    $\mu$ is chosen such that N comes out right. (I could have also just used an integral
    solver I guess) Remember that "BEC only occurs in 2D at T = 0" refers to 2D free
    particles. It is all about the leading power term of the DoS, and e.g. 2D harmonic
    confinement can result in BEC at T > 0.
    Args:
        Np: number of (fermionic) particles (of a single spin species)
        max_setp: number of steps to try before the algorithm is terminated
        tolerance: acceptable fractional error on N.
    Returns:
        mu: chemical potential such that error to Np is less than the tolerance.
        N_BEC: 0 if fermion or non-condensed bosons"""
    def mu_subroutine(mu_min, mu_max, Np, xi, tolerance):
        """Subroutine used in the algorithm.
        
        The algorithm finds mu by reducing the possible range of mu by a factor of 2
        for each iteration. If mu is within tolerance, both mu_min and mu_max is set
        to this acceptable mu. (so mu_min == mu_max signals termination of algorithm)"""
        mu = (mu_min + mu_max) / 2
        N_mu = find_Np(E_orbs, T, mu, xi)
        N_err = N_mu - Np

        if abs(N_err) < Np * tolerance: # good enough
            mu_min, mu_max = mu, mu
        elif N_err >= 0:
            mu_max = mu
        else:
            mu_min = mu
        
        return mu_min, mu_max

    E_min, E_max = min(E_orbs), max(E_orbs)
    E_range = E_max - E_min
    mu_min, mu_max = E_min - 8 * E_range, E_max + 8 * E_range
    N_BEC = 0

    # For bosons, first check if the gas is bose-condensed; otherwise the procedure for
    # finding mu will be the same as Fermions, except that mu_max < E_orbs[0] must hold
    if xi == -1:
        # The ground state is excluded from the sum for ease of calculation
        N_ex = find_Np(E_orbs[1:], T, E_orbs[0], xi)
        if N_ex < Np:
            mu_min, mu_max = E_orbs[0], E_orbs[0]
            N_BEC = Np - N_ex
        else:
            mu_max = E_orbs[0]
    
    # For fermions or non-condesed bosons; mu = mu_min = mu_max if the algorithm is successful
    if N_BEC == 0:
        for _ in range(max_step):
            mu_min, mu_max = mu_subroutine(mu_min, mu_max, Np, xi, tolerance)
            if mu_min == mu_max: break

        if mu_min != mu_max:
            logging.warning("Error in particle number is larger than the tolerance")

    return mu_min, N_BEC

def find_S(E_orbs, T, Np, xi, mu = None, E_total = None, N_BEC: int = 0):
    """Find the fundamental entropy \sigma = S/k_B of a fermionic system.
    
    Although we use grand canonical ensemble for the analytical expression, we actually
    back out \mu from Np. If \mu is not given, then find_mu will be used to find \mu
    """
    if mu is None: mu = find_mu(E_orbs, T, Np, xi, N_BEC)
    if E_total is None: E_total = find_E(E_orbs, T, mu, xi, N_BEC)

    if N_BEC != 0:
        # Intentially coded in a way that assumes mu == E_orbs[0]; if result is
        # still inf then the calculation is wrong
        logging.info("N_BEC = {:.2f}; don't include the ground state in log".format(N_BEC))
        return (E_total - mu * Np) / T + xi * np.log(1 + xi * np.exp((mu - E_orbs[1:]) / T)).sum()
    else:
        return (E_total - mu * Np) / T + xi * np.log(1 + xi * np.exp((mu - E_orbs) / T)).sum()

def find_SvN(rho: np.ndarray):
    """Find the von Neumann entropy of a given density matrix (or the eigenvalues of it).
    
    If rho is 1-dimensional, it is assumed to be the eigenvalues of some density matrix.
    """
    rho_diag = rho
    if rho.ndim > 2:
        raise(Exception("The dimension of input ndarray must be 1 or 2"))
    elif rho.ndim == 2:
        if not util.IsHermitian(rho):
            raise(Exception("The input density matrix is not Hermitian!"))
        else:
            eigvals, _ = eigh(rho)
            rho_diag = eigvals.diagonal()
    if not np.allclose(rho_diag.sum(), 1):
        logging.warning("The trace of the input density matrix is not 1!")
    
    return -(rho_diag * np.log(rho_diag)).sum()

#%% Simulation
if __name__ == "__main__":
    # removed old incompatible code on 20240114
    pass