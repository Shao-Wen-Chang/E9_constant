import numpy as np

from E9_fn import util
# simplest version (spinless fermions in s-bands of, say, a kagome lattice)
# Units: t_hubbard = 1

### Definitions ###
#%% Functions for generating the list of orbital energies E_orbs
def E_orbs_from_DoS(DoS, E_range, sample_num: int, bin_num: int = 500):
    '''Given a density of state DoS, return a list of energies sampled from this DoS.
    
    Assumes continuous DoS. E_range is separated into bin_num bins, and energies in each
    bin are evenly spread out.
        DoS: callable; density of states. Normalization doesn't matter.
        E_range: (E_min, E_max); range of energies considered.
        sample_num: total number of points in E_range. Should be system size.
        bin_num: number of bins in E_range.
        
        E_orbs: ndarray; energies of orbitals.'''
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

#%% Find thermodynamic values
def find_Np(E_orbs, T, mu, stat: str):
    '''Find the number of (non-condensed) particle of a system.
    
    For fermions, this is the total number of particles in the system. For bosons, this
    is the total number of particles less the fraction that forms a BEC.
        T: (fundamental) temperature (i.e. multiplied by k_B)
        stat: "fermi" or "bose"'''
    if stat == "fermi":
        return np.sum(util.fermi_stat(E_orbs, T, mu))
    elif stat == "bose":
        return np.sum(util.bose_stat(E_orbs, T, mu))

def find_E(E_orbs, T, mu, stat: str, N_BEC: int = 0):
    '''Find the total energy of a system.
    
        N_BEC: number of bose-condensed particles, if any'''
    if stat == "fermi":
        return sum(E_orbs * util.fermi_stat(E_orbs, T, mu))
    elif stat == "bose":
        return sum(E_orbs * util.fermi_stat(E_orbs, T, mu)) + N_BEC * E_orbs[0]

def find_mu(E_orbs, T, Np, stat: str = "fermi", max_step: int = 10000, tolerance = 1e-6):
    '''Find the chemical potential $\mu$ of a system, and (if any) the number of bose-
    condensed particles.
    
    $\mu$ is chosen such that N comes out right. (I could have also just used an integral
    solver I guess) Remember that "BEC only occurs in 2D at T = 0" refers to 2D free
    particles. It is all about the leading power term of the DoS, and e.g. 2D harmonic
    confinement can result in BEC at T > 0.
    Arguments:
        Np: number of (fermionic) particles (of a single spin species)
        max_setp: number of steps to try before the algorithm is terminated
        tolerance: acceptable fractional error on N.
    Outputs:
        mu: chemical potential such that error to Np is less than the tolerance.
        N_BEC: 0 if fermion or non-condensed bosons'''
    def mu_subroutine(mu_min, mu_max, Np, stat, tolerance):
        '''Subroutine used in the algorithm.
        
        The algorithm finds mu by reducing the possible range of mu by a factor of 2
        for each iteration. If mu is within tolerance, both mu_min and mu_max is set
        to this acceptable mu. (so mu_min == mu_max signals termination of algorithm)'''
        mu = (mu_min + mu_max) / 2
        N_mu = find_Np(E_orbs, T, mu, stat)
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
    if stat == "bose":
        N_ex = find_Np(E_orbs, T, E_orbs[0], stat)
        if N_ex < Np:
            mu_min, mu_max = E_orbs[0], E_orbs[0]
            N_BEC = Np - N_ex
        else:
            mu_max = E_orbs[0]
    
    # For fermions or non-condesed bosons; mu = mu_min = mu_max if the algorithm is successful
    if N_BEC == 0:
        for _ in range(max_step):
            mu_min, mu_max = mu_subroutine(mu_min, mu_max, Np, stat, tolerance)
            if mu_min == mu_max: break

        if mu_min != mu_max:
            print("Warning: error in particle number is larger than the tolerance")

    return mu_min, N_BEC

def find_S(E_orbs, T, Np, mu = None, E_total = None, stat: str = "fermi", N_BEC: int = 0):
    '''Find the fundamental entropy \sigma = S/k_B of a fermionic system.
    
    Although we use grand canonical ensemble for the analytical expression, we actually
    back out \mu from Np. If \mu is not given, then find_mu will be used to find \mu'''
    if mu is None: mu = find_mu(E_orbs, T, Np, stat, N_BEC)
    if E_total is None: E_total = find_E(E_orbs, T, mu, stat, N_BEC)
    
    if stat == "fermi":
        xi = 1.
    elif stat == "bose":
        xi = -1.
    
    return (E_total - mu * Np) / T + xi * np.log(1 + xi * np.exp((mu - E_orbs) / T)).sum()

#%% Simulation
if __name__ == "__main__":
    # removed old incompatible code on 20240114
    pass