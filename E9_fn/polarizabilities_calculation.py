# Recommended import call: import E9_fn.polarizabilities_calculation as E9pol
import numpy as np
import logging
import scipy.constants as sc
from arc import DynamicPolarizability
from arc.wigner import Wigner6j

import E9_fn.E9_constants as E9c
import E9_fn.E9_atom as E9a
import E9_fn.datasets.transition_line_data as TLData
from E9_fn import util

#%% functions
# TODO: add more comments, I can't read this
def alpha_pol_manual(K, lamb_in, line_list, state, q = None, F = None, I = None, log_fn = logging.info):
    """[A2·s4·kg−1] Returns the (rank-K) polarizability given a list of lines, ignoring scattering.
    
    See [Axner04], esp. for terminologies in Appendix A. You should understand what I mean by: the result applies for
        - a "hf state" to a "hf state," when q, F and I are all given
        - a "hf state" to all E1-allowed "hf states" summed together, when F and I are given
        - a "fs state" to a "fs state," when q is given
        - a "fs state" to all E1-allowed "fs states" summed together, when q, F and I are not given
    Expressions for alphas are taken from [Le Kien13]. (I ignore the imaginary \gamma terms in the expression in that paper.)

    Args:
        K:          [dimless] K = 0/1/2, for scalar/vector/tensor light shifts, respectively.
        lamb_in:    [m] A number or array of wavelength of incident light.
        line_list:  A list of LINES considered. They should have the same gs.
        state:      A string denoting what FINE STRUCTURE (LEVEL) is of interest.
        q:          Polarization of the photon, such that m' = m + q.
        F:          The hyperfine level of the state; contributes to a constant factor. This is only true if one
                    can ignore the coupling between different hyperfine states, i.e. when the Stark shift is small
                    compared to hyperfine splitting. (In particular, this can be violated when one works with very
                    very strong lattices used during microscope imaging.)
                    TODO: review the physics (i.e. when is HF relevant)
        I:          Required for hyperfine states calculations.
        log_fn:     logging function used when handling line date.
    """
    is_scalar = np.isscalar(lamb_in)
    lamb_arr = np.atleast_1d(lamb_in)
    alpha = np.zeros_like(lamb_arr)
    wl = 2 * np.pi * (E9c.c_light / lamb_arr)
    for line in line_list:
        iso, gs, es = line['isotope'], line['gs'], line['es']
        line_name = iso + '_' + gs + '_' + es
        
        # Sorting out the relevant lines and grabbing line data
        lamb, f_ik = line['lambda'], line['f_ik'] # numbers used in calculation
        if f_ik is None:
            # No oscillator strength (although Einstein A-coefficient might be nonzero)
            log_fn(line_name + ' transition does not have f_ik data (not E1 allowed?)')
            continue
        
        if line['gs'] == state:
            J1, J2 = line['Jg'], line['Je']
            wa = 2 * np.pi * (E9c.c_light / lamb) # angular frequencie of atomic transitions
        elif line['es'] == state:
            J1, J2 = line['Je'], line['Jg']
            wa = - 2 * np.pi * (E9c.c_light / lamb) # (negaive if the state of concern is the excited state)
        else:
            log_fn(line_name + ' transition has no effect and is ignored')
            continue
        
        # Actual calculation
        # This is actually just S_ik in S.I.
        mat_ele_sqr = f_ik * (2 * line['Jg'] + 1) * (3 * E9c.hbar * E9c.e_ele**2) / (2 * E9c.m_e * abs(wa)) # reduced matrix element squared
        if F is None:
            alpha += (-1)**(K + J1 + J2 + 1) * np.sqrt(2*K + 1) * float(util.wigner_6j_safe(1, K, 1, J1, J2, J1)) * \
                     mat_ele_sqr * (1 / (wa - wl) + (-1)**K / (wa + wl)) / E9c.hbar
        else:
            F2min, F2max = max(abs(J2 - I), F - 1), min(abs(J2 + I), F + 1)
            logging.debug(line_name + ': F = {}, F\' = {} ~ {}'.format(F, F2min, F2max))
            for F2 in np.arange(F2min, F2max + 1):
                alpha += (-1)**(K + F + F2 + 1) * (2*F + 1) * (2*F2 + 1) * np.sqrt(2*K + 1) * mat_ele_sqr * \
                        float(util.wigner_6j_safe(1, K, 1, F, F2, F)) * float(util.wigner_6j_safe(F, 1, F2, J2, I, J1))**2 * \
                        (1 / (wa - wl) + (-1)**K / (wa + wl)) / E9c.hbar
    
    if F is None:
        G = J1
    else:
        G = F
        
    if K == 0:
        prefactor = 1 / np.sqrt(3 * (2 * G + 1))
    elif K == 1:
        prefactor = - np.sqrt(2 * G / ((G + 1) * (2 * G + 1)))
    elif K == 2:
        prefactor = - np.sqrt(2 * G * (2 * G - 1) / (3 * (G + 1) * (2 * G + 1) * (2 * G + 3)))

    res = prefactor * alpha
    return res[0] if is_scalar else res

def alpha_pol_arc(K, lamb_in, arc_atom, n, l, j, F = None, I = None, include_core = True, n_max = 30):
    """[A2·s4·kg−1] Returns the (rank-K) polarizability using ARC.
    
    This function is vectorized over lamb_in for performance.
    
    Args:
        K:              [dimless] K = 0/1/2, for scalar/vector/tensor light shifts, respectively.
        lamb_in:        [m] A number or array of wavelength of incident light.
        arc_atom:       ARC atom instance.
        n:              Principal quantum number of the state.
        l:              Orbital angular momentum of the state.
        j:              Total angular momentum of the state.
        F:              The hyperfine level of the state. (Currently ignored for ARC)
        I:              Required for hyperfine states calculations. (Currently ignored for ARC)
        include_core:   Whether to include core polarizability.
        n_max:          Max principal quantum number for the basis.
    """
    is_scalar = np.isscalar(lamb_in)
    lamb_arr = np.atleast_1d(lamb_in)
    
    dp = DynamicPolarizability(arc_atom, n, l, j)
    dp.defineBasis(arc_atom.groundStateN, n_max)
    
    # Pre-calculate matrix elements and energies for all states in basis
    initialLevelEnergy = dp.atom.getEnergy(dp.n, dp.l, dp.j, s=dp.s) * sc.e
    
    basis = dp.basis
    mes = []
    energies = []
    j1s = []
    
    for state in basis:
        n1, l1, j1 = state[:3]
        # Dipole transition selection rules: |j - j1| <= 1 and |l - l1| = 1
        if abs(j1 - dp.j) < 1.1 and (abs(l1 - dp.l) > 0.5 and abs(l1 - dp.l) < 1.1):
            mes.append(dp.atom.getReducedMatrixElementJ(dp.n, dp.l, dp.j, n1, l1, j1, s=dp.s))
            energies.append(dp.atom.getEnergy(n1, l1, j1, s=dp.s) * sc.e)
            j1s.append(j1)
            
    mes = np.array(mes)
    energies = np.array(energies)
    j1s = np.array(j1s)
    
    transitionEnergies = energies - initialLevelEnergy
    
    # Vectorize over wavelengths
    # driveEnergies [J]
    driveEnergies = sc.h * sc.c / lamb_arr
    
    # Prefactors and constants for polarizability components (consistent with ARC implementation)
    prefactor1 = 1.0 / ((dp.j + 1) * (2 * dp.j + 1))
    prefactor2 = (6 * dp.j * (2 * dp.j - 1) / (6 * (dp.j + 1) * (2 * dp.j + 1) * (2 * dp.j + 3))) ** 0.5
    bohr_radius = sc.physical_constants["Bohr radius"][0]
    const_factor = (sc.e * bohr_radius) ** 2
    
    # Use broadcasting for efficient summation: (N_basis, 1) and (1, N_wl)
    TE = transitionEnergies[:, np.newaxis]
    DE = driveEnergies[np.newaxis, :]
    ME2 = (mes**2)[:, np.newaxis]
    
    denom = TE**2 - DE**2
    
    # Scalar Polarizability (rank 0)
    d_scalar = ME2 * const_factor * TE / denom
    alpha0 = 2.0 * np.sum(d_scalar, axis=0) / (3.0 * (2.0 * dp.j + 1.0))
    alpha0 /= sc.h # Convert to Hz m^2 / V^2
    
    # Vector Polarizability (rank 1)
    j_term = (dp.j * (dp.j + 1) + 2 - j1s * (j1s + 1))[:, np.newaxis]
    d_vector = (-1.0) * j_term * ME2 * const_factor * DE / denom
    alpha1 = prefactor1 * np.sum(d_vector, axis=0) / sc.h
    
    # Tensor Polarizability (rank 2)
    alpha2 = np.zeros_like(alpha0)
    if dp.j > 0.6:
        w6j = np.array([float(Wigner6j(dp.j, 1, j1, 1, dp.j, 2)) for j1 in j1s])[:, np.newaxis]
        phases = (-1.0)**np.round(dp.j + j1s + 1)[:, np.newaxis]
        d_tensor = phases * ME2 * const_factor * w6j * TE / denom
        alpha2 = -4.0 * prefactor2 * np.sum(d_tensor, axis=0) / sc.h
        
    # Static Core Polarizability
    alphaC = dp.atom.alphaC * 2.48832e-8 # Conversion factor to SI (Hz m^2 / V^2)
    
    if K == 0: val = alpha0 + (alphaC if include_core else 0)
    elif K == 1: val = alpha1
    elif K == 2: val = alpha2
    else: raise ValueError("K must be 0, 1, or 2")
    
    res = sc.h * val
    return res[0] if is_scalar else res

def get_polarizability(lamb_in, K, method="ARC", **kwargs):
    """Unified wrapper for polarizability calculations."""
    if method == "ARC":
        return alpha_pol_arc(K, lamb_in, **kwargs)
    else:
        return alpha_pol_manual(K, lamb_in, **kwargs)

# Backward compatibility alias
alpha_pol = alpha_pol_manual

def _I2depth_from_pol(I_in, alpha_pol, unit):
    """Find the effective potential depth in some specified unit given some polarizability."""
    if unit == "uK":
        factor = 1 / E9c.k_B * 1e6
    elif unit == "J":
        factor = 1.
    peak_E2 = 2 * I_in / E9c.c_light / E9c.epsilon_0    # square of the electric field
    return - peak_E2 * (alpha_pol / 4) * factor

def I2uK_from_pol(I_in, alpha_pol):
    return _I2depth_from_pol(I_in, alpha_pol, "uK")

def I2J_from_pol(I_in, alpha_pol):
    return _I2depth_from_pol(I_in, alpha_pol, "J")

def C_av2B(hfs, ul = np.array([1, 0, 0])):
    """Calculate the conversion factor for effective B field (at E = 1 V/m).
    
    See [Le Kien13] eqn.(21). This can be treated as a magnetic field in almost every sense.
    B_eff = av2Beff * av * |E|**2
    Make sure that the quantization axis used here is consistent with the calculation in which this function is used.

    Args:
        hfs:    a HyperfineState object to get various quantities from
        ul:     polarization of light in spherical basis, (A_{-1}, A_0, A_1) = (\sigma_-, \pi, \sigma_+)."""
    u = util.Normalize(ul)
    pol_fac = abs(ul[0])**2 - abs(ul[2])**2 # this is the i(u* x u) term with all the algebra performed
    return pol_fac / (8 * E9c.mu_B * hfs.gF * hfs.F)

#%% Convenience functions
# For very far detuned light with reasonably small power, the following functions should be enough
# set log_fn to info if the line data is modified to check
def alpha_s_K_4S1o2(lamb_in):
    """Scalar polarizability of K in the (fine structure) ground state manifold using ARC."""
    return get_polarizability(lamb_in, 0, method="ARC", arc_atom=E9c.K40, n=4, l=0, j=0.5)

def alpha_s_Rb_5S1o2(lamb_in):
    """Scalar polarizability of Rb in the (fine structure) ground state manifold using ARC."""
    return get_polarizability(lamb_in, 0, method="ARC", arc_atom=E9c.Rb87, n=5, l=0, j=0.5)