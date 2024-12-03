# Recommended import call: import E9_fn.polarizabilities_calculation as E9pol
import numpy as np
import logging

import E9_fn.E9_constants as E9c
import E9_fn.E9_atom as E9a
import E9_fn.datasets.transition_line_data as TLData
from E9_fn import util

#%% functions
# TODO: add more comments, I can't read this
def alpha_pol(K, lamb_in, line_list, state, q = None, F = None, I = None):
    """[A2·s4·kg−1] Returns the (rank-K) polarizability given a list of lines, ignoring scattering.
    
    See [Axner04], esp. for terminologies in Appendix A. You should understand what I mean by: the result applies for
        - a "hf state" to a "hf state," when q, F and I are all given
        - a "hf state" to all E1-allowed "hf states" summed together, when F and I are given
        - a "fs state" to a "fs state," when q is given
        - a "fs state" to all E1-allowed "fs states" summed together, when q, F and I are not given
    Expressions for alphas are taken from [Le Kien13]. (I ignore the imaginary \gamma terms in the expression in that paper.)

    Args:
        K:          [dimless] K = 0/1/2, for scalar/vector/tensor light shifts, respectively.
        lamb_in:    [nm] A number or array of wavelength of incident light.
        line_list:  A list of LINES considered. They should have the same gs.
        state:      A string denoting what FINE STRUCTURE (LEVEL) is of interest.
        q:          Polarization of the photon, such that m' = m + q.
        F:          The hyperfine level of the state; contributes to a constant factor. This is only true if one
                    can ignore the coupling between different hyperfine states, i.e. when the Stark shift is small
                    compared to hyperfine splitting. (In particular, this can be violated when one works with very
                    very strong lattices used during microscope imaging.)
                    TODO: review the physics (i.e. when is HF relevant)
        I:          Required for hyperfine states calculations."""
    alpha = np.zeros_like(lamb_in)
    wl = 2 * np.pi * (E9c.c_light / lamb_in)
    for line in line_list:
        iso, gs, es = line['isotope'], line['gs'], line['es']
        line_name = iso + '_' + gs + '_' + es
        
        # Sorting out the relevant lines and grabbing line data
        lamb, f_ik = line['lambda'], line['f_ik'] # numbers used in calculation
        if f_ik is None:
            # No oscillator strength (although Einstein A-coefficient might be nonzero)
            logging.info(line_name + ' transition does not have f_ik data (not E1 allowed?)')
            continue
        
        if line['gs'] == state:
            J1, J2 = line['Jg'], line['Je']
            wa = 2 * np.pi * (E9c.c_light / lamb) # angular frequencie of atomic transitions
        elif line['es'] == state:
            J1, J2 = line['Je'], line['Jg']
            wa = - 2 * np.pi * (E9c.c_light / lamb) # (negaive if the state of concern is the excited state)
        else:
            logging.info(line_name + ' transition has no effect and is ignored')
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

    return prefactor * alpha

def I2uK_from_pol(I_in, alpha_pol):
    """Find effective potential in uK given some polarizability."""
    # TODO: think about what I will usually want to use
    # these are copied from my old ipynb
    peak_E2 = 2 * I_in / E9c.c_light / E9c.epsilon_0
    return - peak_E2 * (alpha_pol / 4) / E9c.k_B * 1e6 

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
def alpha_s_K_4S1o2(lamb_in):
    """Scalar polarizability of K in the (find structuree) ground state manifold."""
    return alpha_pol(0, lamb_in, TLData.K_4S1o2_lines, '4S1o2', q = None, F = None, I = None)

def alpha_s_Rb_5S1o2(lamb_in):
    """Scalar polarizability of K in the (find structuree) ground state manifold."""
    return alpha_pol(0, lamb_in, TLData.Rb_D12_doublet, '5S1o2', q = None, F = None, I = None)