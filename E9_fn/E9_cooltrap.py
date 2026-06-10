# Recommended import call: import E9_fn.E9_cooltrap as E9ct
from E9_fn import util
import E9_fn.E9_constants as E9c

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
# See E0_constants for list of references and some file conventions
# All from [PvdS] unless otherwise noted
# The arguments in functions have the following units and meaning unless
# otherwise noted:
#   delta:  [Hz]        detuning
#   gamma:  [Hz]        transition linewidth
#   m:      [kg]        mass of a single atom
#   n:      [1/m^3]     density
#   N:      [#]         number of atoms
#   v:      [m/s]       velocity (of some content-dependent thing)
#   s0:     [dimless]   saturation parameter; s0 := I/I_sat
#   T:      [K]         temperature
#   wbar:   [Hz]        (average) trap frequency

#%% Two-level system (TLS) physics
def Gamma_scat(s0, delta, gamma):
    """[Hz] Returns the photon scattering rate as calculated by TLS optical Bloch equation.
    
    This is \gamma_p in [PvdS] eqn 2.26.
    """
    return (gamma / 2) * s0 / (1 + s0 + (2 * delta / gamma)**2)

#%% Laser cooling related
def F_molasses(v, k, s0, delta, gamma, z = 0, Ag = 0):
    """[N] Returns the velocity dependent force in an optical molasses.
    
    Using (hbar * k * gamma) for the unit of force and (gamma * k) for velocities is convenient for plots.
    When both v and z are arrays they should have the same dimensions, and the two arrays are cycled at the same time.
        k: [1/m] wavevector; k := 2 * pi / lambda
        delta: [Hz] detuning (just from laser)
        z: [m] distance from magnetic field zero
        Ag: [T/m] Magnetic field gradient times g (see kappa_MOT)"""
    return E9c.hbar * k * (Gamma_scat(s0, delta - v * k + z * Ag, gamma) - Gamma_scat(s0, delta + v * k, gamma))

def beta_molasses(k, s0, delta, gamma):
    """[N/(m/s)] Returns the damping coefficient in an optical molasses.
    
    This is [PvdS] eqn 7.2, which is valid for small velocities (F = -\beta v). Use F_molasses for arbitrary velocities.
    Check F_molasses for units.
    Some values of interest:
        (damping rate) beta / m"""
    return - 8 * E9c.hbar * k**2 * delta * s0 / (gamma * (1 + s0 + (2 * delta / gamma)**2)**2)

#%% Trapping related
### Non-collisional properties (atom density etc.)
# MOT
def kappa_MOT(A, k, s0, delta, gamma, g = 1):
    """[N/m] Returns the restoring force coefficient in a MOT.
    
    Check F_molasses for other inputs.
        A: [T/m] magnetic field gradient (= 0.01 * (gradient in [G/cm]))
        g: [dimless] Effective "g-factor," given by (g_e * m_e - g_g * m_g). This is often around 1. ([Foot], but not sure why;
                     probably has something to do with using the stretched states)
    Some values of interest:
        (harmonic oscillator frequency) np.sqrt(kappa / m)
        (MOT rms size) np.sqrt(k_B * T / kappa)
        """
    return g * E9c.mu_B * A * beta_molasses(k, s0, delta, gamma) / (E9c.hbar * k)

# Linear potential (excluding MOT)
def n_peak_lin(N, V_grad, T):
    """[1/m^3] Returns the peak density in a linear trap.
    
    This is probably a magnetic trap. It is assumed that Vx' = Vy' = (1/2) * Vz' = V_grad.
        V_grad:    [J/m]   Gradient of the potential in the xy plane.
    """
    return 2 * N * V_grad**3 / (2 * E9c.k_B * T)**3

def r0_thermal_lin(V_grad, T):
    """[m] Returns the cloud radius for a non-interacting thermal gas in a linear trap.
    
    Note that this is the value at which the density decreases to 1/e of the maximum. Multiply bu 0.9 to get
    an effective gaussian width (obtained by naively fitting to a Gaussian; see fit_gaussian_to_shit).
    """
    return E9c.k_B * T / V_grad

# Harmonic potential
def n_peak_har(N, wbar, T, m):
    """[1/m^3] Returns the peak density in a harmonic trap.
    
    Check BEC (2.39). Note that r0_thermal_har(wbar, T, m) is not "R_bar" .
    This formula assumes a thermal gas, i.e. density is proportional to exp(-V / (k_B * T)).
    Other formulae entail similarly.
    """
    return N / np.pi**(3/2) / r0_thermal_har(wbar, T, m)**3

def n_peak_har_TF(N, wbar, m, a_s):
    U0 = 4 * np.pi * E9c.hbar**2 * a_s / m
    abar = np.sqrt(E9c.hbar / m / wbar)
    mu = 15**(2/5) / 2 * (N * a_s / abar)**(2/5) * E9c.hbar * wbar
    return mu / U0

def PSD_har(N, wbar, T, m):
    return n_peak_har(N, wbar, T, m) * util.lambda_de_broglie(m, T)**3

def r0_thermal_har(w0, T, m):
    """[m] Returns the cloud radius for a non-interacting thermal gas in a harmonic trap.
    
    Args:
        w0: trap ANGULAR frequency in the axis of interest."""
    return np.sqrt(2 * E9c.k_B * T / m / w0**2)

def wr_TOP(B_grad_q, B_bias, mu, m):
    """[rad * Hz] Returns the trap frequency of a TOP trap.
    
    This is the radial direction. wz is sqrt(8) times stronger, and the geometric average is sqrt(2) times stronger.
    
    Args:
        B_grad_q:   quadrupole field gradient in the radial direction.
        B_bias:     TOP bias field in the radial direction, assuming a circular orbit."""
    return np.sqrt(mu * B_grad_q**2 / (2 * B_bias * m))

# arbitrary potential
def n_thermal_norm1_from_V(V, T):
    """[1/m^3] Returns the density in an arbitrary trap, normalized to one particle.
    
    The exact cloud shape is calculated numerically using n(r) ~ exp{[-V(r) / (k_B * T)]}. One needs to also
    know the spatial resolution of V to get the actual density.
    """
    n_not_norm = np.exp(- V / (E9c.k_B * T))
    return n_not_norm / np.sum(n_not_norm)

import E9_fn.polarizabilities_calculation as E9pol

# Optical dipole traps
def V0_from_I(Gamma, nu, fl, I, gF, mF, P_pol = 0, Delta_FS = 0):
    """[J] Gives the trap depth for a hyperfine state (hfs).
    
    Legacy approximation. See [Grimm99] eqn.20. 
    This works for the large detuning (>> fine structure splitting of relevant excited states) limit.
        Gamma: [rad/s] (average) linewidth of relevant excited states. Usually the 2P3/2 and 2P1/2 states.
        nu: [Hz] (average) excited state energy in frequency. Note that w0 = 2 * pi * nu.
        fl: [Hz] frequency of light field. Note that wl = 2 * pi * fl.
        I: [W/m^2] light intensity.
        gF: gF of the (ground) state of trapped atoms.
        mF: mF value of the atom.
        P_pol: Polarization factor. 0 if linear or ignored, +/-1 if sigma+/- polarized
        Delta_FS: [Hz] difference between the two excited state. Note that [Grimm99] use angular frequency [rad/s]."""
    w0 = 2 * np.pi * nu
    wl = 2 * np.pi * fl
    Delta = wl - w0
    Delta_FS = 2 * np.pi * Delta_FS
    return (3 * np.pi * E9c.c_light**2 / 2 / w0**3) * (Gamma / Delta) * (1 + (P_pol * gF * mF / 3) * (Delta_FS / Delta)) * I

def V0_from_I_arc(I, fl, arc_atom, n, l, j, K=0, method="ARC", **kwargs):
    """[J] Gives the exact trap depth using ARC-backed polarizabilities.
    
    Args:
        I:          [W/m^2] Light intensity.
        fl:         [Hz] Light frequency.
        arc_atom:   ARC atom instance.
        n, l, j:    Atomic state.
        K:          Rank of polarizability (0=scalar, 1=vector, 2=tensor).
    """
    lamb_in = E9c.c_light / fl
    alpha = E9pol.get_polarizability(lamb_in, K, method=method, arc_atom=arc_atom, n=n, l=l, j=j, **kwargs)
    return E9pol.I2J_from_pol(I, alpha)

def U_from_Vlat(Vlat, a_s, k_L):
    """Returns U for some lattice parameters under harmonic well assumption.
    
    See e.g. [Tarruell18] eqn.6 or [Bloch08] eqn.(49). Both U and Vlat are in units of photon recoil energy. For a triangular
    lattice, there is an additional factor of 8/9 for potential well depth, and an (approximate) factor of sqrt(3)/2 to
    account for potential well size. (check the factor of 9/16)"""
    return np.sqrt(8 / np.pi) * (k_L * np.sqrt(3)/2) * a_s * ((8/9) * Vlat * (9/16))**(3/4)

def tUFromJx(J, x):
    """Given desired J := 4t^2/U and x := t/U, returns the value of t and U required.
    
    (x is not a commonly used notation in the community.)
    Fermi-Hubbard model is defined in terms of t (hopping integral) and U (on-site interaction), but the phase space
    is oftenmore conveniently expressed in other parameters. For example, the exact half-filling case is normally
    plotted for T (temperature) and x, and the T = 0 doped case J (spin superexchange) and delta (doping factor).
    In this sense it might be more natural to think about the relevant physics in terms of (T, x, J, delta), where
    T and delta are themselves free experimental parameters already."""
    return (J / (4 * x), J / (4 * x**2)) # (t, U)

def VaFortU(t, U, FBres, Vvst, VvsU):
    """(tentative) Generates a table that lists possible values of a and V resulting in desired t and U.
    
    This is useful when e.g. the phase diagram is actually dependent on also either t or U, or when some none-Fermi-
    Hubbard things are relevant, e.g. setting the time scale of evolution (with t I guess).
        FBres: a FeshbachResonance object that is used to calculate a (scattering length).
        Vvst and VvsU: relation between V (lattice depth) and t / U. Should probably be (2, n) arrays, where [0, i]
            are lattice depths, and [1, i] are corresponding t / U.
    Might not implement because it would need some work, but want to remind myself of possible tradeoffs."""
    pass

def lamb_dicke_const(V0, a_lat = E9c.a_sw_tri):
    """(tentative) Returns the Lamb-Dicke constant for a given 532 lattice depth.
    
    This currently assumes using K D1 line cooling in a 532 triangular lattice and can be easily generalized.
        V0:     [kHz] 532 lattice depth
        a_lat:  [m] lattice constant"""
    return (E9c.hbar * np.pi * a_lat**2 / 2 / E9c.m_K40 / 1000)**(1/4) / E9c.lambda_K40_D1 * V0**(-1/4)