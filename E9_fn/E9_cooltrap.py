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
    
    This is \gamma_p in [PvdS] eqn 2.26."""
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

# Optical dipole traps
def FORT_scattering_rate(I, species: str, f_light):
    """[Hz] Photon scattering rate for far-off-resonance optical traps.
    
    From Grimm eqn.(21); only works for delta >> HFS splitting. For us we don't have
    delta >> FS splitting."""
    if species == "Rb":
        delta1 = 2 * np.pi * (E9c.nu_Rb87_5_2P1o2 - f_light)
        delta2 = 2 * np.pi * (E9c.nu_Rb87_5_2P3o2 - f_light)
        gamma = E9c.gamma_Rb87_D2
        w0 = 2 * np.pi * E9c.nu_Rb87_5_2P
    elif species == "K":
        delta1 = 2 * np.pi * (E9c.nu_K40_4_2P1o2 - f_light)
        delta2 = 2 * np.pi * (E9c.nu_K40_4_2P3o2 - f_light)
        gamma = E9c.gamma_K40_D2
        w0 = 2 * np.pi * E9c.nu_K40_4_2P
    else:
        raise ValueError(f"species = {species} is not supported")
    
    return (np.pi * E9c.c_light**2 * gamma**2 / (2 * E9c.hbar * w0**3)
            * (2 / delta2**2 + 1 / delta1**2)) * I

# Collisional properties
def collision_rate(n: float, sigma: float, m1: float, T1: float, m2: float = None, T2: float = None):
    """[Hz] Returns the collision rate for a (possibly mixture of) thermal gas.
    
    Note that m2 = None and T2 != None is a valid input. Perhaps you are combining two clouds at different
    temperatures. We can't really do this in the lab though.
    
    Args:
        sigma:  [m^2] scattering cross-section
        m1:     [kg] mass of the first species
        T1:     [K] temperature of the first species
        m2:     [kg] mass of the second species. If None, assume m1.
        T2:     [K] temperature of the second species. If None, assume T1.
    
    """
    if m2 is None: m2 = m1
    if T2 is None: T2 = T1
    v_rel_avg = np.sqrt((8 * E9c.k_B / np.pi) * (T1 / m1 + T2 / m2))
    return n * sigma * v_rel_avg

def majorana_loss_rate(hfs, B_grad, T, mF = None):
    """[1/s] Returns an approximation of Majorana spin flip loss rate in an unplugged magnetic trap.
    
    What might actually be happening is depolarization, where the atoms are flipped to a not-as-trappable state
    and either are lost from the trap due the the lower potential, or from further mF-changing collisions.
    The prefactor is tricky:
        - 1.85 is from e.g. Y-J Lin's paper, which cites Cornell's paper. This is experimentally measured
          for Rb |1, -1> atoms.
        - There is a factor of np.log(np.pi * np.sqrt(F)) for different F. This is from Jervis' thesis and
          I can't find what he cited there. The prefactor is defined such that for F = 1 I recover 1.85.
    But in any case, this is a factor on the order of 1, and what really matters is the F dependence.

    Args:
        Bgrad_z:    Gradient along the z-axis (the tight direction).
        mF:         Use None if abs(mF) = F. (Which should always be the case in a functional magnetic trap)
    """
    return _majorana_loss_stuff("rate", hfs, B_grad, T, mF)

def majorana_loss_radius(hfs, B_grad, T, mF = None):
    """[m] Returns the "loss radius" of an atom.
    
    See W. Petrich ..., E. A. Cornell, Phys. Rev. Lett. 74, 3352 (1995).
    In the expression for the loss ellipsoid, I replace v with the rms velocity of a gas at temperature T.
    """
    return _majorana_loss_stuff("radius", hfs, B_grad, T, mF)

def _majorana_loss_stuff(stuff, hfs, B_grad, T, mF):
    """See majorana_loss_rate and majorana_loss_radius."""
    F = hfs.F
    if mF is None:
        mF = F
    elif abs(mF) > F or not util.is_int(mF - F):
        raise ValueError(f"mF = {mF} is illegal for F = {F}")
    prefactor = (1.85 / np.log(np.pi)) * np.log(np.pi * np.sqrt(F))
    mu = hfs.gF * mF * E9c.mu_B

    if stuff == "rate":
        return prefactor * E9c.hbar / hfs.mass * (mu * B_grad / E9c.k_B / T)**2
    elif stuff == "radius":
        v_rms = np.sqrt(3 * E9c.k_B * T / hfs.mass)
        return prefactor * np.sqrt(E9c.hbar / mu / B_grad * v_rms)

#%% Thermodynamical properties of Bose gases
def T_BEC_bose(wbar, N, a_s: float = 0, V = 0, m = E9c.m_Rb87):
    """[K] Returns the BEC critical temperature of a Bose gas.
    
    See [BECDilute] p.23. When wbar is 0, assume that atoms are trapped in a box of volume V. In the case where a != 0, Tc is
    shifted accordingly ([Bloch08] eqn.11 & the equation to its right) and the effect is quite sizable (of order 0.1*Tc).
        wbar: [rad/s] Trap frequency
        N: [#] atom number
        a_s: [m] s-wave scattering length
        V: [m^3] (only for wbar = 0) box volume
        m: [kg] (only for wbar = 0) mass of the atom
    """
    if wbar != 0:
        if a_s != 0: print("interaction effect is not included yet in the case of harmonic potential")
        return (1 + 1.32 * (N / V) * a_s) * E9c.hbar * wbar * (N * zeta(3))**(1/3) / E9c.k_B
    else:
        return (1 + 1.32 * (N / V) * a_s) * (2 * np.pi * E9c.hbar**2 / m) * (N / V / zeta(3/2))**(2/3) / E9c.k_B

def mu_BEC_har(m, wbar, a_s, N):
    """[E] Returns the chemical potential of an interacting BEC in a
    harmonic potential at T = 0.
    
    See e.g. [BECDilute] eqn.(6.35).
    """
    if a_s == 0:
        logging.warning("No interaction in mu_BEC_har, results might not make sense")
    abar = np.sqrt(E9c.hbar / m / wbar)
    return (15**(2/5) / 2) * (N * a_s / abar)**(2/5) * E9c.hbar * wbar

def N_collapse_bose(a):
    """[#] Returns the (order of magnutide estimate) of critical number N_c for an attractive Bose gas, above which the
    gas collapses.
    
    See [BECDilute] p.164, and references therein.
        a: [m] s-wave scattering length
    """
    pass

#%% Thermodynamical properties of Fermi gases
def fermi_energy_lat(m, wbar, a_lat, N):
    """Returns the Fermi energy of N fermions in a single spin component, loaded in a (square) lattice + harmonic confinement.
    
    See e.g. Eqn.(20) in [Tarruell18]. (I actually don't know how to compute this.) Often we need to compare E_F to U and t."""
    return (m * wbar**2 * a_lat**2 / 2) * (N / (4 * np.pi / 3))**(2/3)

def fermi_energy_har(wbar, N):
    """Returns the Fermi energy of N fermions in a single spin component, loaded in a harmonic confinement.
    
    See e.g. Eqn.(33) in [Ketterle08]."""
    return E9c.hbar * wbar * (6 * N)**(1/3)

def fermi_radius(m, w, N, xi, a_s = 0.):
    """Return the Fermi radius in the axis with trap frequency w.
    
    The expression is the same for Bose and Fermi gas, but E_F is different.
        w: [rad/s] trap frequency, assumed to be isotropic.
        xi: [#] -1 for Bose gases, +1 for fermi gases.
    Note that
        for Fermi gases, this is the NON-INTERACTING profile.
        for Bose gases, this is the INTERACTING profile in T-F approx."""
    if xi != 1 and xi != -1:
        logging.error("xi = {}".format(xi))
        raise Exception("xi must be 1 or -1 in fermi_radius")
    elif xi == 1: # Non-interacting Fermi gas
        mu0 = fermi_energy_har(w, N) # Fermi energy, i.e. mu(T = 0)
    elif xi == -1: # Interacting Bose gas
        mu0 = mu_BEC_har(m, w, a_s, N) # chemical potential at T = 0
    return np.sqrt(2 * mu0 / (m * w**2))

def density_profile_fermi(m, wx, wy, wz, N, pos_arr, z = 0):
    """Given a harmonic trapping potential with specified trapping frequencies, for each point in pos_arr (where the origin
    is set at trap center), return the number density at zero temperature.
    
    See e.g. Eqn.(34) in [Ketterle08].
        wx/y/z: [rad/s] trapping frequencies in x / y / z direction
        pos_arr: [m] a (2, L)-dim ndarray, where pos[:, i] = (x, y) of the i-th point. 
        z: [m] specifies the z-coordinate shared by all points in pos."""
    R = np.sqrt(pos_arr[0, :]**2 + pos_arr[1, :]**2 + z**2) # Distances from trap center
    wbar = (wx * wy * wz)**(1/3)
    Rx, Ry, Rz = fermi_radius(m, wx, N), fermi_radius(m, wy, N), fermi_radius(m, wz, N)
    tempfill = 1 - (pos_arr[0, :] / Rx)**2 - (pos_arr[1, :] / Ry)**2 - (z / Rz)**2
    return (8 / np.pi**2) * (N / (Rx * Ry * Rz)) * np.maximum(tempfill, np.zeros_like(tempfill))**(3/2)

def kFa_from_TFa(m, T_F, a_s):
    """Prints k_F * a_s given T_F (Fermi temperature) and a_s (s-wave scattering length). (use SI unit inputs.)"""
    k_F = np.sqrt(2 * m * E9c.k_B * T_F) / E9c.hbar
    ka = k_F * a_s
    print('ka = {}; 1 / (ka) = {}'.format(ka, 1 / ka))
    return ka

def mu_fermi():
    """[Bloch08]"""
    pass

#%% Values that depend on optical beams but not B field
def J_from_Vlat(Vlat, theta = np.pi/2):
    """[dimless] Return (t/Er) given some lattice depth V0 = Vlat/Er, where Er is the (photon) recoil energy.
    
    This is the value obtained by solving the Mathieu equation. See e.g. [Bloch08] eqn.(39).
        theta: [dimless] angle between one of the beam and the symmetry plane. (theta = pi/2 for counter-propagating beams)"""
    V0 = Vlat / np.sin(theta)**2
    return (4 / np.sqrt(np.pi)) * V0**(3/4) * np.exp(-2 * np.sqrt(V0)) / np.sin(theta)**2

def wsite_from_Vlat(Vlat, alat, m):
    """[rad/s] Return the trap frequency (angular frequency) for a lattice potential (Vlat / 2) * sin(2 * pi * x / alat).
    
    This result is obtained by approximating the sites as harmonic traps and is valid for deep traps.
        Vlat: [J] lattice depth. Remember e.g. the factor of 1/9 in honeycomb lattices
        alat: [m] lattice constant
        m: [kg] mass of particles"""
    return (2 * np.pi / alat) * np.sqrt(Vlat / 2 / m)

def V0_from_I(Gamma, nu, fl, I, gF, mF, P_pol = 0, Delta_FS = 0):
    """[J] Gives the trap depth for a hyperfine state (hfs).
    
    See [Grimm99] eqn.20. This works for the large detuning (>> fine structure splitting of relevant excited states) limit.
        Gamma: [Hz] (average) linewidth of relevant excited states. Usually the 2P3/2 and 2P1/2 states.
        nu: [Hz] (average) excited state energy in frequency. Note that w0 = 2 * pi * nu.
        fl: [Hz] frequency of light field. Note that wl = 2 * pi * fl.
        I: [W/m^2] light intensity.
        gF: gF of the (ground) state of trapped atoms.
        mF: mF value of the atom.
        P_pol: Polarization factor. 0 if linear or ignored, +/-1 if sigma+/- polarized
        Delta_FS: [Hz] difference between the two excited state. Note that [Grimm99] use angular frequency [rad/s]."""
    w0 = 2 * np.pi * nu
    wl = 2 * np.pi * fl
    Gamma = 2 * np.pi * Gamma
    Delta = wl - w0
    Delta_FS = 2 * np.pi * Delta_FS
    return (3 * np.pi * E9c.c_light**2 / 2 / w0**3) * (Gamma / Delta) * (1 + (P_pol * gF * mF / 3) * (Delta_FS / Delta)) * I

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