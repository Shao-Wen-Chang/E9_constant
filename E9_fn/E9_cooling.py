# Recommended import call: import E9_fn.E9_cooling as cool
import numpy as np
import matplotlib.pyplot as plt

from E9_fn.E9_constants import *
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
    return hbar * k * (Gamma_scat(s0, delta - v * k + z * Ag, gamma) - Gamma_scat(s0, delta + v * k, gamma))

def beta_molasses(k, s0, delta, gamma):
    """[N/(m/s)] Returns the damping coefficient in an optical molasses.
    
    This is [PvdS] eqn 7.2, which is valid for small velocities (F = -\beta v). Use F_molasses for arbitrary velocities.
    Check F_molasses for units.
    Some values of interest:
        (damping rate) beta / m"""
    return - 8 * hbar * k**2 * delta * s0 / (gamma * (1 + s0 + (2 * delta / gamma)**2)**2)

#%% Trapping related
def collision_rate(n: float, sigma: float, v: float = None, T: float = None, m: float = None):
    """[Hz] Returns the collision rate for a thermal gas.
    
    This function can take either a temperature or some (representative) velocity of the gas,
    but not both. If T is given, v = v_rms (in free space), and m is needed.
        sigma:  [m^2] scattering cross-section
        """
    if v is not None:
        if T is not None:
            raise(Exception("Don't feed both v and T to collision_rate!"))
    elif T is None:
        raise(Exception("You need at least v or T in collision_rate!"))
    else: # v is None and T is some value
        v = np.sqrt(3 * k_B * T / m)
    return n * sigma * v

def kappa_MOT(A, k, s0, delta, gamma, g = 1):
    """[N/m] Returns the restoring force coefficient in a MOT.
    
    Check F_molasses for other inputs.
        A: [T/m] magnetic field gradient (= 0.01 * (gradient in [G/cm]))
        g: [dimless] Effective "g-factor," given by (g_e * m_e - g_g * m_g). This is often around 1. ([Foot], but not sure why;
                     probably has something to do with using the stretched states)
    Some values of interest:
        (harmonic oscillator frequency) np.sqrt(kappa / m)
        (MOT rms size) np.sqrt(k_B * T / kappa)"""
    return g * mu_B * A * beta_molasses(k, s0, delta, gamma) / (hbar * k)

def n_peak_har(N, wbar, T, m):
    """[1/m^3] Returns the peak density in a harmonic trap.
    
    Check BEC (2.39). This formula assumes a thermal gas, i.e. density is
    proportional to exp(-V / (k_B * T)). Other formulae entails similarly."""
    Rbar = np.sqrt(2 * k_B * T / m / wbar**2)
    return N / np.pi**(3/2) / Rbar**3

def n_peak_lin(N, Vg_perp, T):
    """[1/m^3] Returns the peak density in a linear trap.
    
    This is probably a magnetic trap. It is assumed to be cylindrically
    symmetric.
        Vg_perp:    [J/m]   Gradient of the potential in the xy plane."""
    return 2 * N * Vg_perp**3 / (2 * k_B * T)**3

#%% Working area
if __name__ == "__main__":
    k_780 = (2*np.pi/lambda_Rb87_D2)
    s0 = 0.5
    displacement = np.linspace(-0.01, 0.01, num = 201)
    delta_2DMOT = -6 * gamma_Rb87_D2
    A_2DMOT = 0.01*20
    beta_2DMOT = beta_molasses(k_780, s0, delta_2DMOT, gamma_Rb87_D2)
    kappa_2DMOT = kappa_MOT(A_2DMOT, k_780, s0, delta_2DMOT, gamma_Rb87_D2)
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(displacement * 100, - displacement * kappa_2DMOT / (hbar * k_780 * gamma_Rb87_D2))
    print('beta =', beta_2DMOT, 'kg/s, damping rate =', beta_2DMOT / m_Rb87, 'Hz')
    print('kappa =', kappa_2DMOT, 'N/m, w_2DMOT =', np.sqrt(kappa_2DMOT / m_Rb87), 'Hz')
    print('restoring time =', 2 * beta_2DMOT / kappa_2DMOT * 1000, 'ms')
    print('MOT trapping radius =', - 1000 * hnobar * delta_2DMOT / A_2DMOT / mu_B, 'mm') # radius within which there is a restoring force
    # transverse velocity distribution results in around v_rms = 5 m/s, so the 100mK one might make more sense
    print('MOT rms size @ 100 mK =', np.sqrt(k_B * 0.1 / kappa_2DMOT) * 1000, 'mm')
    print('MOT rms size @ 100 uK =', np.sqrt(k_B * 1e-4 / kappa_2DMOT) * 1000, 'mm')
    # ax.set_ylim(-0.4, 0.4)