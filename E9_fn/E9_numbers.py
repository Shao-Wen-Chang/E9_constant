# Recommended import call: import E9_fn.E9_numbers as E9n
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta

from E9_fn.E9_constants import *
# See E0_constants for list of references and some file conventions

#%% Scattering properties
def xsection_s(a_s, k, r_eff = 0):
    """[m^2] Returns the s-wave scattering cross-section.
    
    I need to review the physics!
    Also should I have a factor of 2?"""
    return 4 * np.pi * abs(1 / (1 / a_s + r_eff * k**2 / 2 - 1j * k))**2

#%% Thermodynamical properties of Bose gases
def T_BEC_bose(wbar, N, a_s: float = 0, V = 0, m = m_Rb87):
    """[K] Returns the BEC critical temperature of a Bose gas.
    
    See [BECDilute] p.23. When wbar is 0, assume that atoms are trapped in a box of volume V. In the case where a != 0, Tc is
    shifted accordingly ([Bloch08] eqn.11 & the equation to its right) and the effect is quite sizable (of order 0.1*Tc).
        wbar: [rad/s] Trap frequency
        N: [#] atom number
        a_s: [m] s-wave scattering length
        V: [m^3] (only for wbar = 0) box volume
        m: [kg] (only for wbar = 0) mass of the atom"""
    if wbar != 0:
        if a_s != 0: print("interaction effect is not included yet in the case of harmonic potential")
        return (1 + 1.32 * (N / V) * a_s) * hbar * wbar * (N * zeta(3))**(1/3) / k_B
    else:
        return (1 + 1.32 * (N / V) * a_s) * (2 * np.pi * hbar**2 / m) * (N / V / zeta(3/2))**(2/3) / k_B

def mu_BEC_har(m, wbar, a_s, N):
    """[E] Returns the chemical potential of an interacting BEC in a
    harmonic potential at T = 0.
    
    See e.g. [BECDilute] eqn.(6.35)."""
    if a_s == 0:
        logging.warning("No interaction in mu_BEC_har, results might not make sense")
    abar = np.sqrt(hbar / m / wbar)
    return (15**(2/5) / 2) * (N * a_s / abar)**(2/5) * hbar * wbar

def N_collapse_bose(a):
    """[#] Returns the (order of magnutide estimate) of critical number N_c for an attractive Bose gas, above which the
    gas collapses.
    
    See [BECDilute] p.164, and references therein.
        a: [m] s-wave scattering length"""
    pass

#%% Thermodynamical properties of Fermi gases
def fermi_energy_lat(m, wbar, a_lat, N):
    """Returns the Fermi energy of N fermions in a single spin component, loaded in a (square) lattice + harmonic confinement.
    
    See e.g. Eqn.(20) in [Tarruell18]. (I actually don't know how to compute this.) Often we need to compare E_F to U and t."""
    return (m * wbar**2 * a_lat**2 / 2) * (N / (4 * np.pi / 3))**(2/3)

def fermi_energy_har(wbar, N):
    """Returns the Fermi energy of N fermions in a single spin component, loaded in a harmonic confinement.
    
    See e.g. Eqn.(33) in [Ketterle08]."""
    return hbar * wbar * (6 * N)**(1/3)

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

def density_profile(m, wx, wy, wz, N, pos_arr, z = 0):
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
    k_F = np.sqrt(2 * m * k_B * T_F) / hbar
    ka = k_F * a_s
    print('ka = {}; 1 / (ka) = {}'.format(ka, 1 / ka))
    return ka

def mu_fermi():
    """[Bloch08]"""
    pass

#%% (Mainly) values that depends on B field but not optical beams
# By convention I print a more convenient form and return SI unit values, but check 
def BreitRabi(ahf, F, I, S, gI, gJ, mF, B):
    """Breit-Rabi formula for Zeeman level splitting for all fields for the special case L = 0 & S = 1/2"""
    x = (gJ - gI) * mu_B / (ahf * (I + 1/2)) * B
    if abs(mF) == I + S:
        # This is the special case where there is only one state with the given mF, and ...
        sgn = np.sign(mF / F)
        return - ahf / 4 + gI * mu_B * mF * B + (ahf * (I + 1/2) / 2) * (1 + sgn * x)
    else:
        # ... for other cases we have mF = mI + 1/2 and mF = mI' - 1/2, so we need to diagonalize a 2x2 matrix
        sgn = np.sign((F - I) / S)
        return - ahf / 4 + gI * mu_B * mF * B + (sgn * ahf * (I + 1/2) / 2) * np.sqrt(1 + 4 * mF * x / (2 * I + 1) + x**2)

def InterationParameter(T, FBres):
    """Returns |k_F * a|, which characterizes interaction strength and therefore BEC-BCS crossover."""
    pass

def GravityCompensationBGrad(mass, gF, mF):
    """B gradient required to compensate gravity; obtained by setting dE/dz = mg. Returns T/m"""
    BGradSI = mass * g_earth / (gF * mu_B * mF)
    print("{} Gauss/cm".format(BGradSI * 1e4 / 1e2))
    return BGradSI

def Majorana_loss_rate(hfs, mF, Bgrad, T):
    """Returns an approximation of Majorana spin flip loss rate in an unplugged magnetic trap.
    
    See e.g. Y-J Lin's paper.
        mu: = g-factor * mu_B"""
    return 1.85 * hbar / hfs.mass * (hfs.gF * mF * mu_B * Bgrad / k_B / T)**2

def QuadrupoleBField(pos, coil_coeff, I):
    """Returns the B field at pos (relative to coil center) when the coil pair is configured to generate a quadrupole field.
    
    pos: a (3 x n) array, where pos[:,i] is the i-th spatial point
    This is only accurate near the center. For off-center fields, consider modelling with magpylib instead."""
    M = np.diag([0.5, 0.5, -1])
    B = M @ (I * coil_coeff * pos)
    return B

def SpinSeparationBGradToF(m, gF, t, dx):
    """B gradient required to separate neighboring Zeeman sublevels by dx during ToF t. Returns T/m"""
    BGradSI = 2 * dx * m / (gF * mu_B * t**2)
    print("{} Gauss/cm".format(BGradSI * 1e4 / 1e2))
    return BGradSI

def SpinSeparationBGradMSF(m, gF, t, dx):
    """B gradient required to separate neighboring Zeeman sublevels by dx during MSF t. Returns T/m"""
    # omega = pi / (2 * t)
    # BGradSI = m * dx * omega**2 / (gF * mu_B)
    # print("{} Gauss/cm".format(BGradSI * 1e4 / 1e2))
    # return BGradSI
    print("not worked out yet")

# Too many helper functions!
def ZeemanSplitting(gF):
    """Splitting between Zeeman sublevels as a function of B field. Returns J/Tesla"""
    ZsplitSI = gF * mu_B
    print("{} MHz/Gauss".format(ZsplitSI / 1e4 / hnobar / 1e6))
    return ZsplitSI

def ZeemanSplittingdx(gF, dx):
    """Change in Zeeman splitting between lattice sites separated by dx along B gradient. Returns J*m*T^-1"""
    ZsplitGradSI = gF * mu_B * dx
    print("{} Hz*(Gauss/cm)^-1".format(ZsplitGradSI / 1e4 * 1e2 / hnobar))
    return ZsplitGradSI

def ZeemanSplittingdBdx(gF, dBdx):
    """Change in Zeeman splitting given some B gradient dBdx. Returns J*m^-1"""
    ZsplitGradSI = gF * mu_B * dBdx
    print("{} Hz/um".format(ZsplitGradSI / 1e6 / hnobar))
    return ZsplitGradSI


#%% Values that depend on optical beams but not B field
def I_from_power(P0, w0):
    """[W/m^2] Return peak intensity of a gaussian beam with power P and beam waist w0.
    
    Also, I = (c_light * n * epsilon_0 / 2) * |E|**2 .
    P0: [W] Power
    w0: [m] beam WAIST (the RADIUS of the beam at 1/e^2 intensity)"""
    return 2 * P0 / np.pi / w0**2

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
    return (3 * np.pi * c_light**2 / 2 / w0**3) * (Gamma / Delta) * (1 + (P_pol * gF * mF / 3) * (Delta_FS / Delta)) * I

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

def LambDickeConst(V0, a_lat = a_sw_tri):
    """(tentative) Returns the Lamb-Dicke constant for a given 532 lattice depth.
    
    This currently assumes using K D1 line cooling in a 532 triangular lattice and can be easily generalized.
        V0:     [kHz] 532 lattice depth
        a_lat:  [m] lattice constant"""
    return (hbar * np.pi * a_lat**2 / 2 / m_K40 / 1000)**(1/4) / lambda_K40_D1 * V0**(-1/4)

#%% class HyperfineState
class HyperfineState():
    def __init__(self, mass, I, J, F, gJ, gI = 0, ahf = None, bhf = None, nu = 0):
        self.mass = mass
        self.S = 1/2
        self.F = F
        self.I = I
        self.J = J
        self.gJ = gJ
        self.gI = gI
        self.gF = gF(I, J, F, gJ, gI)
        self.ahf = ahf
        self.bhf = bhf
        self.nu = nu # energy relative to ground state (in frequency unit, not radial frequency); ground state should
                     # be obvious in most cases
    
    # methods copying from various functions defined above (might want to delete all these)
    def GetGravityCompensationBGrad(self, mF):
        return GravityCompensationBGrad(self.mass, self.gF, mF)
    
    def GetSpinSeparationBGradToF(self, t = 20e-3, dx = 100e-6):
        return SpinSeparationBGradToF(self.mass, self.gF, t, dx)
    
    def GetSpinSeparationBGradMSF(self, t = 50e-3, dx = 100e-6):
        return SpinSeparationBGradMSF(self.mass, self.gF, t, dx)
    
    def GetZeemanSplitting(self):
        return ZeemanSplitting(self.gF)
    
    def GetZeemanSplittingdx(self, dx = lambda_sw / np.sqrt(3)):
        return ZeemanSplittingdx(self.gF, dx)
    
    def GetZeemanSplittingdBdx(self, dBdx = 0.01):
        return ZeemanSplittingdBdx(self.gF, dBdx)
    
    def GetBreitRabi(self, mF, B):
        return BreitRabi(self.ahf, self.F, self.I, self.S, self.gI, self.gJ, mF, B)
    
    def PlotBreitRabi(self, ax = None, Bmin = 0, Bmax = 1000):
        """Plot the energy of all mF levels within B = [0, Bmax]."""
        Bmid = (Bmin + Bmax) / 2
        BTesla = np.linspace(Bmin, Bmax, 500) / 1e4
        if ax == None:
            f = plt.figure(1000)
            f.clf()
            ax = f.add_subplot(111)
            ax.grid()
        E_mF = np.zeros([int(2 * self.F + 1), len(BTesla)])
        for i, mFnow in enumerate(np.arange(- self.F, self.F + 1, 1)):
            E_mF[i, :] = self.GetBreitRabi(mFnow, BTesla) / hnobar / 1e6
        # Plot the first line to make legend handle and get line color etc., then plot the rest of the lines
        p1 = ax.plot(BTesla * 1e4, E_mF[0, :], label = 'F = {}'.format(self.F))
        for i in range(1, int(2 * self.F + 1)):
            ax.plot(BTesla * 1e4, E_mF[i, :], color = p1[0].get_color())
        ax.plot(Bmid, self.GetBreitRabi(self.F, Bmid / 1e4) / hnobar / 1e6, '+', markersize = 20, color = p1[0].get_color())
        ax.plot(Bmid, self.GetBreitRabi(-self.F, Bmid / 1e4) / hnobar / 1e6, '_', markersize = 20, color = p1[0].get_color())
        ax.legend()
        ax.set_xlabel("B [G]")
        ax.set_ylabel("E/h [MHz]")
        ax.set_title("Zeeman sublevel energies")
        return ax
    
    def PlotBreitRabiDiff(self, ax = None, Bmin = 0, Bmax = 1000, avg2zero = False, tolerance = 0):
        """Plot the Zeeman level splitting between all (mF, mF - 1) level pairs within B = [Bmin, Bmax].
        
        If avg2zero = True, then a dynamical offset is added to each point such that all traces together average
        to zero. The value of splitting is then only relative.
        "tolerance" addes shades around each curve, which represents the resolution of rf drive (often set by the
        stability of magnetic field). Unit is specified in MHz."""
        def BRsplitting(mF, BTesla):
            return (self.GetBreitRabi(mF, BTesla) - self.GetBreitRabi(mF - 1, BTesla)) / hnobar / 1e6
        
        Bmid = (Bmin + Bmax) / 2
        BTesla = np.linspace(Bmin, Bmax, 500) / 1e4
        if ax == None:
            f = plt.figure(1001)
            f.clf()
            ax = f.add_subplot(111)
            ax.grid()
        E_mF = np.zeros([int(2 * self.F), len(BTesla)])
        for i, mFnow in enumerate(np.arange(- self.F + 1, self.F + 1, 1)):
            E_mF[i, :] = BRsplitting(mFnow, BTesla)
        if avg2zero: E_mF = E_mF - np.tile(np.mean(E_mF, axis = 0), (int(2 * self.F), 1))
        # Plot the first line to make legend handle and get line color etc., then plot the rest of the lines
        p1 = ax.plot(BTesla * 1e4, E_mF[0, :], label = 'F = {}'.format(self.F))
        ax.fill_between(BTesla * 1e4, E_mF[0, :] - tolerance, E_mF[0, :] + tolerance, color = p1[0].get_color(), alpha = 0.4)
        for i in range(1, int(2 * self.F)):
            ax.plot(BTesla * 1e4, E_mF[i, :], color = p1[0].get_color())
            ax.fill_between(BTesla * 1e4, E_mF[i, :] - tolerance, E_mF[i, :] + tolerance, \
                            color = p1[0].get_color(), alpha = 0.4)

        ax.plot(Bmid, E_mF[-1, int(E_mF.shape[1] / 2)], '+', markersize = 20, color = p1[0].get_color())
        ax.plot(Bmid, E_mF[0, int(E_mF.shape[1] / 2)], '_', markersize = 20, color = p1[0].get_color())
        ax.legend()
        ax.set_xlabel("B [G]")
        ax.set_ylabel("$\Delta$E/h [MHz]")
        ax.set_title("$\Delta E \equiv E_{m_F} - E_{m_F - 1}$" + "; avg2zero = {}, tolerance = {:.4f} MHz".format(avg2zero, tolerance))
        return ax

# format: (isotope)_n_(term symbol)_F(F value); 9/2 -> 9o2 etc
K40_4_2S1o2_F9o2 = HyperfineState(m_K40, I_K40, 1/2, 9/2, gJ = gJ(1/2, 0, 1/2), ahf = ahf_40K_4S1o2)
K40_4_2S1o2_F7o2 = HyperfineState(m_K40, I_K40, 1/2, 7/2, gJ = gJ(1/2, 0, 1/2), ahf = ahf_40K_4S1o2)
K39_4_2S1o2_F1 = HyperfineState(m_K39, I_K39, 1/2, 1, gJ = gJ(1/2, 0, 1/2), ahf = ahf_39K_4S1o2)
K39_4_2S1o2_F2 = HyperfineState(m_K39, I_K39, 1/2, 2, gJ = gJ(1/2, 0, 1/2), ahf = ahf_39K_4S1o2)
Rb87_5_2S1o2_F1 = HyperfineState(m_Rb87, I_Rb87, 1/2, 1, gJ = gJ(1/2, 0, 1/2), ahf = ahf_87Rb_5S1o2)
Rb87_5_2S1o2_F2 = HyperfineState(m_Rb87, I_Rb87, 1/2, 2, gJ = gJ(1/2, 0, 1/2), ahf = ahf_87Rb_5S1o2)

#%% class FeshbachResonance
class FeshbachResonance():
    """For FeshbachResonance, (scattering) lengths are in a0, and magnetic fields are in Gauss."""
    def __init__(self, a_bg, B0, DB):
        self.a_bg = a_bg
        self.B0 = B0
        self.DB = DB
    
    def Getasc(self, B):
        """Get the scattering length at field B (assuming that there's no nearby resonances)."""
        return self.a_bg * (1 - self.DB / (B - self.B0))
    
    def GetB(self, a):
        """Get the field B required for scattering length a (assuming that there's no nearby resonances)."""
        return self.B0 + self.DB / (1 - a / self.a_bg)
    
    def GetdadBfora(self, a):
        """Get da/dB at scattering length a."""
        return self.a_bg * self.DB / (self.GetB(a) - self.B0)**2
    
    def GetdadBforB(self, B):
        """Get da/dB at field B."""
        return self.a_bg * self.DB / (B - self.B0)**2
    
    def Visualize(self, ax = None):
        """Plot Feshbach resonance in cgs unit."""
        B1s, B2s = np.linspace(max(0, self.B0 - 2 * abs(self.DB)), self.B0 - 1e-3, 200), np.linspace(self.B0 + 1e-3, self.B0 + 2 * abs(self.DB), 200)
        asc1s, asc2s = self.Getasc(B1s), self.Getasc(B2s)
        if ax == None:
            f = plt.figure(2000)
            f.clf()
            ax = f.add_subplot(111)
            ax.grid()
        p1 = ax.plot(B1s, asc1s, '-')
        ax.plot(B2s, asc2s, '-', color = p1[0].get_color())
        ax.plot(self.B0, self.a_bg, 'x', color = p1[0].get_color())
        ax.set_ylim(self.a_bg - abs(self.a_bg) * 10, self.a_bg + abs(self.a_bg) * 10)
        ax.set_xlabel("B [G]")
        ax.set_ylabel(r"$a_{sc}/a_0$")
        return ax

# format: (mononuclear) FBres_(isotope)_mF1_mF2[_x]; 9/2 -> 9o2 etc
#         (heteronuclear) FBres_(isotope1)_F_mF1_(isotope2)_F_mF2[_x]
FBres_K40_9o2_7o2 = FeshbachResonance(174, 202.1, 7.8)
FBres_K40_9o2_5o2 = FeshbachResonance(174, 224.21, 9.7)
FBres_K40_1o2_n1o2 = FeshbachResonance(174, 389.7, 26.7)
FBres_K39_n1_n1_1 = FeshbachResonance(-19, 32.6, -55)
FBres_K39_n1_n1_2 = FeshbachResonance(-19, 162.8, 37)
FBres_K40_9o2_9o2_Rb87_1_1 = FeshbachResonance(-215, 545.4, -1.2)
FBres_Li6_1_2 = FeshbachResonance(-1405, 834, -300) # not sure about a_bg

#%% Codes for execution; use figure numbers < 100 for additional figures

# Only use this for simple stuff as I delete them promptly
if __name__ == "__main__":
    Bgrad = 1.5 # T/m
    Temp = 25e-6 # K
    print("Trap gradient = {} [G/cm], temperature = {} [uK]".format(Bgrad * 100, Temp * 1e6))
    print("Rb Lifetime = {} [s]".format(1 / Majorana_loss_rate(Rb87_5_2S1o2_F2, 2, Bgrad, Temp)))
    print("40K Lifetime = {} [s]".format(1 / Majorana_loss_rate(K40_4_2S1o2_F9o2, 9/2, Bgrad, Temp)))
    print("39K Lifetime = {} [s]".format(1 / Majorana_loss_rate(K39_4_2S1o2_F2, 2, Bgrad, Temp)))
# %%
