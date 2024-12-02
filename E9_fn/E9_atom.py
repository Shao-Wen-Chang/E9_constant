# Recommended import call: import E9_fn.E9_atom as E9a
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
    Also should I have a factor of 2?
    """
    return 4 * np.pi * abs(1 / (1 / a_s + r_eff * k**2 / 2 - 1j * k))**2

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

def gravity_compensation_BGrad(mass, gF, mF):
    """B gradient required to compensate gravity; obtained by setting dE/dz = mg. Returns T/m"""
    BGradSI = mass * g_earth / (gF * mu_B * mF)
    print("{} Gauss/cm".format(BGradSI * 1e4 / 1e2))
    return BGradSI

def majorana_loss_rate(hfs, mF, Bgrad, T):
    """Returns an approximation of Majorana spin flip loss rate in an unplugged magnetic trap.
    
    See e.g. Y-J Lin's paper.
        mu: = g-factor * mu_B"""
    return 1.85 * hbar / hfs.mass * (hfs.gF * mF * mu_B * Bgrad / k_B / T)**2

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
    def Getgravity_compensation_BGrad(self, mF):
        return gravity_compensation_BGrad(self.mass, self.gF, mF)
    
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
#%%
