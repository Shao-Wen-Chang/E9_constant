# Recommended import call: import E9_fn.E9_constants as E9c
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns
# All in SI units unless otherwise noted. Also, I try to use FREQUENCIES instead of ANGULAR FREQUENCIES all the time, and if
# I ever use ANGULAR FREQUENCIES I prefer to include the factor of 2pi explicitly. e.g. (2 * pi) * 20. The units for ANGULAR
# FREQUENCIES is [rad/s].
# In the beginning of comments / docstrings, specify the unit of each input and output with [].

# References (alphabetical):
#     [Axner04]: Ove Axner 04 Line strengths, A-factors and absorption cross-sections for fine structure lines in
#                multiplets and hyperfine structure components in lines in atomic spectrometry — a user's guide
#     [BECDilute]: "Bose-Einstein Condensation in Dilute Gases" by C. J. Pethick, H. Smith
#     [Bloch08]: Bloch 08 Many-body physics with ultracold gases
#     [Claire]: Claire's thesis
#     [Foot]: "Atomic physics" by C. Foot
#     [Grimm99]: Grimm 99 Optical dipole traps for neutral atoms
#     [Ishikawa17]: Ishikawa 17 Vapor pressure of alkali-metal binary alloys in glass cells
#     [Ketterle08]: Ketterle 08 Making, probing and understanding ultracold Fermi gases
#     [Le Kien13]: Le Kien 13 Dynamical polarizability of atoms in arbitrary light fields: general theory and application to cesium
#     [PvdS]: Peter van der Straten & Metcalf
#     [SteckRb]: "Rubidium 87 D Line data" v2.2.1 by Daniel Adam Steck
#     [TBarter]: Tom's thesis
#     [TieckeK]: "Properties of Potassium" v1.03 by Tobias Tiecke
#     [Tarruell18]: Tarruell 18 Quantum simulation of the Hubbard model with ultracold fermions in optical lattices

# Figure assignments:
#     0100s: External quantities
#         100:  Partial pressure plots
#     1000s: Hyperfine level related
#         1000: Breit-Rabi formula for Zeeman level energy vs B field
#         1001: Breit-Rabi formula for neighboring Zeeman level energy splittings vs B field
#     2000s: Feshbach resonance rated
#         2000: scattering length vs B field

#%% natural constants
# all values are in SI unit, and (perhaps in the future) adjusted to different
# units based on some conversion factor
c_light = 299792458             # speed of light
hbar = 1.054571817e-34          # reduced Planck constant
hnobar = hbar * 2 * np.pi       # Planck constant
k_B = 1.380649e-23              # Boltzmann constant
e_ele = 1.602176634e-19         # Charge of electrons
m_e = 9.1093837015e-31          # Mass of electrons
m_u = 1.66053906892e-27         # Atomic mass unit
epsilon_0 = 8.8541878128e-12    # vacuum permittivity
mu_0 = 1.25663706212e-6         # vacuum permeability
Z_0 = np.sqrt(mu_0 / epsilon_0) # vacuum impedance
R_gas = 8.31446261815324        # Ideal gass constant (exact)
N_A = 6.02214076e23             # Avogadro's number
g_earth = 9.8

# Bohr stuff aka atomic units in SI
a0 = 0.529177210903e-10         # Bohr radius, = 4 * pi * epsilon_0 * hbar^2 / (m_e * e_ele**2)
mu_B = 9.274009994e-24          # Bohr magneton, = e_ele * hbar / (2 * m_e)

# Other unit conversion constants
pol_SI2au = 1/1.64877727436e-41 # Conversion factor from SI to a.u. for polarizability | TODO: from where?

########## isotope specific constants (from [SteckRb] and [TieckeK] unless otherwise noted) ##########
m_Rb87 = 1.4431606e-25
m_K40 = 6.636178e-26
m_K39 = 6.470076e-26
I_Rb87 = 3/2
I_K40 = 4
I_K39 = 3/2
gJ_K_4S1o2 = 2.00229421
gJ_Rb_5S1o2 = 2.00233113
# Hyperfine structure coefficients, ahf (M1) and bhf (E2), in J
ahf_39K_4S1o2 = 230.8598601e6 * hnobar
ahf_40K_4S1o2 = -285.7308e6 * hnobar
ahf_87Rb_5S1o2 = 3.417341305452145e9 * hnobar

# scattering lengths
# also see FeshbachResonance in E9_atom.py
# TODO: add references
# format: (mononuclear) a_bg_(isotope)
#         (heteronuclear) a_bg_(isotope1)_(isotope2)
a_bg_Rb87 = 100.4 * a0      # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.093201
a_bg_K40 = 174 * a0
a_bg_K39 = -19 * a0
a_bg_K40_Rb87 = -215 * a0   # or -185? not sure, both from C Ospelkaus thesis
a_bg_Li6 = -1405 * a0       # not sure

# Line constants organized by species
    # Saturation intensities are in [W/m^2]; divide by 10 for [mW/cm^2]. Assuming circularly polarized light
    # nu's and gamma's are in FREQUENCIES, not ANGULAR FREQUENCIES
# 39K
lambda_K39_D1 = 770.108385049e-9
lambda_K39_D2 = 766.700921822e-9
# 40K
nu_K40_4_2P1o2 = 389.286184353e12 # 40K D1 line
nu_K40_4_2P3o2 = 391.016296050e12 # 40K D2 line
nu_K40_4_2P = (nu_K40_4_2P1o2 + nu_K40_4_2P3o2) / 2 # average of D1 & D2 line for convenience
Delta_K40_4_2P = (nu_K40_4_2P3o2 - nu_K40_4_2P1o2) # diffference of D1 & D2 line for convenience
lambda_K40_D1 = 770.108136507e-9
lambda_K40_D2 = 766.700674872e-9
gamma_40K_D2 = 6.035e6
# 87Rb
nu_Rb87_4_2P1o2 = 377.107463380e12      # 87Rb D1 line
nu_Rb87_4_2P3o2 = 384.2304844685e12     # 87Rb D2 line
nu_Rb87_4_2P = (nu_Rb87_4_2P1o2 + nu_Rb87_4_2P3o2) / 2  # average of D1 & D2 line for convenience
Delta_Rb87_4_2P = (nu_Rb87_4_2P3o2 - nu_Rb87_4_2P1o2)   # diffference of D1 & D2 line for convenience
lambda_Rb87_D2 = 780.241209686e-9
gamma_Rb87_D2 = 6.0666e6
I_sat_Rb87_D2 = 16.6933

# Trap loss constants
# naming convention is loosely Gxloss(_situation)_species(_state), where x is the number of particles involved
# "situation" might include
#   _MOT (MOT; light assisted) (G2)
#   _sdr (spin dipole relaxation) (G2)
# units are in [m^(3 * x) s-1]; values are for thermal gases, divide by 6 for atoms in a condensate
G2loss_MOT_Rb87 = 5.8e-12 * 1e-6        # Rb MOT, Phys. Rev. Lett. 69, 897 (1992).
G3loss_Rb87_F1mFm1 = 4.3e-29 * 1e-12    # Rb |1, -1>, Phys. Rev. Lett. 79, 337 (1997).
G3loss_Rb87_F2mF2 = 1.1e-28 * 1e-12     # Rb |2, 2>, Appl Phys B 69, 257 (1999).
# hard to quote a number for K39 due to its many low-lying Feshbach resonances - see Phys. Rev. A 106, 043320 (2022).
G3loss_Rb87_K40 = 2.8e-28 * 1e-12       # Rb-Rb-K40, C Ospelkaus thesis

# Non-Rb or K constants
m_Li6 = 9.9883414e-27

#%% lab constants
# B field related
FBcoil_coeff = 1.64 * 1e-4 * 1e2 # [Claire] in T/m/Ampere; determines B field near FB coil center when running a quadrupole field
# beam waists, taken from [Claire] and [TBarter] (I believe those "w"'s are actually diameters, i.e. w0*2; usually beam waist is called w0)
w0_sw = 40e-6
w0_lw = 50e-6
w0_ODT = 50e-6
# optical lattice related
lambda_sw = 532e-9
lambda_lw = 1064e-9
lambda_vert = 1064e-9
f_sw = c_light / lambda_sw
f_lw = c_light / lambda_lw
f_vert = c_light / lambda_vert
# Lattice vectors following Thomas' convention
# lowercase or sw: 532; uppercase or lw: 1064 & superlattice
    # k's: laser wavevectors
k_sw = 2 * np.pi / lambda_sw
k_lw = 2 * np.pi / lambda_lw
k1 = k_sw * np.array([np.sqrt(3)/2, -1/2])
k2 = k_sw * np.array([0, 1])
k3 = k_sw * np.array([-np.sqrt(3)/2, -1/2])
K1 = k_lw * np.array([np.sqrt(3)/2, -1/2])
K2 = k_lw * np.array([0, 1])
K3 = k_lw * np.array([-np.sqrt(3)/2, -1/2])
    # g's: reciprocal lattice vectors
g_sw = k_sw * np.sqrt(3)
g_lw = k_lw * np.sqrt(3)
g1 = k2 - k3
g2 = k3 - k1
g3 = k1 - k2
G1 = K2 - K3
G2 = K3 - K1
G3 = K1 - K2
g1g = g1/k_sw   # "Normalized" G vectors, or g_tilde in Claire's Thesis
G1G = G1/k_lw   # Should be the same as g1g
g2g = g2/k_sw
G2G = G2/k_lw
    # e_pol's: polarization vectors
e_pol1 = np.array([-1/2, np.sqrt(3)/2])
e_pol2 = np.array([1, 0])
e_pol3 = np.array([-1/2, np.sqrt(3)/2])
    # a's: real space lattice vectors
a_sw_tri = lambda_sw * (2 / 3)
a_lw_hex = lambda_lw * (2 / 3 / np.sqrt(3))
a_vert = lambda_vert / 2
a1 = a_sw_tri * np.array([0, 1])
a2 = a_sw_tri * np.array([-np.sqrt(3)/2, 1/2])
a3 = a_sw_tri * np.array([np.sqrt(3)/2, 1/2])
A1 = a_lw_hex * np.array([0, 1])
A2 = a_lw_hex * np.array([-np.sqrt(3)/2, 1/2])
A3 = a_lw_hex * np.array([np.sqrt(3)/2, 1/2])
    # (all) high-symmetry points in the quasimomentum space
    # (in clockwise order, starting from the kappa point in +y axis) (multiplied by g/2)
    # k - [0, 2/np.sqrt(3)] // [1, 1/np.sqrt(3)] // [1, -1/np.sqrt(3)] // (negative of them)
    # M - [1/2, np.sqrt(3)/2] // [1, 0] // [1/2, -np.sqrt(3)/2] // (negative of them)
Gp = np.array([0, 0])
Kp, Kp2, Kp3, Kp4, Kp5, Kp6 = -K1, K3, -K2, K1, -K3, K2
Mp, Mp2, Mp3, Mp4, Mp5, Mp6 = G2 / 2, -G1 / 2, G3 / 2, -G2 / 2, G1 / 2, -G3 / 2
mp, mp2, mp3, mp4, mp5, mp6 = g2 / 2, -g1 / 2, g3 / 2, -g2 / 2, g1 / 2, -g3 / 2
kp, kp2, kp3, kp4, kp5, kp6 = -k1, k3, -k2, k1, -k3, k2
    # "Jsets" in Claire's thesis
Jset1064 = {(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)}
J1set1064 = {(1, 0), (0, 1), (-1, -1)} # J1set and J2set are defined for including A-B offset in calculation
J2set1064 = {(-1, 0), (0, -1), (1, 1)}
Jset532 = {(2, 0), (-2, 0), (0, 2), (0, -2), (-2, -2), (2, 2)}
Jsets = [None, Jset532, Jset1064]
    # for plot: points that define BZ boundaries, normalized by k_lw
pt11, pt12, pt13, pt14, pt15, pt16 = K2 / k_lw, -K3 / k_lw, K1 / k_lw, -K2 / k_lw, K3 / k_lw, -K1 / k_lw
pt21, pt22, pt23, pt24, pt25, pt26 = -G3 / k_lw, G1 / k_lw, -G2 / k_lw, G3 / k_lw, -G1 / k_lw, G2 / k_lw
pt41, pt42, pt43, pt44, pt45, pt46 = K2 * 2 / k_lw, -K3 * 2 / k_lw, K1 * 2 / k_lw, -K2 * 2 / k_lw, K3 * 2 / k_lw, -K1 * 2 / k_lw
BZ1_vertices = [pt11, pt12, pt13, pt14, pt15, pt16, pt11]
BZ2_vertices = [pt21, pt11, pt22, pt12, pt23, pt13, pt24, pt14, pt25, pt15, pt26, pt16, pt21]
BZ3_vertices = [pt21, pt22, pt23, pt24, pt25, pt26, pt21]
BZ4_vertices = [pt41, pt42, pt43, pt44, pt45, pt46, pt41]
    # Lattice acceleration related; for convention that is rotated 60 deg from the lab (s.t. I can choose the Kp
# on y axis), kB12 = K2, kB23 = -K1
kB12 = K3
kB23 = -K1
# freq_K = np.linalg.norm(G3)*hbar/np.sqrt(3)/m_unit/A/l_unit**2 # don't remember what's this

# recoil energies
E_R1064_Rb87 = hbar**2 / 2 / m_Rb87 * (2*np.pi/lambda_lw)**2 # 1064 photon recoil energy
E_R532_Rb87  = hbar**2 / 2 / m_Rb87 * (2*np.pi/lambda_sw)**2 # 532 photon recoil energy
E_R1064_K40  = hbar**2 / 2 / m_K40 * (2*np.pi/lambda_lw)**2 # 1064 photon recoil energy
E_R532_K40   = hbar**2 / 2 / m_K40 * (2*np.pi/lambda_sw)**2 # 532 photon recoil energy

#%% Utility constants (plots etc.)
# BZ color schemes
# [qarea, BZ1, BZ2, BZ3, BZ4, ...]
BZcolor_PRL = ['red', '#B3FFB3', '#B3FFFF', '#FFB3B3', '#B3B3FF']
BZcolor_white = ['white', 'white', 'white', 'white', 'white']
BZcolor_trans = ['none', 'none', 'none', 'none', 'none']
cmp_husl = ListedColormap(sns.color_palette('husl', 256)) # for plotting phase information

#%% Utility functions
def LinearFn(a, b, x):
    return a + b * x

def Gaussian(x, sigma, mu = 0):
    """Gaussian function normalized such that Gaussian(x = mu) = 1. (Integrates to sigma * np.sqrt(2 * pi))"""
    return np.exp(-(x - mu)**2 / 2 / sigma**2)

#%% natural constants
def gJ(S, L, J, gS = 2, gL = 1):
    """Returns g_J, i.e. the Lambde g-factor (see Rb data); replace gS and gL by actual values if that accuracy (~0.1%) is needed.
    
    This, and gF, are easily obtained by applying the projection theorem."""
    return gL * (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1)) \
        + gS * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

def gF(I, J, F, gJ = 2, gI = 0):
    """Returns g_F (see Rb data), but note that gJ is often not ~2; replace gI by actual values if that accuracy (~0.1%) is needed.
    
    Special case: for mF = F = J + I (stretched state), gF * mF = gJ * J
    gF * mu_B / 1e4 / hnobar gives the change in energy difference between Zeeman sublevels, in MHz/Gauss"""
    return gJ * (F * (F + 1) - I * (I + 1) + J * (J + 1)) / (2 * F * (F + 1)) \
        + gI * (F * (F + 1) + I * (I + 1) - J * (J + 1)) / (2 * F * (F + 1))