import numpy as np
import matplotlib.pyplot as plt
# All in SI units unless otherwise noted. Also, I try to use FREQUENCIES instead of ANGULAR FREQUENCIES all the time, and if
# I ever use ANGULAR FREQUENCIES I prefer to include the factor of 2pi explicitlu. e.g. (2 * pi) * 20. The units for ANGULAR
# FREQUENCIES is [rad].
# In the beginning of comments / docstrings, specify the unit of each input and output with [].

# References (alphabetical):
#     [BECDilute]: "Bose-Einstein Condensation in Dilute Gases" by C. J. Pethick, H. Smith
#     [Bloch08]: Bloch 08 Many-body physics with ultracold gases
#     [Claire]: Claire's thesis
#     [Foot]: "Atomic physics" by C. Foot
#     [Grimm99]: Grimm 99 Optical dipole traps for neutral atoms
#     [Ishikawa17]: Ishikawa 17 Vapor pressure of alkali-metal binary alloys in glass cells
#     [Ketterle08]: Ketterle 08 Making, probing and understanding ultracold Fermi gases
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

#%% Plot parameters
if 1:
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['font.size'] = 12           # Default is 10
    if 0:
        plt.rcParams['figure.autolayout'] = True # Use tight layout

#%% natural constants
pi = np.pi
c_light = 299792458          # speed of light
hbar = 1.054571817e-34
hnobar = hbar * 2 * pi
k_B = 1.380649e-23           # Boltzmann constant
e_ele = 1.602176634e-19
m_e = 9.1093837015e-31
epsilon_0 = 8.8541878128e-12 # vacuum permittivity
mu_0 = 1.25663706212e-6      # vacuum permeability
R_gas = 8.31446261815324     # Ideal gass constant (exact)
g_earth = 9.8

# Bohr stuff aka atomic units
a0 = 0.529177e-10 # Bohr radius, = 4 * pi * epsilon_0 * hbar^2 / (m_e * e_ele**2)
mu_B = 	9.274009994e-24 # Bohr magneton, = e_ele * hbar / (2 * m_e)

# isotope specific constants (from [SteckRb] and [TieckeK] unless otherwise noted)
# I chose to organize according to physical quantities instead of isotopes
m_Rb87 = 1.4431606e-25
m_K40 = 6.636178e-26
m_K39 = 6.470076e-26
I_Rb87 = 3/2
I_K40 = 4
I_K39 = 3/2
gJ_K_4S1o2 = 2.00229421
gJ_Rb_5S1o2 = 2.00233113
# Fine structure energies (in frequency unit, relative to the lowest fine structure manifold; note that hyperfine splitting
# is not included)
nu_K40_4_2P1o2 = 389.286184353e12 # 40K D1 line
nu_K40_4_2P3o2 = 391.016296050e12 # 40K D2 line
nu_K40_4_2P = (nu_K40_4_2P1o2 + nu_K40_4_2P3o2) / 2 # average of D1 & D2 line for convenience
Delta_K40_4_2P = (nu_K40_4_2P3o2 - nu_K40_4_2P1o2) # diffference of D1 & D2 line for convenience
# Hyperfine structure coefficients, ahf (M1) and bhf (E2), in J
ahf_39K_4S1o2 = 230.8598601e6 * hnobar
ahf_40K_4S1o2 = -285.7308e6 * hnobar
ahf_87Rb_5S1o2 = 3.417341305452145e9 * hnobar
# D2 line (and nP_(3/2) states) constants
lambda_K39_D2 = 766.700921822e-9
lambda_K40_D2 = 766.700674872e-9
lambda_Rb87_D2 = 780.241209686e-9
    # Saturation intensities are in [W/m^2]; divide by 10 for [mW/cm^2]. Assuming circularly polarized light
I_sat_Rb87_D2 = 16.6933
    # These are FREQUENCIES, not ANGULAR FREQUENCIES
gamma_Rb87_D2 = 6.0666e6
# D1 line (and nP_(1/2) states) constants
lambda_K39_D1 = 770.108385049e-9
lambda_K40_D1 = 770.108136507e-9

# Non-Rb or K constants
m_Li6 = 9.9883414e-27

#%% lab constants
# B field related
FBcoil_coeff = 1.64 * 1e-4 * 1e2 # [Claire] in T/m/Ampere; determines B field near FB coil center when running a quadrupole field
# optical lattice related
lambda_sw = 532e-9
lambda_lw = 1064e-9
lambda_vert = 1064e-9
a_sw_tri = lambda_sw * (2 / 3)
a_lw_hex = lambda_lw * (2 / 3 / np.sqrt(3))
a_vert = lambda_vert / 2
f_sw = c_light / lambda_sw
f_lw = c_light / lambda_lw
f_vert = c_light / lambda_vert
# beam waists, taken from [Claire] and [TBarter] (I believe those "w"'s are actually diameters, i.e. w0*2; usually beam waist is called w0)
w0_sw = 40e-6
w0_lw = 50e-6
w0_ODT = 50e-6
# recoil energies
E_R1064_Rb87 = hbar**2/2/m_Rb87*(2*pi/lambda_sw/2)**2 # 1064 photon recoil energy (made dimensionless)
E_R532_Rb87 = hbar**2/2/m_Rb87*(2*pi/lambda_sw)**2 # 532 photon recoil energy (made dimensionless)
E_R1064_K40 = hbar**2/2/m_K40*(2*pi/lambda_sw/2)**2 # 1064 photon recoil energy (made dimensionless)
E_R532_K40 = hbar**2/2/m_K40*(2*pi/lambda_sw)**2 # 532 photon recoil energy (made dimensionless)

#%% Utility functions
def LinearFn(a, b, x):
    return a + b * x

def Gaussian(x, sigma, mu = 0):
    '''Gaussian function normalized such that Gaussian(x = mu) = 1. (Integrates to sigma * np.sqrt(2 * pi))'''
    return np.exp(-(x - mu)**2 / 2 / sigma**2)

#%% natural constants
def gJ(S, L, J, gS = 2, gL = 1):
    '''Returns g_J, i.e. the Lambde g-factor (see Rb data); replace gS and gL by actual values if that accuracy (~0.1%) is needed.
    
    This, and gF, are easily obtained by applying the projection theorem.'''
    return gL * (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1)) \
        + gS * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

def gF(I, J, F, gJ = 2, gI = 0):
    '''Returns g_F (see Rb data), but note that gJ is often not ~2; replace gI by actual values if that accuracy (~0.1%) is needed.
    
    Special case: for mF = F = J + I (stretched state), gF * mF = gJ * J
    gF * mu_B / 1e4 / hnobar gives the change in energy difference between Zeeman sublevels, in MHz/Gauss'''
    return gJ * (F * (F + 1) - I * (I + 1) + J * (J + 1)) / (2 * F * (F + 1)) \
        + gI * (F * (F + 1) + I * (I + 1) - J * (J + 1)) / (2 * F * (F + 1))