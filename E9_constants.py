import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_6j 
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

custom_plot_style = True
use_tight_layout = False
unit_system = 'SI' # SI, a.u. ((Hartree) atomic unit) (not walked through)

#%% Plot parameters
if custom_plot_style:
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['font.size'] = 18             # Default is 10
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    if use_tight_layout:
        plt.rcParams['figure.autolayout'] = True # Use tight layout

#%% natural constants
pi = np.pi

if unit_system == 'SI':
    c_light = 299792458          # speed of light
    hbar = 1.054571817e-34       # reduced Planck constant
    hnobar = hbar * 2 * pi       # Planck constant
    k_B = 1.380649e-23           # Boltzmann constant
    e_ele = 1.602176634e-19      # Charge of electrons
    m_e = 9.1093837015e-31       # Mass of electrons
    epsilon_0 = 8.8541878128e-12 # vacuum permittivity
    mu_0 = 1.25663706212e-6      # vacuum permeability
    R_gas = 8.31446261815324     # Ideal gass constant (exact)
    g_earth = 9.8
    
    # Bohr stuff aka atomic units
    a0 = 0.529177210903e-10      # Bohr radius, = 4 * pi * epsilon_0 * hbar^2 / (m_e * e_ele**2)
    mu_B = 9.274009994e-24       # Bohr magneton, = e_ele * hbar / (2 * m_e)
    
    # isotope specific constants (from [SteckRb] and [TieckeK] unless otherwise noted)
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
    nu_Rb87_4_2P1o2 = 377.107463380e12 # 87Rb D1 line
    nu_Rb87_4_2P3o2 = 384.2304844685e12 # 87Rb D2 line
    nu_Rb87_4_2P = (nu_Rb87_4_2P1o2 + nu_Rb87_4_2P3o2) / 2 # average of D1 & D2 line for convenience
    Delta_Rb87_4_2P = (nu_Rb87_4_2P3o2 - nu_Rb87_4_2P1o2) # diffference of D1 & D2 line for convenience
    lambda_Rb87_D2 = 780.241209686e-9
    gamma_Rb87_D2 = 6.0666e6
    I_sat_Rb87_D2 = 16.6933
    
    # Non-Rb or K constants
    m_Li6 = 9.9883414e-27
elif unit_system == 'a.u.':
    # SI values for unit conversion
    m_e_SI = 9.1093837015e-31
    hbar_SI = 1.054571817e-34
    e_ele_SI = 1.602176634e-19
    epsilon_0_SI = 8.8541878128e-12
    # Values to multiply to convert a.u. to SI (or divide, from SI to a.u.)
    length_au2SI = 0.529177210903e-10
    time_au2SI = 2.4188843265857e-17
    freq_au2SI = 1 / time_au2SI
    
    # numbers that I haven't worked out are set to None
    c_light = 137.035999074      # speed of light (= 1/alpha)
    hbar = 1                     # reduced Planck constant
    hnobar = hbar * 2 * pi       # Planck constant
    k_B = None           # Boltzmann constant
    e_ele = 1                    # Charge of electrons
    m_e = 1                      # Mass of electrons
    epsilon_0 = 1 / (4 * pi)     # vacuum permittivity
    mu_0 = None      # vacuum permeability
    R_gas = None     # Ideal gass constant (exact)
    g_earth = None
    
    # Bohr stuff aka atomic units
    a0 = 1 # Bohr radius, = 4 * pi * epsilon_0 * hbar^2 / (m_e * e_ele**2)
    mu_B = 1/2                   # Bohr magneton, = e_ele * hbar / (2 * m_e)
    
    # isotope specific constants (from [SteckRb] and [TieckeK] unless otherwise noted)
    m_Rb87 = 1.4431606e-25 / m_e_SI
    m_K40 = 6.636178e-26 / m_e_SI
    m_K39 = 6.470076e-26 / m_e_SI
    I_Rb87 = 3/2
    I_K40 = 4
    I_K39 = 3/2
    gJ_K_4S1o2 = 2.00229421
    gJ_Rb_5S1o2 = 2.00233113
    # Hyperfine structure coefficients, ahf (M1) and bhf (E2), in J
    ahf_39K_4S1o2 = None#230.8598601e6 * hnobar
    ahf_40K_4S1o2 = None#-285.7308e6 * hnobar
    ahf_87Rb_5S1o2 = None#3.417341305452145e9 * hnobar
    # Line constants organized by species
        # Saturation intensities assume circularly polarized light
        # nu's and gamma's are in FREQUENCIES, not ANGULAR FREQUENCIES
    # 39K
    lambda_K39_D1 = 770.108385049e-9 / length_au2SI
    lambda_K39_D2 = 766.700921822e-9 / length_au2SI
    # 40K
    nu_K40_4_2P1o2 = 389.286184353e12 / freq_au2SI # 40K D1 line
    nu_K40_4_2P3o2 = 391.016296050e12 / freq_au2SI # 40K D2 line
    nu_K40_4_2P = (nu_K40_4_2P1o2 + nu_K40_4_2P3o2) / 2 # average of D1 & D2 line for convenience
    Delta_K40_4_2P = (nu_K40_4_2P3o2 - nu_K40_4_2P1o2) # diffference of D1 & D2 line for convenience
    lambda_K40_D1 = 770.108136507e-9 / length_au2SI
    lambda_K40_D2 = 766.700674872e-9 / length_au2SI
    gamma_40K_D2 = 6.035e6 / freq_au2SI
    # 87Rb
    nu_Rb87_4_2P1o2 = 377.107463380e12 / freq_au2SI # 87Rb D1 line
    nu_Rb87_4_2P3o2 = 384.2304844685e12 / freq_au2SI # 87Rb D2 line
    nu_Rb87_4_2P = (nu_Rb87_4_2P1o2 + nu_Rb87_4_2P3o2) / 2 # average of D1 & D2 line for convenience
    Delta_Rb87_4_2P = (nu_Rb87_4_2P3o2 - nu_Rb87_4_2P1o2) # diffference of D1 & D2 line for convenience
    lambda_Rb87_D2 = 780.241209686e-9 / length_au2SI
    gamma_Rb87_D2 = 6.0666e6 / freq_au2SI
    I_sat_Rb87_D2 = None
    
    # Non-Rb or K constants
    m_Li6 = 9.9883414e-27 / m_e_SI
else:
    raise('Unit system ' + str(unit_system) + ' not defined')

#%% lab constants
if unit_system == 'SI':
    # B field related
    FBcoil_coeff = 1.64 * 1e-4 * 1e2 # [Claire] in T/m/Ampere; determines B field near FB coil center when running a quadrupole field
    # optical lattice related
    lambda_sw = 532e-9
    lambda_lw = 1064e-9
    lambda_vert = 1064e-9
    # beam waists, taken from [Claire] and [TBarter] (I believe those "w"'s are actually diameters, i.e. w0*2; usually beam waist is called w0)
    w0_sw = 40e-6
    w0_lw = 50e-6
    w0_ODT = 50e-6
elif unit_system == 'a.u.':
    # B field related
    FBcoil_coeff = None#1.64 * 1e-4 * 1e2 # [Claire] in T/m/Ampere; determines B field near FB coil center when running a quadrupole field
    # optical lattice related
    lambda_sw = 532e-9 / length_au2SI
    lambda_lw = 1064e-9 / length_au2SI
    lambda_vert = 1064e-9 / length_au2SI
    # beam waists, taken from [Claire] and [TBarter] (I believe those "w"'s are actually diameters, i.e. w0*2; usually beam waist is called w0)
    w0_sw = 40e-6 / length_au2SI
    w0_lw = 50e-6 / length_au2SI
    w0_ODT = 50e-6 / length_au2SI
else:
    raise('Unit system ' + str(unit_system) + ' not defined')

# optical lattice related (derived)
a_sw_tri = lambda_sw * (2 / 3)
a_lw_hex = lambda_lw * (2 / 3 / np.sqrt(3))
a_vert = lambda_vert / 2
f_sw = c_light / lambda_sw
f_lw = c_light / lambda_lw
f_vert = c_light / lambda_vert
# recoil energies
E_R1064_Rb87 = hbar**2/2/m_Rb87*(2*pi/lambda_lw)**2 # 1064 photon recoil energy
E_R532_Rb87 = hbar**2/2/m_Rb87*(2*pi/lambda_sw)**2 # 532 photon recoil energy
E_R1064_K40 = hbar**2/2/m_K40*(2*pi/lambda_lw)**2 # 1064 photon recoil energy
E_R532_K40 = hbar**2/2/m_K40*(2*pi/lambda_sw)**2 # 532 photon recoil energy

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