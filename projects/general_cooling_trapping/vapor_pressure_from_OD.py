import numpy as np
from dataclasses import dataclass

@dataclass
class RBLine:
    name: str
    wavelength_m: float      # vacuum wavelength
    gamma_fwhm_hz: float     # natural linewidth FWHM in Hz
    f_osc: float             # absorption oscillator strength (not used in calc, kept for reference)

# Physical constants (SI)
k_B  = 1.380649e-23        # Boltzmann constant [J/K]
c    = 299_792_458.0       # speed of light [m/s]
amu  = 1.660_539_066_60e-27  # atomic mass unit [kg]
TORR_PER_PA = 1.0/133.322
MBAR_PER_PA = 1e-2

# Rubidium constants (from Steck data sheets)
RB85_MASS_KG = 84.911789 * amu
RB87_MASS_KG = 86.909183 * amu

D2 = RBLine(
    name="D2",
    wavelength_m=780.241_368_271e-9,   # m
    gamma_fwhm_hz=6.0666e6,            # Hz
    f_osc=0.69577
)
D1 = RBLine(
    name="D1",
    wavelength_m=794.979_014_933e-9,   # m
    gamma_fwhm_hz=5.7500e6,            # Hz
    f_osc=0.34231
)

# Natural-abundance isotope fractions
ISO_FRAC = {
    "85": 0.7217,
    "87": 0.2783,
    "mix": 1.0,   # if you somehow measure a feature representing both isotopes simultaneously (rare for Rb D-lines)
}

# Isotope masses for Doppler width
ISO_MASS = {
    "85": RB85_MASS_KG,
    "87": RB87_MASS_KG,
    "mix": (RB85_MASS_KG*ISO_FRAC["85"] + RB87_MASS_KG*ISO_FRAC["87"]),  # not usually used
}

def doppler_fwhm_hz(nu0_hz: float, T_K: float, mass_kg: float) -> float:
    """
    Thermal Doppler FWHM (Hz):
      Δν_FWHM = ν0 * sqrt( 8 k_B T ln2 / (m c^2) )
    """
    return nu0_hz * np.sqrt(8.0 * k_B * T_K * np.log(2.0) / (mass_kg * c**2))

def sigma0_two_level(lambda_m: float) -> float:
    """
    Resonant scattering cross-section at line center for a closed two-level transition
    (natural-width limited, weak probe):
      σ0 = 3 λ^2 / (2π)
    """
    return 3.0 * lambda_m**2 / (2.0 * np.pi)

def sigma_peak_doppler(lambda_m: float, gamma_fwhm_hz: float, doppler_fwhm_hz: float) -> float:
    """
    Peak absorption cross-section at line center for a Doppler-broadened line in the
    common regime Δν_D >> Γ:
      σ_peak ≈ σ0 * Γ * sqrt(π ln 2) / Δν_D

    where Γ is the natural FWHM (Hz) and Δν_D is the Doppler FWHM (Hz).
    """
    sigma0 = sigma0_two_level(lambda_m)
    return sigma0 * gamma_fwhm_hz * np.sqrt(np.pi * np.log(2.0)) / doppler_fwhm_hz

def rb_params(line: str = "D2"):
    if line.upper() == "D1":
        return D1
    return D2

def rb87_internal_factor(addressed_F=2, probe_pol='isotropic', uniform_ground=True):
    """
    addressed_F: 2 if you use the main 87Rb D2 peak (F=2→F'=3), else 1 for the F=1 group.
    probe_pol: 'isotropic' (unpolarized/unknown), 'linear', or 'circular_pumped'.
    uniform_ground: True to assume uniform population over F=1 and F=2 and all mF.

    Returns multiplicative factor f_internal to scale σ_peak (two-level) to the ensemble-averaged value.
    """
    # Fraction of atoms in the addressed F (uniform over F=1 and F=2)
    f_pop = ((2*addressed_F + 1) / (3 + 5)) if uniform_ground else 1.0

    # Polarization/mF averaging, using saturation-intensity ratios for 87Rb D2 F=2→F'=3 (Steck)
    if probe_pol == 'circular_pumped':
        f_pol = 1.0               # cycling two-level limit used in the original code
    elif probe_pol == 'linear':
        f_pol = 1.66933 / 3.05381 # ≈ 0.547
    else:  # 'isotropic' (default)
        f_pol = 1.66933 / 3.57710 # ≈ 0.467

    return f_pop * f_pol

def pressure_from_OD(
    OD_peak: float,
    L_m: float,
    T_K: float,
    line: str = "D2",
    isotope: str = "85",
    isotope_fraction: float | None = None
):
    """
    Convert on-resonance OD (peak absorption when scanning through the full Doppler feature)
    to total Rb vapor pressure.

    Assumptions:
      - Weak probe (no power broadening, no saturation).
      - You used the maximum absorption at line center of ONE isotope group.
      - Beer–Lambert: OD = n_isotope * σ_peak * L;   n_total = n_isotope / (isotope_fraction).
    """
    isotope = isotope if isotope in ISO_MASS else "85"
    frac = ISO_FRAC[isotope] if isotope_fraction is None else isotope_fraction
    if not (0 < frac <= 1.0):
        raise ValueError("isotope_fraction must be in (0, 1].")

    line_params = rb_params(line)
    nu0 = c / line_params.wavelength_m
    dnu_D = doppler_fwhm_hz(nu0, T_K, ISO_MASS[isotope])
    sigma_peak = sigma_peak_doppler(line_params.wavelength_m, line_params.gamma_fwhm_hz, dnu_D)
    # Example defaults for 87Rb D2, peak at F=2→F'=3 with uniform ground-state population:
    f_internal = rb87_internal_factor(addressed_F=2, probe_pol='isotropic', uniform_ground=True)
    sigma_peak *= f_internal


    # number density of the measured isotope, then total number density
    n_isotope = OD_peak / (sigma_peak * L_m)
    n_total = n_isotope / frac

    # ideal gas
    P_Pa = n_total * k_B * T_K
    return {
        "P_Pa": P_Pa,
        "P_Torr": P_Pa * TORR_PER_PA,
        "P_mbar": P_Pa * MBAR_PER_PA,
        "n_total_m3": n_total,
        "sigma_peak_m2": sigma_peak,
        "doppler_fwhm_MHz": dnu_D / 1e6,
        "nu0_THz": nu0 / 1e12,
        "lambda_nm": line_params.wavelength_m * 1e9
    }

# ----------------- Example -----------------
if __name__ == "__main__":
    result = pressure_from_OD(
        OD_peak=-np.log(179.7/208.9),
        L_m=32e-3 / np.cos(np.pi / 6),
        T_K=330.0,
        line="D2",
        isotope="87"        # use "87" if you locked to 87Rb peak; pass isotope_fraction=1.0 for enriched cells
        # isotope_fraction=None  # leave None to use natural abundance of chosen isotope
    )
    for k, v in result.items():
        print(f"{k}: {v}")
