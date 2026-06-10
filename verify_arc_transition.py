import numpy as np
import scipy.constants as const
from arc import Rubidium87, Potassium40, Potassium39, DynamicPolarizability
import E9_fn.E9_constants as E9c
import E9_fn.E9_atom as E9a
import E9_fn.polarizabilities_calculation as E9pol
import E9_fn.datasets.transition_line_data as TLData
import E9_fn.E9_cooltrap as E9ct

def verify_constants():
    print("--- Verifying Constants ---")
    
    # ARC Instances (Capitalized naming convention as requested)
    Rb87 = Rubidium87()
    K40 = Potassium40()
    
    # 1. Mass
    print(f"Rb87 Mass: manual={E9c.m_Rb87:.6e}, ARC={Rb87.mass:.6e}, Diff={(E9c.m_Rb87 - Rb87.mass)/E9c.m_Rb87:.2%}")
    print(f"K40 Mass:  manual={E9c.m_K40:.6e}, ARC={K40.mass:.6e}, Diff={(E9c.m_K40 - K40.mass)/E9c.m_K40:.2%}")
    
    # 2. Wavelengths (D2)
    # Rb87 D2: 5S1/2 -> 5P3/2
    wl_Rb87_d2_arc = Rb87.getTransitionWavelength(5, 0, 0.5, 5, 1, 1.5)
    print(f"Rb87 D2 Wavelength: manual={E9c.lambda_Rb87_D2:.6e}, ARC={wl_Rb87_d2_arc:.6e}, Diff={(E9c.lambda_Rb87_D2 - wl_Rb87_d2_arc)/E9c.lambda_Rb87_D2:.2%}")
    
    # K40 D2: 4S1/2 -> 4P3/2
    wl_K40_d2_arc = K40.getTransitionWavelength(4, 0, 0.5, 4, 1, 1.5)
    print(f"K40 D2 Wavelength:  manual={E9c.lambda_K40_D2:.6e}, ARC={wl_K40_d2_arc:.6e}, Diff={(E9c.lambda_K40_D2 - wl_K40_d2_arc)/E9c.lambda_K40_D2:.2%}")

    # 3. Linewidths (Gamma in rad/s)
    gamma_Rb87_d2_arc = Rb87.getTransitionRate(5, 1, 1.5, 5, 0, 0.5)
    print(f"Rb87 D2 Linewidth: manual={E9c.gamma_Rb87_D2:.6e}, ARC={gamma_Rb87_d2_arc:.6e}, Diff={(E9c.gamma_Rb87_D2 - gamma_Rb87_d2_arc)/E9c.gamma_Rb87_D2:.2%}")
    
    # 4. Hyperfine Constants (A in J)
    ahf_Rb87_arc_hz, _ = Rb87.getHFSCoefficients(5, 0, 0.5)
    ahf_Rb87_arc_j = ahf_Rb87_arc_hz * const.h
    print(f"Rb87 S1/2 ahf: manual={E9c.ahf_87Rb_5S1o2:.6e}, ARC={ahf_Rb87_arc_j:.6e}, Diff={(E9c.ahf_87Rb_5S1o2 - ahf_Rb87_arc_j)/E9c.ahf_87Rb_5S1o2:.2%}")
def verify_calculations():
    print("\n--- Verifying Calculations ---")
    from arc import DynamicPolarizability
    
    # 1. Polarizability for K40 4S1/2 at 1064nm
    lamb_test = 1064.0e-9 # m
    # Explicitly call manual function for baseline
    pol_K40_manual = E9pol.alpha_pol_manual(0, lamb_test, TLData.K_4S1o2_lines, '4S1o2')
    
    # ARC values using new refactored functions
    pol_K40_arc_val = E9pol.get_polarizability(lamb_test, 0, method="ARC", arc_atom=E9c.K40, n=4, l=0, j=0.5, include_core=False)
    pol_K40_arc_tot = E9pol.get_polarizability(lamb_test, 0, method="ARC", arc_atom=E9c.K40, n=4, l=0, j=0.5, include_core=True)

    print(f"K40 4S1/2 alpha_s (manual NIST sum) @ 1064nm: {pol_K40_manual:.6e}")
    print(f"K40 4S1/2 alpha_s (ARC valence)    @ 1064nm: {pol_K40_arc_val:.6e}, Diff={(pol_K40_manual - pol_K40_arc_val)/pol_K40_manual:.2%}")
    print(f"K40 4S1/2 alpha_s (ARC total)      @ 1064nm: {pol_K40_arc_tot:.6e}, Diff={(pol_K40_manual - pol_K40_arc_tot)/pol_K40_manual:.2%}")

    # 2. Trap Depth comparison for Rb87 in 1064nm ODT
    intensity = 1e8 # W/m^2
    depth_legacy = E9ct.V0_from_I(E9c.gamma_Rb87_D2, E9c.nu_Rb87_5_2P, const.c/1064e-9, intensity, E9c.gF(E9c.I_Rb87, 0.5, 1, E9c.gJ_Rb_5S1o2), 1)

    # New ARC-based depth
    depth_arc = E9ct.V0_from_I_arc(intensity, const.c/1064e-9, E9c.Rb87, 5, 0, 0.5, include_core=True)

    print(f"Rb87 Trap Depth @ 1064nm, 10^8 W/m^2: manual(approx)={depth_legacy:.6e}, ARC(exact+core)={depth_arc:.6e}, Diff={(depth_legacy - depth_arc)/depth_legacy:.2%}")



if __name__ == "__main__":
    verify_constants()
    verify_calculations()
