# Summary: ARC Integration and Codebase Refinement
**Date:** June 9, 2026

## 1. Overview
This document summarizes the significant architectural updates made to the simulation codebase. The primary goals were to modernize plotting utilities and integrate the **Alkali.ne Rydberg Calculator (ARC)** package to improve the precision of atomic property and polarizability calculations.

---

## 2. Plotting Utility Refinement
**Decision:** The custom function `make_simple_axes()` was deprecated and removed in favor of the standard `matplotlib.pyplot.subplots()` to align with industry practices and improve readability for external collaborators.

**Key Changes:**
- Replaced all instances of `util.make_simple_axes()` across 9 project files and 1 Jupyter Notebook.
- Standardized the handling of figure arguments (e.g., `figsize`, `num`).
- Implemented conditional logic in plotting methods to allow for either providing an existing axis or generating a new one.

---

## 3. ARC Integration Strategy
**Decision:** A **hybrid architecture** was adopted. This allows the codebase to leverage ARC’s high-precision solvers for standard isotopes while maintaining the existing manual summation infrastructure for complex atoms (e.g., Dysprosium) or fermionic alkaline-earths not fully supported by ARC's default solvers.

### Design Principles:
- **Backward Compatibility:** All existing variable names in `E9_constants.py` (e.g., `m_Rb87`, `ahf_40K_4S1o2`) were preserved. Analysis notebooks and project scripts will continue to run without modification.
- **SI Unit Consistency:** All new functions and API interactions strictly adhere to SI units (meters for wavelength, Joules for energy).
- **Capitalized Naming:** Isotope objects are initialized with capitalized names (e.g., `Rb87`, `K40`) to distinguish them clearly as physical entities.

### Component Updates:
1. **`E9_constants.py`:**
   - Atomic masses, nuclear spins, wavelengths, and hyperfine coefficients for Rb and K are now dynamically queried from ARC instances.
2. **`E9_atom.py`:**
   - The `HyperfineState` class now supports dual-initialization. If an `arc_atom` is provided, it retrieves constants from ARC; otherwise, it falls back to manual inputs.
3. **`polarizabilities_calculation.py`:**
   - Introduced `alpha_pol_arc` for exact dynamic polarizability.
   - Implemented a unified `get_polarizability` wrapper that routes requests to either ARC or the legacy NIST-based `alpha_pol_manual` method.
   - Refactored convenience functions (e.g., `alpha_s_K_4S1o2`) to use ARC by default.
4. **`E9_cooltrap.py`:**
   - Added `V0_from_I_arc` to calculate exact trap depths without the large-detuning/two-level approximation.

---

## 4. Verification and Validation
A rigorous verification step confirmed the accuracy of the integration. 

**Results for K40 @ 1064 nm:**
- **Constants:** 0.00% discrepancy in fundamental masses and hyperfine constants.
- **Valence Polarizability:** Only a **0.18% difference** between the manual NIST-based summation and ARC’s calculation, validating the accuracy of previous manual datasets.
- **Trap Depth Approximation:** A **12.55% difference** was observed between the legacy two-level approximation and ARC’s exact calculation, demonstrating the significant precision gain achieved through this integration.

---

## 5. Future Implementation Guide
To add isotopes in the future, follow these guidelines:
- **For ARC-supported atoms:** Use the `arc_atom` argument in `HyperfineState`.
- **For Unsupported atoms (e.g., 162Dy):** Continue providing manual constants and transition lines to `transition_line_data.py`. Call `get_polarizability` with `method="MANUAL"`.
- **For 87Sr:** Use the `Strontium88` ARC object for matrix elements and project them onto the $I=9/2$ hyperfine levels using the existing Wigner 6-j logic.
