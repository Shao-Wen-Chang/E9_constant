# Polarizability Discrepancy Summary: Manual vs. ARC

This document summarizes the investigation into the slight discrepancies observed between "Manual" (NIST-based) and "ARC" (Alkali.ne Rydberg Calculator) calculations for vector polarizabilities, as reported in the project's comparison notebooks.

## Context
For Alkali atoms in their ground S-state ($L=0, J=1/2$), the scalar polarizability ($\alpha_s$) is large and dominated by the D-line transitions. The vector polarizability ($\alpha_v$), however, arises from the difference between the $D_1$ ($J'=1/2$) and $D_2$ ($J'=3/2$) contributions. Because these contributions have opposite signs and similar magnitudes, they nearly cancel, making $\alpha_v$ extremely sensitive to small variations in atomic data.

## Key Findings

### 1. Source of Discrepancy: D-Line Ratios
The primary cause of the discrepancy is the difference in the ratio of reduced matrix elements (or oscillator strengths $f$) between the $D_1$ and $D_2$ lines.
- **NIST (Manual):** Uses experimental values. For Potassium (K), the ratio $f_{D2}/f_{D1} \approx 2.0063$.
- **ARC:** Uses a model potential. For Potassium, the ratio $f_{D2}/f_{D1} \approx 2.0007$.
- This $0.3\%$ difference in the ratio is significantly amplified in $\alpha_v$ due to the cancellation effect.

### 2. Sensitivity to Fine Structure Splitting
The magnitude of $\alpha_v$ is inversely proportional to the fine structure splitting at large detunings.
- **Potassium (K):** Small splitting ($\sim 57 \text{ cm}^{-1}$). The cancellation is more "perfect," making the result highly sensitive. Discrepancy at 1064 nm is $\sim 4.5\%$.
- **Rubidium (Rb):** Larger splitting ($\sim 237 \text{ cm}^{-1}$). The sensitivity is reduced. Discrepancy at 1064 nm is much smaller ($\sim 0.9\%$).

### 3. Verification of Implementation
The re-implementation of ARC's logic in `E9_fn/polarizabilities_calculation.py` (`alpha_pol_arc`) was verified against ARC's internal `getPolarizability` method. The results matched exactly (ratio 1.000000), confirming that the discrepancy is not due to a programming error but rather the underlying atomic data.

### 4. Line List Completeness
The manual line list for Potassium was found to be missing the $n=10$ transitions ($4S \to 10P$). Adding these transitions changed the result by only $\sim 0.0006\%$, confirming that higher-lying states contribute negligible amounts to the discrepancy compared to the D-line parameters.

## Conclusion
The observed differences are physically expected when comparing experimental data (NIST) to model-based calculations (ARC). NIST data is generally preferred for transition strengths of low-lying states (AAA rating), while ARC is superior for calculating contributions from a complete set of Rydberg states or at very small detunings where its internally consistent potential is beneficial.

## Recommendations
- For high-precision vector light shift calculations in Potassium, use the manual method if the most accurate D-line ratios are required.
- The `E9_fn/datasets/transition_line_data.py` file should be updated to include the $n=10$ lines for completeness.
