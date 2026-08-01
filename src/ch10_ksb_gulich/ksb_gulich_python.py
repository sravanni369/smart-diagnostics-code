"""
Python port of the KSB and Gulich viscosity-correction models.

Ported from the MATLAB in book_original.m
  "Parameter Identification of Empirical Models for Head Estimation of an
   Electrical Submersible Pump Handling Viscous Fluids", H. Hadabi et al.
  Printed pp. 277-280 (PDF pp. 284-287); reference data pp. 281 (PDF 288).

WHY THIS ONE IS SPECIAL
-----------------------
The other two chapters do not ship their datasets, so they can only run on synthetic
data. This chapter is different: the book prints both its model inputs AND the
correction factors it computed from them (printed p.281 / PDF p.288). That makes this
port fully verifiable, and `validate()` checks all 64 published values.

Result: all 64 match to 4 decimal places, which is the precision the book prints.

MATLAB -> Python notes:
  * MATLAB `log` is natural log -> np.log;  `log10` -> np.log10.
  * `^` on scalars -> `**`.
  * 1-based indexing is irrelevant here (no indexing in the model core).
  * The model core is vectorised over operating points instead of a `for i = 1:height(T)`
    loop. The arithmetic is unchanged.

Run:  python ksb_gulich_python.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- physical constants, exactly as in the MATLAB -------------------------------------
BPD_TO_M3S = 0.00000184013   # barrels per day -> m^3/s
G = 9.81                     # m/s^2
R_IMPELLER = 0.056           # m
FT_TO_M = 0.3048
WATER_CP = 0.898             # cP, reference viscosity

# Initial / "original" empirical parameters, from the book's theta0
# (KSB a..f, then Gulich a_g, b_g, c_g)
ORIGINAL_PARAMS = np.array(
    [10.41, 0.039, 0.113, 5.323, 0.390, 0.643, 0.507, 12.09, 0.709]
)

# Optimised parameter sets, book printed p.283 / PDF p.290 ("full dataset")
NELDER_MEAD_FULL = np.array([0.125, 0.0776, 5.466, 3.466, 0.337, 0.682, 2.319, 6.357, 0.664])
LEVENBERG_MARQUARDT_FULL = np.array([1.270, 0.0420, 0.631, 9.999, 0.355, 0.656, 2.000, 6.762, 0.660])


# --- reference data, transcribed from the book ----------------------------------------
# Operating points, printed p.281 / PDF p.288 (first table)
OPERATING_POINTS = pd.DataFrame(
    [
        (1800, 25, 0.5853, 1388.6, 13.80, 1763.314),
        (1800, 120, 0.5853, 1388.6, 13.80, 777.9429),
        (1800, 240, 0.5853, 1388.6, 13.80, 1348.457),
        (1800, 520, 0.5853, 1388.6, 13.80, 881.4857),
        (2400, 63, 0.5854, 1851.4, 24.54, 2350.971),
        (2400, 107, 0.5854, 1851.4, 24.54, 1037.143),
        (2400, 240, 0.5854, 1851.4, 24.54, 1797.943),
        (2400, 380, 0.5854, 1851.4, 24.54, 1175.657),
        (3000, 25, 0.5851, 2314.3, 38.31, 2938.629),
        (3000, 85, 0.5851, 2314.3, 38.31, 1296.686),
        (3000, 105, 0.5851, 2314.3, 38.31, 2247.429),
        (3000, 170, 0.5851, 2314.3, 38.31, 1469.486),
        (3500, 90, 0.5785, 2700.0, 52.17, 3428.571),
        (3500, 120, 0.5785, 2700.0, 52.17, 1512.686),
        (3500, 130, 0.5785, 2700.0, 52.17, 2621.829),
        (3500, 250, 0.5785, 2700.0, 52.17, 1714.286),
    ],
    columns=["RPM", "Viscosity_cP", "omega_s", "Q_BEP_BPD", "H_BEP_ft", "Q_BPD"],
)

# Correction factors the book reports for ORIGINAL_PARAMS, same page, second table.
PUBLISHED_CORRECTION_FACTORS = pd.DataFrame(
    [
        (0.8329, 0.9043, 0.8856, 0.8631),
        (0.6250, 0.8626, 0.7337, 0.8275),
        (0.4810, 0.7052, 0.6294, 0.6374),
        (0.3005, 0.6765, 0.4879, 0.6358),
        (0.7676, 0.8544, 0.8367, 0.8047),
        (0.6941, 0.8964, 0.7832, 0.8596),
        (0.5442, 0.7453, 0.6756, 0.6826),
        (0.4415, 0.7527, 0.6000, 0.7154),
        (0.8711, 0.9340, 0.9148, 0.8981),
        (0.7582, 0.9267, 0.8297, 0.8897),
        (0.7300, 0.8630, 0.8092, 0.8134),
        (0.6543, 0.8658, 0.7546, 0.8255),
        (0.7707, 0.8567, 0.8380, 0.8062),
        (0.7335, 0.9153, 0.8107, 0.8774),
        (0.7221, 0.8580, 0.8025, 0.8068),
        (0.6124, 0.8441, 0.7236, 0.8034),
    ],
    columns=["CQ_KSB", "CH_KSB", "CQ_Gulich", "CH_Gulich"],
)


def correction_factors(points: pd.DataFrame, theta=ORIGINAL_PARAMS) -> pd.DataFrame:
    """Compute KSB and Gulich flow/head correction factors.

    Direct translation of the KSB/Gulich blocks in `modelError` / `modelResiduals`.

    Args:
        points: operating points with columns RPM, Viscosity_cP, omega_s,
                Q_BEP_BPD, H_BEP_ft, Q_BPD.
        theta:  9 empirical parameters [a, b, c, d, e, f, a_g, b_g, c_g].

    Returns:
        DataFrame with CQ_KSB, CH_KSB, CQ_Gulich, CH_Gulich.
    """
    a, b, c, d, e, f, a_g, b_g, c_g = theta

    visc = points["Viscosity_cP"].to_numpy(float)
    omega_s = points["omega_s"].to_numpy(float)
    rpm = points["RPM"].to_numpy(float)

    # unit conversions, as in the MATLAB
    nu = (visc / WATER_CP) * 1e-6                       # m^2/s
    omega = 2.0 * np.pi * rpm / 60.0                    # rad/s
    q_bep = points["Q_BEP_BPD"].to_numpy(float) * BPD_TO_M3S
    h_bep = points["H_BEP_ft"].to_numpy(float) * FT_TO_M
    q_w = points["Q_BPD"].to_numpy(float) * BPD_TO_M3S

    # --- KSB model ---
    b_hi = (480.0 * np.sqrt(nu)) / (q_bep**0.25 * (G * h_bep) ** 0.125)
    b_ksb = np.exp(np.log(b_hi) + 0.5 * (np.log(a) - np.log(52.933) - np.log(omega_s)))
    cq_ksb = np.exp(b * b_ksb * np.log(a / (52.933 * omega_s)) - c * (np.log10(b_ksb)) ** d)
    xi = 1.0 - 0.014 * (b_hi - 1.0) * ((q_w / q_bep) - 1.0)
    ch_ksb = (e + f * cq_ksb) * xi

    # --- Gulich model ---
    re_omega = (omega * R_IMPELLER**2) / nu
    re_g = re_omega * omega_s**a_g
    x = -np.exp(np.log(b_g) - c_g * (np.log(re_omega) + a_g * np.log(omega_s)))
    ch_bep = re_g**x
    cq_g = ch_bep
    ch_g = 1.0 - (1.0 - ch_bep) * (q_w / q_bep) ** 0.75

    return pd.DataFrame(
        {"CQ_KSB": cq_ksb, "CH_KSB": ch_ksb, "CQ_Gulich": cq_g, "CH_Gulich": ch_g}
    )


def validate(tol: float = 5e-5) -> tuple[bool, pd.DataFrame]:
    """Check the port against the book's published correction factors.

    Returns (all_passed, comparison_table).
    """
    computed = correction_factors(OPERATING_POINTS, ORIGINAL_PARAMS)
    published = PUBLISHED_CORRECTION_FACTORS

    comparison = OPERATING_POINTS[["RPM", "Viscosity_cP"]].copy()
    for col in published.columns:
        comparison[f"{col}_computed"] = computed[col].round(4)
        comparison[f"{col}_book"] = published[col]
        comparison[f"{col}_absdiff"] = (computed[col] - published[col]).abs()

    diff_cols = [c for c in comparison.columns if c.endswith("_absdiff")]
    max_diff = comparison[diff_cols].to_numpy().max()
    return bool(max_diff <= tol), comparison


def main() -> None:
    print("=" * 78)
    print("KSB / Gulich viscosity correction — MATLAB -> Python port")
    print("Source: Hadabi et al., printed pp. 277-281 (PDF pp. 284-288)")
    print("=" * 78)

    passed, comparison = validate()

    print("\nComputed vs published correction factors (original parameters):\n")
    show = comparison[
        ["RPM", "Viscosity_cP",
         "CQ_KSB_computed", "CQ_KSB_book",
         "CH_KSB_computed", "CH_KSB_book",
         "CQ_Gulich_computed", "CQ_Gulich_book",
         "CH_Gulich_computed", "CH_Gulich_book"]
    ]
    print(show.to_string(index=False))

    diff_cols = [c for c in comparison.columns if c.endswith("_absdiff")]
    max_diff = comparison[diff_cols].to_numpy().max()
    n_values = len(comparison) * 4

    print(f"\nValues compared        : {n_values}")
    print(f"Max absolute deviation : {max_diff:.2e}")
    print(f"VALIDATION             : {'PASS' if passed else 'FAIL'}")
    if passed:
        print("\nAll published values reproduced to the 4 decimal places the book prints.")
        print("This is a real reproduction of the authors' results, not synthetic data.")

    # Also show what the optimised parameter sets do to the factors.
    # NOTE: means are printed only when the set is fully evaluable. pandas skips NaN
    # silently, so averaging a partly-NaN column would quietly describe a subset.
    print("\n" + "-" * 78)
    print("Correction factors under the book's optimised parameters (PDF p.290):")
    for name, params in [
        ("Nelder-Mead", NELDER_MEAD_FULL),
        ("Levenberg-Marquardt", LEVENBERG_MARQUARDT_FULL),
    ]:
        with np.errstate(invalid="ignore"):
            cf = correction_factors(OPERATING_POINTS, params)
        n_nan = int(cf.isna().sum().sum())
        if n_nan:
            bad = ", ".join(cf.columns[cf.isna().any()])
            print(f"  {name:<22} NOT EVALUABLE — {n_nan} NaN values in [{bad}]")
        else:
            print(f"  {name:<22} mean CH_KSB={cf['CH_KSB'].mean():.4f}  "
                  f"mean CH_Gulich={cf['CH_Gulich'].mean():.4f}")

    if int(correction_factors(OPERATING_POINTS, NELDER_MEAD_FULL).isna().sum().sum()):
        print("\nSee issue C10-1 in ISSUES_FOUND.md: with the published Nelder-Mead")
        print("parameters (a=0.125) the KSB term B falls below 1 for 11 of the 16")
        print("operating points, so log10(B) < 0 and (log10(B))**d with fractional d")
        print("leaves the reals. numpy yields NaN; MATLAB silently yields complex.")


if __name__ == "__main__":
    main()
