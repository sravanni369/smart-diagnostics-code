"""
Demonstration of issue C6-1 — a silent correctness bug in the published ESP code.

Source: Rahman et al., Appendix 2, printed p.134 (PDF p.141).

THE BUG
-------
The chapter builds its scaler like this:

    input_transformer = ColumnTransformer(transformers=[
        ('std',    StandardScaler(), ["Speed", "Flowrate"]),
        ('robust', RobustScaler(),   ["Viscosity"])
    ])
    input_columns = ["Speed", "Viscosity", "Flowrate"]

    training_input_df_norm = pd.DataFrame(
        input_transformer.fit_transform(training_input_df), columns=input_columns)

`ColumnTransformer` emits columns in *transformer* order, not in the order of the
original DataFrame. So the output columns are:

    position 0 -> Speed      (StandardScaler)
    position 1 -> Flowrate   (StandardScaler)
    position 2 -> Viscosity  (RobustScaler)

but they are then relabelled ["Speed", "Viscosity", "Flowrate"]. Flowrate is labelled
"Viscosity" and Viscosity is labelled "Flowrate".

WHY IT MATTERS
--------------
Nothing raises. Shapes match, training runs, the loss goes down. The mislabelling is
consistent across train/validation/test because the same transformer is reused, so the
network still learns a usable mapping. The damage is to interpretation: any statement
about "the effect of viscosity" made by reading these columns is actually a statement
about flowrate, and vice versa. It also silently changes which variable receives robust
vs standard scaling relative to what the text describes.

This is worth flagging because the chapter reports (printed p.133 / PDF p.140) an
unexplained result: "the unexpected sensitivity of the MLP model to the output
parameter normalisation remain unexplored and unexplained". Column handling of exactly
this kind is the first place to look. I am NOT claiming this bug causes that
observation — I cannot test that without their dataset. I am flagging it as a
candidate worth checking.

THE FIX
-------
Ask the transformer what it produced instead of assuming:
    columns=input_transformer.get_feature_names_out()
or list the transformers in the same order as `input_columns`, or set
`ColumnTransformer(..., verbose_feature_names_out=False)` and reindex.

Run:  python column_transformer_bug.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

INPUT_COLUMNS = ["Speed", "Viscosity", "Flowrate"]


def sample_frame() -> pd.DataFrame:
    """Deliberately distinct magnitudes so mislabelling is impossible to miss."""
    return pd.DataFrame({
        "Speed":     [1800.0, 2400.0, 3000.0, 3500.0],   # order 1e3
        "Viscosity": [25.0, 107.0, 240.0, 520.0],        # order 1e2
        "Flowrate":  [1763.0, 1037.0, 2247.0, 1714.0],   # order 1e3
    })


def build_transformer() -> ColumnTransformer:
    """Exactly as printed on p.134."""
    return ColumnTransformer(transformers=[
        ("std", StandardScaler(), ["Speed", "Flowrate"]),
        ("robust", RobustScaler(), ["Viscosity"]),
    ])


def main() -> None:
    print("=" * 78)
    print("Issue C6-1 — ColumnTransformer silently reorders columns")
    print("Source: Rahman et al., printed p.134 (PDF p.141)")
    print("=" * 78)

    df = sample_frame()
    print("\nOriginal input frame:")
    print(df.to_string(index=False))

    transformer = build_transformer()
    raw = transformer.fit_transform(df[INPUT_COLUMNS])

    print(f"\nTransformer emits columns in this order:")
    print(f"  {list(transformer.get_feature_names_out())}")
    print(f"But the book relabels them as:")
    print(f"  {INPUT_COLUMNS}")

    # --- the book's version -----------------------------------------------------
    book = pd.DataFrame(raw, columns=INPUT_COLUMNS)
    print("\n--- AS PUBLISHED (mislabelled) ---")
    print(book.to_string(index=False))

    # --- the corrected version --------------------------------------------------
    correct = pd.DataFrame(raw, columns=transformer.get_feature_names_out())
    correct.columns = [c.split("__")[-1] for c in correct.columns]
    correct = correct[INPUT_COLUMNS]          # reorder back to the intended order
    print("\n--- CORRECTED (labels follow the transformer) ---")
    print(correct.to_string(index=False))

    # --- prove the mismatch -----------------------------------------------------
    print("\n" + "-" * 78)
    print("Proof — correlate each labelled column against the true source column:")
    print(f"{'label':<12}{'book column tracks':<22}{'correlation':>12}")
    for label in INPUT_COLUMNS:
        best, best_r = None, 0.0
        for source in INPUT_COLUMNS:
            r = abs(np.corrcoef(book[label], df[source])[0, 1])
            if r > best_r:
                best, best_r = source, r
        flag = "OK" if best == label else "<-- MISLABELLED"
        print(f"{label:<12}{best:<22}{best_r:>12.4f}   {flag}")

    swapped = not np.allclose(book["Viscosity"], correct["Viscosity"])
    print(f"\nBook's 'Viscosity' column differs from the true Viscosity: {swapped}")
    print("Conclusion: 'Viscosity' and 'Flowrate' are transposed in the published code.")


if __name__ == "__main__":
    main()
