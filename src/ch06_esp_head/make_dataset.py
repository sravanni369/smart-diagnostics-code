"""
SYNTHETIC dataset generator for the ESP head-estimation chapter.

The chapter reads a local "dataset.csv" that the book does not distribute, so the
published pipeline cannot be run on the authors' data.

*** THE DATA IS INVENTED. Downstream MAE/MAPE numbers describe this generator,
*** NOT the authors' pump. The book reports MAE < 0.6 ft and MAPE < 8%
*** (printed p.133); do not compare.

It is, however, generated from physics rather than noise, using two facts taken
from the book itself:

1. The book's own BEP table (printed p.281 / PDF p.288) obeys the pump affinity laws
   exactly:
        Q_BEP proportional to N      1388.6 * (3500/1800) = 2700.0   (book: 2700)
        H_BEP proportional to N^2    13.80  * (3500/1800)^2 = 52.19  (book: 52.17)
   so BEP is extrapolated from the 1800 rpm anchor.

2. Viscous degradation is applied with the KSB correction factor from the
   ksb_gulich_python module — the port that reproduces the book's published
   correction factors exactly. So the viscosity effect here is the book's own model.

Schema matches what the chapter's code expects: Speed, Viscosity, Flowrate, Head.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "ch10_ksb_gulich"))
from ksb_gulich_python import ORIGINAL_PARAMS, correction_factors  # noqa: E402

SEED = 1746448054           # the chapter's own rnd_seed
N_SAMPLES = 1200

# anchor from the book's BEP table at 1800 rpm
N_REF = 1800.0
Q_BEP_REF = 1388.6          # BPD
H_BEP_REF = 13.80           # ft
OMEGA_S = 0.5853

SPEEDS = [1800, 2400, 3000, 3500]
VISCOSITIES = [25, 63, 85, 90, 105, 107, 120, 130, 170, 240, 250, 380, 520]


def bep(speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Affinity laws: Q ~ N, H ~ N^2."""
    ratio = speed / N_REF
    return Q_BEP_REF * ratio, H_BEP_REF * ratio**2


def water_head_shape(q_norm: np.ndarray) -> np.ndarray:
    """Normalised water head curve: h(1) = 1, rising to ~1.25 at shutoff."""
    return 1.25 - 0.25 * q_norm**2


def generate(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    speed = rng.choice(SPEEDS, size=n_samples).astype(float)
    visc = rng.choice(VISCOSITIES, size=n_samples).astype(float)

    q_bep, h_bep = bep(speed)
    q_norm = rng.uniform(0.25, 1.45, size=n_samples)     # operate around BEP
    flowrate = q_norm * q_bep                            # BPD

    # viscous correction from the validated KSB model
    points = pd.DataFrame({
        "RPM": speed,
        "Viscosity_cP": visc,
        "omega_s": OMEGA_S,
        "Q_BEP_BPD": q_bep,
        "H_BEP_ft": h_bep,
        "Q_BPD": flowrate,
    })
    ch = correction_factors(points, ORIGINAL_PARAMS)["CH_KSB"].to_numpy()

    head = h_bep * water_head_shape(q_norm) * ch
    head *= 1.0 + rng.normal(0.0, 0.01, size=n_samples)  # 1% measurement noise
    head = np.clip(head, 0.05, None)

    return pd.DataFrame({
        "Speed": speed,
        "Viscosity": visc,
        "Flowrate": flowrate.round(3),
        "Head": head.round(4),
    })


def main() -> None:
    df = generate()
    out = ROOT / "data" / "esp_head_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"SYNTHETIC ESP dataset written to: {out}")
    print(f"  rows={len(df)}")
    print("\nRanges:")
    print(df.describe().loc[["min", "max", "mean"]].to_string())
    print("\nMean head by speed (should rise ~ N^2):")
    print(df.groupby("Speed")["Head"].mean().round(2).to_string())
    print("\nMean head by viscosity (should fall as viscosity rises):")
    print(df.groupby("Viscosity")["Head"].mean().round(2).to_string())
    print("\nREMINDER: invented data, physics-shaped. Not the authors' measurements.")


if __name__ == "__main__":
    main()
