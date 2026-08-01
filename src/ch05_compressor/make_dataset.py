"""
SYNTHETIC dataset generator for the reciprocating-compressor chapter.

The chapter's own data file is a Google Drive path
('/content/drive/MyDrive/Thesis/data (1).csv') and is not distributed with the book,
so it cannot be used here. This module fabricates data matching the schema the chapter
describes so the published pipeline can actually execute.

*** THE DATA IS INVENTED. Any accuracy/recall number produced downstream describes
*** this generator, NOT the authors' compressor and NOT their published results.
*** The book reports 99% macro F1 and 100% recall on real data (printed p.97).
*** Do not compare the two.

Schema, from the chapter text:
  - 20 numeric sensor features (the model's Input(shape=(20,)))
  - 4 component condition labels, each binary:
        bearings  Ok/Noisy      wpump     Ok/Noisy
        radiator  Clean/Dirty   exvalve   Clean/Dirty
  - the chapter notes faults are roughly 20% of the data (printed p.93)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 1697639          # the chapter's own random_state
N_SAMPLES = 1000
N_FEATURES = 20
FAULT_RATE = 0.20       # "they only account for 20% of the total data"

SENSOR_NAMES = [
    "vib_rms_de", "vib_rms_nde", "vib_peak_de", "vib_kurtosis", "vib_crest",
    "temp_bearing_de", "temp_bearing_nde", "temp_discharge", "temp_suction", "temp_oil",
    "press_suction", "press_discharge", "press_ratio", "press_pulsation",
    "flow_rate", "motor_current", "motor_power", "rpm", "oil_level", "humidity",
]

LABELS = {
    "bearings": ("Ok", "Noisy"),
    "wpump": ("Ok", "Noisy"),
    "radiator": ("Clean", "Dirty"),
    "exvalve": ("Clean", "Dirty"),
}

# Which sensors each fault perturbs, so labels are actually learnable from features.
FAULT_SIGNATURE = {
    "bearings": [0, 1, 2, 3, 4, 5, 6],
    "wpump": [8, 9, 14, 19],
    "radiator": [5, 6, 7, 9, 19],
    "exvalve": [10, 11, 12, 13, 14],
}


def generate(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # healthy baseline: standardised sensor readings
    x = rng.normal(0.0, 1.0, size=(n_samples, N_FEATURES))
    frame = {}

    for label, (ok_word, bad_word) in LABELS.items():
        faulty = rng.random(n_samples) < FAULT_RATE
        # push the signature sensors when the component is faulty
        for col in FAULT_SIGNATURE[label]:
            shift = rng.uniform(1.4, 2.4)
            x[faulty, col] += shift
            x[faulty, col] *= rng.uniform(1.1, 1.5)
        frame[label] = np.where(faulty, bad_word, ok_word)

    df = pd.DataFrame(x, columns=SENSOR_NAMES)
    for label, values in frame.items():
        df[label] = values
    return df


def main() -> None:
    from pathlib import Path

    df = generate()
    out = Path(__file__).resolve().parents[2] / "data" / "compressor_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"SYNTHETIC compressor dataset written to: {out}")
    print(f"  rows={len(df)}  numeric features={len(SENSOR_NAMES)}  labels={list(LABELS)}")
    print("\nFault prevalence (should be near 20%):")
    for label, (ok_word, bad_word) in LABELS.items():
        rate = (df[label] == bad_word).mean()
        print(f"  {label:<10} {bad_word:<6} {rate:6.1%}")
    print("\nREMINDER: invented data. Downstream metrics are not the book's results.")


if __name__ == "__main__":
    main()
