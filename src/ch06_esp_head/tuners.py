"""
Hyperparameter tuners for the ESP head models (MLP and FCC).

Source: Appendix 4, printed pp. 136-137 (PDF pp. 143-144), Rahman et al.

`build_model_mlp` and `build_model_fcc` are exactly as printed. The chapter compares
RandomSearch against BayesianOptimization (printed p.127: "the best validation MAEs
were discovered by random search algorithm"), so both are run here.

Documented deviation:
  C6-4  BUDGET REDUCED to 10 trials / 40 epochs by default so the run finishes in
        minutes. Override with --trials / --epochs.

DATA IS SYNTHETIC.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quiet_tf  # noqa: F401,E402  — must precede tensorflow; see src/quiet_tf.py

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import HyperParameters
from keras_tuner.tuners import BayesianOptimization, RandomSearch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import RND_SEED, preprocess     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def build_model_mlp(hp: HyperParameters):
    """Exactly as printed on p.136."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(3,)))

    for num_layer in range(hp.Int("num_layers", 1, 2)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f"units_{num_layer}", min_value=3, max_value=10, step=1),
            activation=hp.Choice(f"activation_{num_layer}", ["relu", "tanh"]),
            name=f"layers_hidden_{num_layer}",
        ))

    model.add(tf.keras.layers.Dense(
        1, activation=hp.Choice("activation_output", ["relu", "tanh"])))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_model_fcc(hp: HyperParameters):
    """Exactly as printed on p.137."""
    inputs = tf.keras.layers.Input(shape=(3,))
    layers = []

    for num_layer in range(hp.Int("num_layers", 1, 2)):
        units = hp.Int(f"units_{num_layer}", min_value=3, max_value=10, step=1)
        activation = hp.Choice(f"activation_{num_layer}", ["relu", "tanh"])
        concat = tf.keras.layers.Concatenate()(layers + [inputs]) if layers else inputs
        dense = tf.keras.layers.Dense(units, activation=activation)(concat)
        layers.append(dense)

    final_concat = tf.keras.layers.Concatenate()(layers + [inputs])
    output = tf.keras.layers.Dense(
        1, activation=hp.Choice("activation_output", ["relu", "tanh"]))(final_concat)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    print("=" * 78)
    print("Ch.6 hyperparameter search — MLP and FCC (printed pp.136-137)")
    print(f"budget: {args.trials} trials x {args.epochs} epochs  (REDUCED, see C6-4)")
    print("*** SYNTHETIC DATA ***")
    print("=" * 78)

    (tr_x, tr_y), (va_x, va_y), _, _, _ = preprocess()
    tr_x, tr_y = tr_x.to_numpy(np.float32), tr_y.to_numpy(np.float32)
    va_x, va_y = va_x.to_numpy(np.float32), va_y.to_numpy(np.float32)

    rows = []
    for arch, builder in [("MLP", build_model_mlp), ("FCC", build_model_fcc)]:
        for algo_offset, (algo_name, algo) in enumerate(
                [("RandomSearch", RandomSearch),
                 ("BayesianOptimization", BayesianOptimization)]):
            # Each (architecture, algorithm) gets its own fixed seed. Reusing ONE seed
            # across both algorithms makes them walk the same trial sequence and return
            # identical results at this small budget, which would fake an agreement
            # that isn't there. Distinct fixed seeds keep each run reproducible while
            # letting the two searches actually differ.
            seed = RND_SEED + algo_offset
            directory = ROOT / "outputs" / "tuning" / "ch06" / f"{arch}_{algo_name}"
            if directory.exists():
                shutil.rmtree(directory)

            tf.keras.utils.set_random_seed(seed)

            tuner = algo(
                builder,
                objective="val_mae",
                max_trials=args.trials,
                directory=str(directory),
                project_name="search",
                seed=seed,
            )
            print(f"\n--- {arch} / {algo_name} ---")
            tuner.search(tr_x, tr_y, validation_data=(va_x, va_y),
                         epochs=args.epochs, verbose=0)

            best_hp = tuner.get_best_hyperparameters(1)[0]
            best_model = tuner.get_best_models(1)[0]
            val_mae = min(t.score for t in tuner.oracle.get_best_trials(args.trials)
                          if t.score is not None)

            units = [best_hp.get(f"units_{i}")
                     for i in range(best_hp.get("num_layers"))]
            acts = [best_hp.get(f"activation_{i}")
                    for i in range(best_hp.get("num_layers"))]
            print(f"  hidden layers   : {best_hp.get('num_layers')}")
            print(f"  units           : {units}")
            print(f"  activations     : {acts}")
            print(f"  output act.     : {best_hp.get('activation_output')}")
            print(f"  parameters      : {best_model.count_params()}")
            print(f"  best val MAE    : {val_mae:.6f} (normalised)")

            rows.append({
                "architecture": arch, "algorithm": algo_name,
                "num_layers": best_hp.get("num_layers"), "units": str(units),
                "activations": str(acts),
                "output_activation": best_hp.get("activation_output"),
                "params": best_model.count_params(), "val_mae": round(val_mae, 6),
            })

    print("\n" + "=" * 78)
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    out = ROOT / "outputs" / "ch06_tuning_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nsaved: {out.name}")
    print("\nThe chapter found random search gave the best validation MAE, with both")
    print("algorithms selecting 2 hidden layers (printed p.127). Any agreement here")
    print("is on synthetic data and is not a reproduction.")


if __name__ == "__main__":
    main()
