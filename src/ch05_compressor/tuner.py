"""
Bayesian hyperparameter tuner for the compressor classifier.

Source: Appendix 3, printed p.100 (PDF p.107), Al-Obaidani et al.

The search space, objective, model builder and tuner class are exactly as printed.
Two documented deviations:

  C5-4  `import keras_tuner as kt` added — the book calls `kt.BayesianOptimization`
        without ever importing keras_tuner.
  C5-5  BUDGET REDUCED. The book uses max_trials=150, executions_per_trial=2,
        epochs=100, i.e. 30,000 epochs of training. That is a multi-hour run.
        Defaults here are 12 trials x 1 execution x 25 epochs. Override with:
            python tuner.py --trials 150 --executions 2 --epochs 100
        to reproduce the published budget exactly.

DATA IS SYNTHETIC — the "best" hyperparameters found describe the generator.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras_tuner as kt          # C5-4
import numpy as np
from tensorflow import keras
from tensorflow.keras.metrics import AUC, Precision, Recall

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import as_dataset, load_and_preprocess     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def build_model(hp):
    """Exactly the builder printed on p.100."""
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(units=hp.Int("units_layer1", min_value=16, max_value=32, step=2),
                           activation="relu"),
        keras.layers.Dense(units=hp.Int("units_layer2", min_value=8, max_value=16, step=2),
                           activation="relu"),
        keras.layers.Dense(4, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[Precision(name="precision"), Recall(name="recall"),
                           AUC(name="auc")])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--executions", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    published = args.trials == 150 and args.executions == 2 and args.epochs == 100
    print("=" * 78)
    print("Ch.5 Bayesian hyperparameter search (printed p.100 / PDF p.107)")
    print(f"budget: max_trials={args.trials}  executions_per_trial={args.executions}  "
          f"epochs={args.epochs}")
    print("        " + ("PUBLISHED BUDGET" if published
                        else "REDUCED from the book's 150 x 2 x 100 — see C5-5"))
    print("*** SYNTHETIC DATA ***")
    print("=" * 78)

    (xtr, ytr), (xva, yva), _ = load_and_preprocess()
    training_ds = as_dataset(xtr, ytr, shuffle=True)
    validation_ds = as_dataset(xva, yva)

    directory = ROOT / "outputs" / "tuning" / "ch05"
    if directory.exists():
        shutil.rmtree(directory)          # fresh search each run

    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective("val_recall", direction="max"),
        max_trials=args.trials,
        executions_per_trial=args.executions,
        directory=str(directory),
        project_name="bayesian_tuning",
    )

    print("\nsearch space:")
    tuner.search_space_summary()

    print("\nsearching ...")
    tuner.search(training_ds, epochs=args.epochs,
                 validation_data=validation_ds, verbose=0)
    print("done.\n")

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    print("Best hyperparameters:")
    for name, value in best_hp.values.items():
        print(f"  {name:<14} {value}")

    baseline = 20 * 41 + 41 + 41 * 21 + 21 + 21 * 4 + 4     # the p.99 architecture
    tuned = best_model.count_params()
    print(f"\nparameters: baseline(41,21)={baseline}   tuned="
          f"({best_hp.get('units_layer1')},{best_hp.get('units_layer2')})={tuned}")
    print(f"reduction: {100 * (1 - tuned / baseline):.1f}%")
    print("\nThe chapter reports Bayesian tuning 'reduced the number of parameters by")
    print("almost half' (printed p.97). Whether that holds here is a property of the")
    print("synthetic data, not a reproduction of their result.")


if __name__ == "__main__":
    main()
