"""
Runnable version of the reciprocating-compressor multi-label classifier.

Source appendices: printed pp. 98-100 (PDF pp. 105-107), Al-Obaidani et al.
Verbatim original: book_original.py  |  Deviations recorded in ISSUES_FOUND.md

The network architecture, loss, metrics, optimiser, split ratios and random_state are
exactly as printed. What changed is only what was needed to make it execute:
  C5-1  the dataset is actually loaded (the book sets `file_path` but never reads it,
        then immediately uses an undefined `df`)
  C5-2  `.copy()` before assigning label columns (book mutates a slice)
  C5-3  `training_ds` / `validation_ds` are actually built (book calls model.fit on
        names that appear nowhere in the appendices)
  C5-4  `import keras_tuner as kt` added (book uses `kt.` with no import)

DATA IS SYNTHETIC — see make_dataset.py. Metrics below describe the generator,
not the authors' compressor.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieten TF banner

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.metrics import AUC, Precision, Recall

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "compressor_synthetic.csv"
FIGS = ROOT / "outputs" / "figures"
RANDOM_STATE = 1697639          # the chapter's value
EPOCHS = 100                    # the chapter's value
BATCH = 32
LABEL_COLS = ["bearings", "wpump", "radiator", "exvalve"]


def load_and_preprocess():
    """Appendix 1, printed p.98 — corrected so it runs."""
    # C5-1: the book assigns file_path but never reads it, then uses `df`.
    df = pd.read_csv(DATA)

    numeric_columns = df.select_dtypes(include=["number"]).columns
    df_numeric = df[numeric_columns]

    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns
    # C5-2: .copy() — the book assigns into a slice (SettingWithCopyWarning / no-op risk)
    df_non_numeric = df[non_numeric_columns].copy()

    df_non_numeric["bearings"] = df["bearings"].map({"Ok": 0, "Noisy": 1})
    df_non_numeric["wpump"] = df["wpump"].map({"Ok": 0, "Noisy": 1})
    df_non_numeric["radiator"] = df["radiator"].map({"Clean": 0, "Dirty": 1})
    df_non_numeric["exvalve"] = df["exvalve"].map({"Clean": 0, "Dirty": 1})

    input_data = df_numeric
    output_data = df_non_numeric[LABEL_COLS]     # keep label order deterministic

    input_train, input_temp, output_train, output_temp = train_test_split(
        input_data, output_data, test_size=0.3, random_state=RANDOM_STATE)
    input_val, input_test, output_val, output_test = train_test_split(
        input_temp, output_temp, test_size=0.5, random_state=RANDOM_STATE)

    return (input_train, output_train), (input_val, output_val), (input_test, output_test)


def as_dataset(inputs, outputs, shuffle=False):
    """C5-3: build the `training_ds` / `validation_ds` the book's model.fit expects."""
    ds = tf.data.Dataset.from_tensor_slices(
        (inputs.to_numpy(np.float32), outputs.to_numpy(np.float32)))
    if shuffle:
        ds = ds.shuffle(len(inputs), seed=RANDOM_STATE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_model():
    """Appendix 2, printed p.99 — architecture exactly as printed."""
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(41, activation="relu"),
        keras.layers.Dense(21, activation="relu"),
        keras.layers.Dense(4, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(name="precision"),
                 Recall(name="recall"), AUC(name="auc")],
    )
    return model


def plot_history(history):
    FIGS.mkdir(parents=True, exist_ok=True)
    pairs = [("accuracy", "Accuracy"), ("loss", "Loss"),
             ("precision", "Precision"), ("recall", "Recall")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.ravel(), pairs):
        ax.plot(history.history[key], label="training")
        ax.plot(history.history[f"val_{key}"], label="validation")
        ax.set_title(f"{title} per epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel(title.lower())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Compressor MLP training history  [SYNTHETIC DATA]", fontweight="bold")
    fig.tight_layout()
    path = FIGS / "ch05_training_history.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion(y_true, y_pred):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, i in zip(axes, range(4)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(LABEL_COLS[i])
    fig.suptitle("Per-component confusion matrices  [SYNTHETIC DATA]", fontweight="bold")
    fig.tight_layout()
    path = FIGS / "ch05_confusion_matrices.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    print("=" * 78)
    print("Ch.5 Reciprocating compressor — multi-label fault classification")
    print("Appendices printed pp.98-100 (PDF pp.105-107)")
    print("*** SYNTHETIC DATA — not the authors' dataset, not their results ***")
    print("=" * 78)

    (xtr, ytr), (xva, yva), (xte, yte) = load_and_preprocess()
    print(f"\nsplit: train={len(xtr)}  val={len(xva)}  test={len(xte)}  features={xtr.shape[1]}")

    training_ds = as_dataset(xtr, ytr, shuffle=True)
    validation_ds = as_dataset(xva, yva)

    model = build_model()
    print()
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs ...")
    # shuffle=False: the tf.data pipeline already shuffles, and Keras warns otherwise
    history = model.fit(training_ds, epochs=EPOCHS,
                        validation_data=validation_ds, verbose=0, shuffle=False)
    print("done.")

    last = {k: v[-1] for k, v in history.history.items()}
    print("\nFinal epoch:")
    for k in ["loss", "accuracy", "precision", "recall", "auc"]:
        print(f"  {k:<10} train={last[k]:.4f}   val={last['val_' + k]:.4f}")

    # ---- evaluate on the held-out test split ----
    probs = model.predict(xte.to_numpy(np.float32), verbose=0)
    preds = (probs >= 0.5).astype(int)
    y_true = yte.to_numpy(int)

    acc_per_label = (preds == y_true).mean(axis=0)
    acc_per_row = (preds == y_true).all(axis=1).mean()

    print("\nAccuracy per label:")
    for name, value in zip(LABEL_COLS, acc_per_label):
        print(f"  {name:<10} {value:.4f}")
    print(f"  {'average':<10} {acc_per_label.mean():.4f}")
    print(f"\nAccuracy per row (all 4 labels correct): {acc_per_row:.4f}")
    print(f"Macro F1: {f1_score(y_true, preds, average='macro', zero_division=0):.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, preds, target_names=LABEL_COLS,
                                zero_division=0))

    p1 = plot_history(history)
    p2 = plot_confusion(y_true, preds)
    print(f"figures: {p1.name}, {p2.name}")
    print("\nREMINDER: synthetic data. The book reports 99% macro F1 / 100% recall")
    print("on its real compressor data (printed p.97); these numbers are unrelated.")


if __name__ == "__main__":
    main()
