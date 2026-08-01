"""
Runnable version of the ESP head-estimation models (MLP and FCC).

Source appendices: printed pp. 134-137 (PDF pp. 141-144), Rahman et al.
Verbatim original: book_original.py  |  Deviations recorded in ISSUES_FOUND.md

Both architectures are exactly as printed:
  MLP  3 -> Dense(10, tanh) -> Dense(10, relu) -> Dense(1, tanh)
  FCC  fully cross-connected: every hidden layer sees the input and all previous
       hidden layers, output layer linear

Changes made only to execute correctly:
  C6-1  ColumnTransformer output columns are labelled from get_feature_names_out()
        rather than assumed  (see column_transformer_bug.py for the proof)
  C6-2  the book's split leaves `testing_output_df_norm` unscaled; added for symmetry
  C6-3  predictions are inverse-transformed before reporting error in engineering
        units, otherwise MAE is in normalised space and not comparable to the
        book's "MAE < 0.6 ft"

DATA IS SYNTHETIC — see make_dataset.py.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "esp_head_synthetic.csv"
FIGS = ROOT / "outputs" / "figures"

RND_SEED = 1746448054           # the chapter's value
EPOCHS = 200
BATCH = 32
INPUT_COLUMNS = ["Speed", "Viscosity", "Flowrate"]
OUTPUT_COLUMNS = ["Head"]


def preprocess():
    """Appendix 2, printed p.134 — with the column-labelling bug fixed."""
    df = pd.read_csv(DATA)

    input_transformer = ColumnTransformer(transformers=[
        ("std", StandardScaler(), ["Speed", "Flowrate"]),
        ("robust", RobustScaler(), ["Viscosity"]),
    ])
    output_transformer = MinMaxScaler()

    training_df, buffer_df = train_test_split(df, test_size=0.3, random_state=RND_SEED)
    validation_df, testing_df = train_test_split(buffer_df, test_size=0.5,
                                                 random_state=RND_SEED)

    def split_xy(frame):
        return frame[INPUT_COLUMNS], frame[OUTPUT_COLUMNS]

    tr_x, tr_y = split_xy(training_df)
    va_x, va_y = split_xy(validation_df)
    te_x, te_y = split_xy(testing_df)

    # C6-1: label from the transformer, then restore the intended column order
    def to_frame(array):
        names = [c.split("__")[-1] for c in input_transformer.get_feature_names_out()]
        return pd.DataFrame(array, columns=names)[INPUT_COLUMNS]

    tr_xn = to_frame(input_transformer.fit_transform(tr_x))
    va_xn = to_frame(input_transformer.transform(va_x))
    te_xn = to_frame(input_transformer.transform(te_x))

    tr_yn = pd.DataFrame(output_transformer.fit_transform(tr_y), columns=OUTPUT_COLUMNS)
    va_yn = pd.DataFrame(output_transformer.transform(va_y), columns=OUTPUT_COLUMNS)
    te_yn = pd.DataFrame(output_transformer.transform(te_y), columns=OUTPUT_COLUMNS)  # C6-2

    return (tr_xn, tr_yn), (va_xn, va_yn), (te_xn, te_yn), te_y, output_transformer


def build_mlp() -> tf.keras.Model:
    """Appendix 3, printed p.135 — preliminary MLP, exactly as printed."""
    model = tf.keras.Sequential([
        tf.keras.Input((3,), name="layers_input"),
        tf.keras.layers.Dense(10, activation="tanh", name="layers_hidden_1"),
        tf.keras.layers.Dense(10, activation="relu", name="layers_hidden_2"),
        tf.keras.layers.Dense(1, activation="tanh", name="layers_output"),
    ], name="MLP")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_fcc() -> tf.keras.Model:
    """Appendix 3, printed p.135 — preliminary FCC, exactly as printed."""
    input_layer = tf.keras.layers.Input(shape=(3,), name="layers_input")
    hidden_1 = tf.keras.layers.Dense(10, activation="relu",
                                     name="layers_hidden_1")(input_layer)
    hidden_2_concat = tf.keras.layers.Concatenate()([input_layer, hidden_1])
    hidden_2 = tf.keras.layers.Dense(10, activation="relu",
                                     name="layers_hidden_2")(hidden_2_concat)
    output_concat = tf.keras.layers.Concatenate()([input_layer, hidden_1, hidden_2])
    output_layer = tf.keras.layers.Dense(1, name="ouput_layer")(output_concat)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name="FCC")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def evaluate(model, te_xn, te_y_true, output_transformer):
    """C6-3: report error in feet, not in normalised units."""
    pred_norm = model.predict(te_xn.to_numpy(np.float32), verbose=0)
    pred_ft = output_transformer.inverse_transform(pred_norm).ravel()
    true_ft = te_y_true["Head"].to_numpy()

    mae = np.mean(np.abs(pred_ft - true_ft))
    rmse = np.sqrt(np.mean((pred_ft - true_ft) ** 2))
    mape = np.mean(np.abs((pred_ft - true_ft) / true_ft)) * 100
    ss_res = np.sum((true_ft - pred_ft) ** 2)
    ss_tot = np.sum((true_ft - true_ft.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    within10 = np.mean(np.abs(pred_ft - true_ft) / true_ft <= 0.10) * 100
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2, within10=within10), pred_ft, true_ft


def plot_predictions(results):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, pred, true) in zip(axes, results):
        lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
        ax.scatter(true, pred, s=12, alpha=0.55, edgecolor="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
        ax.plot([lo, hi], [lo * 1.1, hi * 1.1], "r:", lw=1, label="+/-10%")
        ax.plot([lo, hi], [lo * 0.9, hi * 0.9], "r:", lw=1)
        ax.set_xlabel("experimental head (ft)")
        ax.set_ylabel("predicted head (ft)")
        ax.set_title(f"{name} model")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("ESP head prediction vs actual  [SYNTHETIC DATA]", fontweight="bold")
    fig.tight_layout()
    path = FIGS / "ch06_predictions.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_history(histories):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, hist) in zip(axes, histories):
        ax.plot(hist.history["mae"], label="training")
        ax.plot(hist.history["val_mae"], label="validation")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MAE (normalised)")
        ax.set_title(f"{name} training")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("ESP model training history  [SYNTHETIC DATA]", fontweight="bold")
    fig.tight_layout()
    path = FIGS / "ch06_training_history.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    print("=" * 78)
    print("Ch.6 ESP head estimation — MLP vs FCC")
    print("Appendices printed pp.134-137 (PDF pp.141-144)")
    print("*** SYNTHETIC DATA — not the authors' dataset, not their results ***")
    print("=" * 78)

    (tr_xn, tr_yn), (va_xn, va_yn), (te_xn, _), te_y_true, out_tf = preprocess()
    print(f"\nsplit: train={len(tr_xn)}  val={len(va_xn)}  test={len(te_xn)}")

    results, histories, metric_rows = [], [], []
    for name, builder in [("MLP", build_mlp), ("FCC", build_fcc)]:
        print(f"\n--- {name} ---")
        # Reseed before each build so both models start from a reproducible state and
        # the comparison between them is not luck of initialisation.
        tf.keras.utils.set_random_seed(RND_SEED)
        model = builder()
        model.summary()
        hist = model.fit(
            tr_xn.to_numpy(np.float32), tr_yn.to_numpy(np.float32),
            validation_data=(va_xn.to_numpy(np.float32), va_yn.to_numpy(np.float32)),
            epochs=EPOCHS, batch_size=BATCH, verbose=0)

        metrics, pred, true = evaluate(model, te_xn, te_y_true, out_tf)
        params = model.count_params()
        print(f"  trainable parameters : {params}")
        print(f"  MAE  : {metrics['mae']:.4f} ft")
        print(f"  RMSE : {metrics['rmse']:.4f} ft")
        print(f"  MAPE : {metrics['mape']:.2f} %")
        print(f"  R2   : {metrics['r2']:.4f}")
        print(f"  within +/-10% band : {metrics['within10']:.1f} %")

        results.append((name, pred, true))
        histories.append((name, hist))
        metric_rows.append({"model": name, "params": params, **metrics})

    print("\n" + "=" * 78)
    print(pd.DataFrame(metric_rows).to_string(index=False))

    p1 = plot_predictions(results)
    p2 = plot_history(histories)
    print(f"\nfigures: {p1.name}, {p2.name}")
    print("\nREMINDER: synthetic data. The book reports MAE < 0.6 ft and MAPE < 8%")
    print("on its real ESP measurements (printed p.133); these numbers are unrelated.")


if __name__ == "__main__":
    main()
