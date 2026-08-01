# Issues found in the published code

Every deviation between `book_original.*` (verbatim) and the runnable modules is listed
here. Nothing was changed silently.

Severity key — **blocking**: the code cannot execute; **silent**: it executes and
produces wrong or misleading results without any error; **hygiene**: works, but relies
on behaviour that is deprecated or fragile.

---

## Chapter 5 — Reciprocating compressors (Al-Obaidani et al., printed pp. 98–100)

### C5-1 `df` is used before it exists — **blocking**
Printed p.98. The listing assigns the path but never reads the file:

```python
file_path = '/content/drive/MyDrive/Thesis/data (1).csv'
numeric_columns = df.select_dtypes(include=['number']).columns   # NameError
```

`pd.read_csv` is never called. The very next line dereferences `df`.

**Fix** — `df = pd.read_csv(file_path)`. In `pipeline.py` the path points at the
synthetic CSV, since the authors' file is a private Google Drive path.

### C5-2 Label columns are assigned into a slice — **hygiene**
Printed p.98. `df_non_numeric = df[non_numeric_columns]` returns a new object whose
copy/view status pandas does not guarantee, and the code then writes into it:

```python
df_non_numeric['bearings'] = df['bearings'].map({'Ok': 0, 'Noisy': 1})
```

This is the classic `SettingWithCopyWarning` pattern. Under pandas 3.0's
copy-on-write semantics the write is not guaranteed to be visible in the original.

**Fix** — `df[non_numeric_columns].copy()`.

### C5-3 `training_ds` / `validation_ds` never exist — **blocking**
Printed p.99. The preprocessing appendix produces `input_train`, `output_train`,
`input_val`, `output_val`. The model appendix then calls:

```python
history = model.fit(training_ds, epochs=100, validation_data=validation_ds)
```

Neither name is defined anywhere in the three appendices. The step that turns the
split arrays into batched datasets is missing from the published listings.

**Fix** — `as_dataset()` in `pipeline.py` builds them with
`tf.data.Dataset.from_tensor_slices(...).batch(32)`.

### C5-4 `kt` is used without importing keras_tuner — **blocking**
Printed p.100. `tuner = kt.BayesianOptimization(...)` with no
`import keras_tuner as kt` anywhere in the chapter.

### C5-5 Tuning budget is impractical to reproduce — **hygiene**
Printed p.100: `max_trials=150, executions_per_trial=2` with `epochs=100` is 30,000
epochs of training. `tuner.py` defaults to 12 × 1 × 25 and takes `--trials/--executions/
--epochs` to run the published budget exactly.

### C5-6 `objective='val_recall'` passed as a bare string — **hygiene**
keras-tuner infers optimisation direction from the metric name. For a metric registered
as `Recall(name='recall')` the inference works, but it is silent if it ever fails.
`tuner.py` uses the explicit `kt.Objective("val_recall", direction="max")`.

---

## Chapter 6 — ESP head estimation (Rahman et al., printed pp. 134–137)

### C6-1 ColumnTransformer output columns are mislabelled — **silent**, most serious
Printed p.134.

```python
input_transformer = ColumnTransformer(transformers=[
    ('std',    StandardScaler(), ["Speed", "Flowrate"]),
    ('robust', RobustScaler(),   ["Viscosity"])
])
input_columns = ["Speed", "Viscosity", "Flowrate"]

training_input_df_norm = pd.DataFrame(
    input_transformer.fit_transform(training_input_df), columns=input_columns)
```

`ColumnTransformer` emits columns in **transformer order**, so the array columns are
`[Speed, Flowrate, Viscosity]`. They are then relabelled `[Speed, Viscosity, Flowrate]`.
**Viscosity and Flowrate are transposed.**

Nothing raises. Shapes match, training converges, and because the same transformer is
reused for validation and test the mapping stays internally consistent — so the model
still learns. The damage is interpretive: any statement about "the effect of viscosity"
read off these columns is really about flowrate.

Demonstrated in `src/ch06_esp_head/column_transformer_bug.py`, which correlates each
labelled column against its true source:

```
label       book column tracks     correlation
Speed       Speed                       1.0000   OK
Viscosity   Flowrate                    1.0000   <-- MISLABELLED
Flowrate    Viscosity                   1.0000   <-- MISLABELLED
```

**Fix** — label from `input_transformer.get_feature_names_out()` and reindex, rather
than assuming the order.

**A note on scope.** The chapter reports an unexplained result (printed p.133): "the
unexpected sensitivity of the MLP model to the output parameter normalisation remain
unexplored and unexplained". Column-handling bugs of exactly this kind are the first
thing to check. I am **not** claiming C6-1 causes it — that cannot be tested without
the authors' dataset. It is flagged as a candidate, nothing more.

### C6-2 Test targets are never scaled — **hygiene**
Printed p.134 ends with `testing_input_df_norm`; there is no
`testing_output_df_norm`. Harmless if predictions are inverse-transformed instead, but
the asymmetry is a trap. Added for symmetry.

### C6-3 Error reported in normalised units — **silent**
The models train on MinMax-scaled head, so `mae` from Keras is in [0,1] space, not feet.
Comparing it to the chapter's "MAE under 0.6 ft" is a category error.
`pipeline.py` inverse-transforms predictions before computing MAE/RMSE/MAPE/R².

### C6-4 `tanh` output against MinMax-scaled targets — **hygiene**
Printed p.135, the preliminary MLP ends `Dense(1, activation="tanh")` while targets are
scaled to [0, 1]. `tanh` spans [-1, 1], so half its range is unreachable and gradients
saturate near the top of the target range. The FCC model on the same page uses a linear
output. Kept as printed — it is the authors' design choice — but worth noting given
C6-1's remark about unexplained normalisation sensitivity.

### C6-5 f-string with no placeholder — **hygiene**
Printed pp.136–137: `hp.Choice(f"activation_output", [...])` — the `f` prefix does
nothing. Cosmetic.

---

## Chapter 10 — KSB / Gülich parameter identification (Hadabi et al., printed pp. 277–280)

The MATLAB is self-consistent and the model core ports to Python cleanly.
`ksb_gulich_python.py` reproduces all 64 published correction factors (printed p.281)
to 4 decimal places, max absolute deviation 4.99e-05 — which is just rounding at the
precision the book prints. One issue was found in the *optimised parameters*, not in
the code.

### C10-1 The published Nelder–Mead parameters put the KSB model outside the reals — **silent**
Printed p.283 (PDF p.290), "Parameter values after optimization with full dataset".

The KSB flow correction is

```
CQ_KSB = exp( b * B * log(a / (52.933 * omega_s)) - c * (log10(B))^d )
```

`(log10(B))^d` is only real-valued when `log10(B) >= 0`, i.e. `B >= 1`, unless `d`
happens to be an integer. The book's three parameter sets give:

| parameter set | a | d | B range over the 16 operating points | rows with B < 1 |
|---|---|---|---|---|
| original (θ₀) | 10.41 | 5.323 | 3.178 – 18.706 | 0 / 16 |
| **Nelder–Mead** | **0.125** | **3.466** | **0.348 – 2.050** | **11 / 16** |
| Levenberg–Marquardt | 1.270 | 9.999 | 1.110 – 6.534 | 0 / 16 |

With the Nelder–Mead parameters, `B` drops below 1 for 11 of the book's own 16
operating points, so the model raises a negative number to the power 3.466.

- **numpy** returns `NaN` (with `RuntimeWarning: invalid value encountered in power`).
- **MATLAB** does not error — it silently promotes to a **complex** number. So
  `fminsearch` would have been minimising an objective that had gone complex over much
  of the domain, with `rmse = sqrt(mean(err.^2))` complex too.

That silent complex promotion is the likely reason this was never noticed: MATLAB gives
no warning, and the optimiser keeps running.

I am reporting the arithmetic, which is checkable from the published tables alone. What
I cannot check without the authors' three spreadsheets is what `fminsearch` actually did
with a complex objective, so I make no claim about whether the published Nelder–Mead
optimum is wrong — only that substituting those parameters into the published formula
does not stay in the reals.

Reproduce with `python src/ch10_ksb_gulich/ksb_gulich_python.py` — the run reports
`Nelder-Mead  NOT EVALUABLE — 22 NaN values in [CQ_KSB, CH_KSB]`.

> A note on my own code: the first version of that summary printed
> `mean CH_KSB=0.7522` for Nelder–Mead, because pandas skips NaN when averaging. That
> silently described 5 rows while looking like it described 16. Fixed to report the NaN
> count instead — the same class of quiet-wrong-number bug as C6-1.

The only obstacle is that the three data files (`TogetCorrectionFactors.xlsx`,
`TrainSet.xlsx`, `Water_Combined_Head_vs_Flow_Data__All_RPMs_.csv`) are not
distributed, so the *optimisation* drivers cannot be re-run — only the model core,
which is what the published tables let us verify.

---

## Cross-cutting: neither Python chapter ships its dataset

Chapter 5 reads `/content/drive/MyDrive/Thesis/data (1).csv` (a private Drive path) and
Chapter 6 reads a bare `dataset.csv`. Neither is in the book or its supplementary
material, so **no published metric in either chapter can be reproduced**. The synthetic
generators here exist so the code paths execute; every output they produce is labelled
as synthetic, and none of it should be read as reproducing the authors' results.
