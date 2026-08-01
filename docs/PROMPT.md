# The Prompt

This is the executable brief generated before any code was written, from a full scan of
*Smart Diagnostics and Predictive Maintenance* (ed. Aydin Azizi, Springer, Emerging Trends
in Mechatronics). It is reproduced here verbatim so the work can be audited against it.

---

## Objective

Extract **every code listing in the book**, make each one actually run on this Windows
machine, capture genuine output, document the real defects found in the published code,
add a classic-problems module (FizzBuzz and relatives), and publish the whole thing to
GitHub with a README.

## Step 1 — Locate every code listing

Scan all 293 PDF pages with a Python-token heuristic. Do not trust the page numbers given
in conversation without checking: the book's *printed* page numbers are offset from the PDF
page indices by 7 (printed 98 = PDF 105). Report the full inventory and explicitly separate
genuine code from prose that merely mentions MATLAB.

Expected inventory (to be confirmed by the scan):

| Source | Chapter | Printed pp. | PDF pp. | Language |
|---|---|---|---|---|
| A | Data-Driven Condition Monitoring of Reciprocating Compressors (Al-Obaidani et al.) | 98–100 | 105–107 | Python / Keras |
| B | AI Model to Estimate the Head of an ESP (Rahman et al.) | 134–137 | 141–144 | Python / Keras |
| C | Parameter Identification of Empirical Models for Head Estimation (Hadabi et al.) | 277–280 | 284–287 | MATLAB |
| D | Reference data tables for C | 281–283 | 288–290 | data |

## Step 2 — Transcribe faithfully, then make it run

For each listing keep **two** files:

- `book_original.py` / `.m` — a verbatim transcription, unmodified, marked clearly as
  not-runnable where that is the case. This preserves what the book actually printed.
- working modules — the same logic, corrected only where it cannot execute, with every
  change recorded in `ISSUES_FOUND.md`.

**Do not silently "fix" the book.** Every deviation must be justified and listed.

## Step 3 — Solve the data problem honestly

Neither chapter ships its dataset (`data (1).csv`, `dataset.csv` are Google-Drive/local
paths). Therefore:

- Sources A and B run on **clearly-labelled synthetic data** generated to the schema the
  chapters describe (A: 20 numeric features → 4 binary component labels; B: Speed,
  Viscosity, Flowrate → Head). Every output must state that it is synthetic. Accuracy
  figures from synthetic data must **never** be presented as reproducing the authors'
  results.
- Source C is different: the book prints its actual input table and its actual computed
  correction factors (PDF 288). So the MATLAB→Python port must be **numerically validated
  against the book's own published numbers**, and the comparison reported honestly
  whether it matches or not.

## Step 4 — Find the real issues

Audit the published listings for genuine defects — undefined names, pandas copy-vs-view
errors, missing imports, and especially any silent correctness bug. Record each in
`ISSUES_FOUND.md` with: the exact page, the offending line, why it breaks, and the fix.
Where a bug can be *demonstrated*, write a script that demonstrates it rather than merely
asserting it.

## Step 5 — Classic problems ("viral issue like FizzBuzz")

Add `classic_problems/` containing FizzBuzz and a set of similarly famous interview
problems, each with a clean Python implementation and passing unit tests. Keep the style
consistent with the rest of the repo.

## Step 6 — Run everything and capture output

Execute every runnable module. Capture:
- terminal transcripts to `outputs/logs/`
- matplotlib figures to `outputs/figures/` (training curves, confusion matrices,
  prediction-vs-actual, validation scatter)

These figures and logs *are* the program output. Where a literal screenshot of the editor
is wanted, state plainly what was captured and what was not.

## Step 7 — Publish

`git init`, commit, create the GitHub repo under the user's account, push. README must
document: what the book is, what each module does, which page it came from, how to run it,
what is synthetic vs validated, and the issues found. Repo is code-only — no contact
details, no motivation prose.

## Standing constraints

- Report outcomes faithfully. If a validation fails, say so and show the numbers.
- Never present synthetic-data metrics as reproductions of published results.
- Every changed line must trace to a stated reason.
