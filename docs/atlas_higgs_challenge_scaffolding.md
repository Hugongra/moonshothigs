# ATLAS Higgs Boson Machine Learning Challenge (2014) — analysis scaffolding

This note ties **`run_atlas_pipeline.py`** to the **ATLAS Higgs Challenge** setup documented on [Kaggle](https://www.kaggle.com/c/higgs-boson-competition-2014) and companion materials (including **`references/atlas-higgs-challenge-2014.pdf`** in this repo).

## Scientific setting

- **Final states**: simulated \(\tau^+\tau^-\) signatures from vector-boson fusion (VBF) and gluon fusion–like production (see challenge brief).
- **Goal**: discriminate **Higgs signal** (`Label == s`) from **background** (`Label == b`) using reconstructed **DER** (derived) and **PRI** (primary) quantities.

## Data columns (CSV)

| Group | Meaning |
|-------|--------|
| `DER_*` | Physics-motivated derived variables (masses, \(\Delta\eta\), \(\Delta R\), \(p_T\) sums, …). |
| `PRI_*` | Tau / lepton / MET / jet \(p_T\), \(\eta\), \(\phi\)… |
| `Weight` | Monte Carlo weight **must be used** when training/scoring (imbalance + physics). |
| `Label` | `s` (signal) or `b` (background). |
| `KaggleSet` | `t` = training (used here); other letters were for leaderboard splits in the competition. |

Sentinel **`-999`** denotes missing / undefined quantities (e.g. no second jet).

## Competition-style workflow (what the pipeline implements)

1. **Load** training rows (`KaggleSet == t`).
2. **Clean**: replace `-999` with **missing** so tree models treat unknown jets consistently (HistGradientBoosting handles NaNs).
3. **Train / validate**: stratified split on `Label`, **sample weights** = `Weight`.
4. **Score**: predicted signal probability \(\hat p(s)\).
5. **METRIC — AMS**: competition metric **Approximate Median Significance** (with regulator \(b_r = 10\)), evaluated after scanning a cut on \(\hat p(s)\) on the validation fold.

Not implemented in this scaffold (you can extend):

- full leaderboard test-set scoring (`KaggleSet == u`),
- deep nets / XGBoost parity with winning entries,
- systematic-driven calibration beyond AMS-on-holdout.

## Outputs

Under **`output/atlas_challenge/`**: **signal vs background** weighted **`DER_mass_MMC`** histogram (Higgs peak vs continuum), ROC curve, AMS vs. threshold, optional feature importance (where available), **`metrics.json`**.

LaTeX and Overleaf (same rule as other pipelines — upload the **whole** bundle, not only `.tex`):

- **`output/atlas_paper/paper.tex`** — local compile next to figures via `\graphicspath`.
- **`output/atlas_overleaf_bundle/`** — **`main.tex`** + **`figures/`** (PDFs copied for upload to Overleaf).

Use **`run_atlas_pipeline.py --no-compile`** if you only want plots and TeX (no `pdflatex`).

## References

- ATLAS / ML challenge documentation PDF in **`references/atlas-higgs-challenge-2014.pdf`**
- Dataset file: **`data/atlas-higgs-challenge-2014-v2.csv`**
