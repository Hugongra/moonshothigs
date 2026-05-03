# Higgs peak pipeline from CSV (paper mapping)

This describes **`run_higgs_pipeline.py`** and **`analysis/higgs_peak_analysis.py`**.

## Physics mapping (CMS arXiv:1207.7235)

| Paper idea | CSV | What we do |
|------------|-----|------------|
| **§5.1 H → γγ** — narrow peak in **diphoton invariant mass** between ~110–150 GeV on falling backgrounds | `data/diphoton.csv` | Histogram **M(γγ)**; define **signal region** near 125 GeV; estimate background from **left/right sidebands** (events/GeV averaged × SR width). |
| **H → ZZ → 4ℓ** — resolved mass peak | `data/hto4leptons.csv` | Same recipe on **four-lepton invariant mass M**. |

This mirrors the **spirit** of a bump hunt (localized excess vs smooth background). It is **not** the CMS **binned likelihood**, **MC shapes**, **systematics**, or **blinding** workflow.

## Current sample size

The bundled extracts are **tiny** (on the order of 10 γγ and a handful of 4ℓ events). **Sidebands may be empty**, so **B_expected ≈ 0** and “significance” is **not meaningful** — the code still prints the recipe so you can swap in larger CSVs later.

For a **fuller** 4ℓ replication path (AOD/ROOT), use **`references/HiggsExample20112012/`**, not these CSVs.

## Outputs

- Plots: `output/higgs_figures/diphoton_mass_higgs_search.{pdf,png}`, `fourlepton_mass_higgs_search.{pdf,png}`
- LaTeX: `output/higgs_paper/paper.tex` (figures resolved via `../higgs_figures/`)
- Overleaf zip layout: `output/higgs_overleaf_bundle/main.tex` + `figures/*.pdf`
- PDF (local): run **`run_higgs_pipeline.py`** without `--no-compile` if **`pdflatex`** is installed

Signal region (defaults) is shaded in yellow; vertical dashed line at **125 GeV**.

## Extending

- Add rows to `data/diphoton.csv` / `data/hto4leptons.csv` (keep columns consistent).
- Tune `sr_lo` / `sr_hi` / sidebands in **`analyze_diphoton`** / **`analyze_four_lepton`** if your selections change.
