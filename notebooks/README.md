# Tier 2 — CSV + Jupyter track (pedagogy)

This folder documents how to run the **CMS open-data education notebooks** shipped under `references/cms-jupyter-materials-english-1.0/`. Together with `../data/PROVENANCE.md`, this is the **Tier 2** workflow: you reproduce **analysis methods** (invariant mass, histograms, comparisons, basic statistics), **not** the full combined CMS publication chain from [arXiv:1207.7235](https://arxiv.org/abs/1207.7235).

## Scope (read once)

- These exercises use **reduced CSV extracts** from Run 2011 open data, tuned for teaching.
- They **do not** replace CMSSW/ROOT workflows or recover official CMS significances from the discovery paper.
- For the closest **paper-linked** channel replication in your bundle, see `references/HiggsExample20112012/README.md` (H → ZZ → 4ℓ, ROOT-based).

## Environment

Upstream lists old pins in `references/cms-jupyter-materials-english-1.0/requirements.txt`. Prefer a **fresh virtual environment** and modern equivalents, for example:

```bash
cd /Users/anonymous2/Desktop/higgs
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter pandas numpy matplotlib scipy
```

If a notebook errors on API changes, adjust cell-by-cell (pandas/matplotlib evolved since the original release).

## Where the notebooks live

| Purpose | Path (relative to repo root) |
|--------|-------------------------------|
| Open-data exercises (main Tier 2 chain) | `references/cms-jupyter-materials-english-1.0/Exercises-with-open-data/` |
| Jupyter/Python primer | `references/cms-jupyter-materials-english-1.0/Introduction-to-jupyter/` |
| CSV data used by notebooks | `references/cms-jupyter-materials-english-1.0/Data/` |

**Important:** Notebooks load CSVs with paths such as `../Data/….csv`. That assumes your **working directory** is the `Exercises-with-open-data` folder when kernels resolve paths. Easiest workflow:

```bash
cd /Users/anonymous2/Desktop/higgs/references/cms-jupyter-materials-english-1.0/Exercises-with-open-data
jupyter notebook
```

Start notebooks from that directory so relative paths stay valid.

## Recommended order (increasing complexity)

Aligned with `Exercises-with-open-data/README.md`:

1. **`Calculate-invariant-mass.ipynb`** — Build invariant mass from four-momenta; connects mass peaks to resonances (Z, J/ψ, ϒ, etc.).
2. **`Invariant-mass-histogram.ipynb`** — Histogram mass distributions.
3. **`Invariant-mass-histogram-weights.ipynb`** — Weighted histograms (when the exercise introduces weights).
4. **`Invariant-mass-histogram-select-data.ipynb`** — Selections on kinematics before plotting.
5. **`Overlaid-histograms.ipynb`** — Compare spectra (e.g. categories overlaid).
6. **`Statistics.ipynb`** — Statistical concepts used in bump hunts (toy uncertainty / inference mindset; not full CMS likelihood).
7. **`Pseudorapidity-resolution.ipynb`** — Detector/experiment literacy (η coverage).

Optional primer if needed:

- `Introduction-to-jupyter/Jupyter-getting-started.ipynb`
- `Introduction-to-jupyter/Import-csv-data.ipynb`
- `Introduction-to-jupyter/Python-basics.ipynb`

## Connecting concepts to the Higgs discovery paper

Use your indexed PDF notes while working:

- **Invariant mass and narrow peaks** ↔ γγ and ZZ high-resolution channels in the paper (conceptually).
- **Background-dominated spectra** ↔ why searches look for **localized excesses** on smooth shapes.
- **Combined significance** ↔ understood only qualitatively here; the paper combines many channels and systematic treatments you are **not** reproducing in Tier 2.

Write up conclusions in your own `docs/` notes if you want a graded report: methods learned, plots produced, and explicit limitations versus arXiv:1207.7235.

## Script alternative (no Jupyter UI)

For a single command that runs analysis, saves plots, and writes a LaTeX note (optional PDF), see the project **`README.md`** and **`run_pipeline.py`**.

## Sanity checklist

- [ ] Jupyter launched from `Exercises-with-open-data` (or edit CSV paths consistently).
- [ ] `../data/PROVENANCE.md` reviewed so each CSV’s role is clear.
- [ ] Figures saved/exported with filenames noting which notebook produced them.
