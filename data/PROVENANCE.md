# Data provenance (Tier 2 CSVs)

All files below live under the CMS education bundle included in this repo:

**Bundle root:** `references/cms-jupyter-materials-english-1.0/`  
**Upstream project:** [CMS Jupyter materials (English)](https://github.com/cms-opendata-education/cms-jupyter-materials-english) — classroom materials using CMS open data (CC BY 4.0).

CSV paths below are **relative to that bundle root** (`Data/…`).

## Files

| File | Role (typical use in bundle) |
|------|-------------------------------|
| `Data/Dimuon_DoubleMu.csv` | Dimuon sample; columns include four-momenta and invariant mass `M` |
| `Data/DoubleMuRun2011A.csv` | Double-muon primary-dataset extract (see notebook text for selection context) |
| `Data/Zmumu_Run2011A.csv` | Z → μμ candidate events; used in invariant-mass exercises |
| `Data/Zmumu_Run2011A_masses.csv` | Precomputed masses variant where referenced |
| `Data/Zee_Run2011A.csv` | Z → ee candidates |
| `Data/dielectron.csv` | Dielectron sample referenced in exercises |
| `Data/Jpsimumu_Run2011A.csv` | J/ψ → μμ region |
| `Data/Ymumu_Run2011A.csv` | ϒ → μμ region |
| `Data/Wenu_Run2011A.csv` | W → eν sample |
| `Data/Wmunu_Run2011A.csv` | W → μν sample |
| `Data/peakdata1.csv` … `Data/peakdata6.csv` | Small pedagogical peak/histogram examples |
| `Data/test` | Bundle test artifact (not primary analysis data) |

## Higgs-oriented extracts (`data/` at repo root)

| File | Role |
|------|------|
| `data/diphoton.csv` | Diphoton invariant mass **M** (GeV) + photon kinematics — **H→γγ-style** search (see `run_higgs_pipeline.py`). |
| `data/hto4leptons.csv` | Four-lepton invariant mass **M** — **H→ZZ→4ℓ-style** illustration. |

Provenance of these files should be recorded here when you add more events (portal record, generation script, etc.).

## Notes

- Selection criteria and dataset citations appear inside the corresponding `.ipynb` markdown cells (e.g. DoubleMu AOD record DOI in `Calculate-invariant-mass.ipynb`).
- Row counts and kinematics reflect **teaching skims**, not the full statistics or filtering of the discovery analysis.
- If you add new CSVs under `references/`, append a row here and re-run `rag-ai-scientist setup-rag` if you index CSV content.
