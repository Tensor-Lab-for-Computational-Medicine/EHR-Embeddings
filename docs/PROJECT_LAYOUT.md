# Project layout and manuscript reproducibility

This document describes **where manuscript-related code and artifacts live** and **how to reproduce the paper results**. It supersedes the outdated folder sketch in the root `README.md` (there is no top-level `src/` package today).

## What counts as “the manuscript”

Published analyses correspond to **Phase 1–6** under `notebooks/`, plus exported tables and figures under `manuscript_figures/`. Later notebook phases and auxiliary local datasets exist only on disk for exploratory work; **they are not tracked in git** (see `.gitignore`). Do not rely on them to reproduce the manuscript.

## Scope caveat

Many generated artifacts (PNGs, PDFs, pickles, embeddings) are large or sensitive; `.gitignore` excludes broad patterns (`*.pkl`, `*.csv`, etc.). What you need for reproduction is the **code**, pinned **environment**, and documented **run order** below; large binaries may need to be regenerated or supplied per `data/README.md`.

## Top-level layout

| Path | Role |
|------|------|
| `data/` | Data staging (`data/README.md`): manuscript pipeline expects material under `raw/` / `processed/` as described there |
| `docs/` | Analysis plans and this layout/repro doc |
| `manuscript_figures/` | Publication-ready figures and LaTeX tables produced from Phases 1–6 |
| `notebooks/` | Study code by phase (**1–6** for the manuscript); exploratory code may exist locally but is ignored |
| `cache/` | Cached embeddings / intermediates (gitignored patterns apply) |
| `logs/` | Runtime logs |

Miscellaneous drafts at repo root (`abstract.md`, slides, `.xlsx`) are editorial; consolidate new manuscript outputs under `docs/` or `manuscript_figures/` as you prefer.

## Manuscript pipeline (`notebooks/`)

Typical dependency chain for **reproducing findings**:

1. **Phase 1 and 2** — Tabular preprocessing, elastic net / XGBoost baselines. Central artifacts: `notebooks/Phase 1 and 2/phase_1_outputs/` (`X_train`/`X_test`/`y_*` pickles, per-task subfolders, calibrated baseline models, calibration plots). Downstream phases consume `phase_1_outputs` paths (often hard-coded).
2. **Phase 3** — Clinical text for embedding: dataset construction (`create_text_dataset.py`, etc.) and embedding jobs under `Embeddings/` (`generate_embeddings*.py`, Vertex/Gemini configs).
3. **Phase 4** — Supervised runs on embeddings: `config_embedding_analysis_*.py` targets task-specific output dirs; figures under `notebooks/Phase 4/figures/` and embedding caches per config (many paths under `notebooks/Phase 4/`).
4. **Phase 5** — Aggregated embedding-model comparisons: `notebooks/Phase 5/embedding_model_results/<embedding_model_id>/<clinical_task>/` (logs, calibrated champion models, Optuna studies where saved).
5. **Phase 6 - H2 Analysis** — Semantic vs numeric disagreement and archetype/meta_characterization: `h2a/`, `h2b/`, `feature_engineering/` with outputs under each folder’s `h2_results*` / `outputs/`.

Many modules **hard-code** strings such as `notebooks/Phase 1 and 2/phase_1_outputs` and `notebooks/Phase 5/embedding_model_results/...`. Keep the repo root as the working directory when running scripts unless a script states otherwise.

### One-pass reproduction checklist

| Step | Action |
|------|--------|
| 1 | From repo root: `conda env create -f environment.yml` then `conda activate mimic_legacy` (or `pip install -r requirements.txt` in a compatible Python 3.7 env). |
| 2 | Place cohort sources per `data/README.md` (`raw/` HDF5, etc.); run Phase 1–2 preprocessing and baseline training scripts as needed to populate `phase_1_outputs/` (see `elastic_net_analysis.py`, `xgboost_analysis.py`, cohort generators). |
| 3 | Run Phase 3 text + embedding generation for the embedding IDs your configs reference (Vertex/local scripts in `notebooks/Phase 3/`). |
| 4 | Run Phase 4 configs matching the tasks/models in the paper (each `config_embedding_analysis_*.py` writes under Phase 5–style dirs). |
| 5 | Confirm Phase 5 result folders contain the champion arms referenced by Phase 6 configs (`embedding_model_results/...`). |
| 6 | Run Phase 6 `h2a` / `h2b` pipelines with the matching `config_h2_*.py` for mortality vs readmission as applicable. |
| 7 | Regenerate or copy manuscript-ready exports into `manuscript_figures/` using the Phase 4–6 plotting/table scripts tied to your publication revision. |

API keys (e.g. Google/Vertex) belong in `.env` (gitignored). Regenerated pickles and CSVs stay local per `.gitignore` unless you intentionally release artifacts.

## Root-level Python utilities

| Location | Role |
|----------|------|
| `tests/` | Small checks (`test_*.py`); run from **repository root** so relative paths resolve |
| `scripts/` | One-off helpers (`print_labels.py`, etc.) |

```text
tests/
  test_linear_probe.py
  test_r2.py
  test_sensitivity_analysis.py
  test_vars.py
scripts/
  print_labels.py
```

## Environment reproducibility

| File | Purpose |
|------|---------|
| `environment.yml` | Conda spec (`mimic_legacy`), Python 3.7.16 and pinned scientific stack |
| `requirements.txt` | Pip mirror / complement where conda is not used |
| `setup_environment.bat` / `setup_environment.sh` | `conda env create -f environment.yml` |

The `README.md` has been updated to reflect the study results and current repository structure. The active manuscript pipeline lives under `notebooks/` Phases **1–6** and `manuscript_figures/`. Root `requirements.txt` supports pip installs.
