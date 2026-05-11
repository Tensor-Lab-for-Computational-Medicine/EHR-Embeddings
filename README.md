# Numeric versus Embedding Pipelines For Optimized Clinical Risk Prediction

This repository contains the analysis code and selected manuscript exports for a clinical risk prediction study comparing structured numerical features with semantic embedding features derived from the same EHR data.

The repository is designed for reproducibility of the manuscript workflow, but it intentionally does not version raw patient data, embedding arrays, fitted model artifacts, API credentials, or submission-package files. Those artifacts must be regenerated or supplied locally.

## Reproducibility Scope

Tracked in Git:

- code for Phases 1-6 under `notebooks/`;
- phenotype rules in `notebooks/Phase_6/feature_engineering/feature_rules.csv`;
- data staging notes in `data/README.md`;
- lab reference ranges in `data/Lab_reference_ranges.csv`;
- selected manuscript-ready outputs in `manuscript_outputs/`;
- environment files and documentation.

Not tracked by design:

- `data/raw/`, `data/processed/`, and large or derived local data files;
- `.env` and API credentials;
- `Submission/`;
- preprocessed pickles, model files, embedding vectors, Optuna studies, logs, and most generated CSV/JSON/NumPy/Pickle files;
- `notebooks/Phase_5/embedding_model_results/`, even though Phase 5 and Phase 6 consume results from that local tree.

The `.gitignore` is intentionally strict. If you regenerate a file and Git does not see it, that is usually expected.

## Environment

Recommended setup:

```bash
conda env create -f environment.yml
conda activate mimic_legacy
```

Fallback:

```bash
pip install -r requirements.txt
```

The checked environment is centered on Python `3.7.16`, XGBoost `1.6.2`, scikit-learn `1.0.2`, NumPy `1.21.6`, SciPy `1.7.3`, and pandas-compatible legacy code.

Run commands from the repository root unless a command explicitly changes directories.

## Required Local Inputs

Place or confirm the following files before running the full pipeline:

| Local path | Purpose |
|---|---|
| `data/raw/all_hourly_data.h5` | Main MIMIC-Extract hourly EHR HDF5 input. Download/supply locally from the MIMIC-Extract release bucket: `https://console.cloud.google.com/storage/browser/mimic_extract;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false`. The upstream extraction code is at `https://github.com/MLforHealth/MIMIC_Extract`. |
| `data/processed/eda_results_corrected/feature_classification.csv` | Feature category map used during Phase 1-2 feature engineering. This derived helper file remains local unless explicitly released. |
| `data/Lab_reference_ranges.csv` | Tracked sex-specific reference ranges used by F2 text representation high/low/normal flags. |
| `notebooks/Phase_6/feature_engineering/feature_rules.csv` | Tracked phenotype rule dictionary used for Phase 6 archetype and phenotype analyses. |
| `.env` or cloud auth environment | Google/Vertex credentials if regenerating hosted embeddings. |

The raw MIMIC-Extract file, derived processed data, and credentials must be acquired on physionet. 

## Path Conventions

The tracked folder names use underscores:

- `notebooks/Phase_1-2`
- `notebooks/Phase_3`
- `notebooks/Phase_4`
- `notebooks/Phase_5`
- `notebooks/Phase_6`

Some older scripts/configs still contain historical paths with spaces, such as `notebooks/Phase 4` or `notebooks/Phase 5`. Before a clean rerun, confirm that any paths in the configs you are using point to the underscored folders above, or create local compatibility copies/junctions. The downstream manuscript and Phase 6 code currently expect the underscored layout.

## Pipeline Overview

| Phase | Role | Main tracked entry points | Main local outputs |
|---|---|---|---|
| Phase 1-2 | Cohort preprocessing, subject-level splits, tabular features, numerical baselines. | `data_preprocessing_LOS.py`, `xgboost_analysis.py`, `elastic_net_analysis.py` | `notebooks/Phase_1-2/phase_1_outputs/` |
| Phase 3 | Text representations and task label CSVs. | `create_text_dataset.py`, `text_generator.py`, `config.py` | `notebooks/Phase_3/phase_3_serialized_data/` |
| Phase 4 | XGBoost on embedding arrays; manuscript figures; probe/SHAP analyses. | `xgboost_embedding_analysis.py`, `config_embedding_analysis_*.py`, `generate_manuscript_figure3.py`, `map_clinical_to_embeddings_combined.py` | embedding folders under Phase 4; model results under Phase 5 |
| Phase 5 | DeLong statistical comparison only. | `delong_statistical_analysis.py` | `notebooks/Phase_5/statistical_analysis_output/` |
| Phase 6 | H2a/H2b discordance, archetypes, phenotypes, and meta-feature analyses. | `run_exploration.bat`, `h2a/`, `h2b/`, `feature_engineering/` | ignored H2 result folders; selected exports to `manuscript_outputs/` |

## Step 1: Preprocess Tabular Data

Run preprocessing after local data inputs are in place. This invocation writes the fixed filenames expected by the newer downstream scripts:

```powershell
@'
import sys
sys.path.insert(0, r"notebooks\Phase_1-2")
import data_preprocessing_LOS

data_preprocessing_LOS.main({
    "OUTPUT_DIR": r"notebooks\Phase_1-2\phase_1_outputs",
    "USE_PREFIXED_FILENAMES": False,
})
'@ | python -
```

This creates subject-level train, validation, and test splits. The split is generated by the project code, not by an external predefined benchmark:

- split unit: `subject_id`;
- test fraction: `25%`;
- validation fraction: `12.5%` of the train+validation pool;
- stratification target: `mort_hosp`;
- seed: `42`;
- saved ICU stay ID files: `icustay_ids_train.pkl`, `icustay_ids_val.pkl`, `icustay_ids_test.pkl`.

Expected local outputs:

```text
notebooks/Phase_1-2/phase_1_outputs/
  X_train.pkl
  X_val.pkl
  X_test.pkl
  y_train.pkl
  y_val.pkl
  y_test.pkl
  scaler.pkl
  label_encoders.pkl
  imputation_values.pkl
  icustay_ids_train.pkl
  icustay_ids_val.pkl
  icustay_ids_test.pkl
```

Create compatibility copies with the historical prefix used by the Phase 3 text serializer:

```powershell
@'
from pathlib import Path
import shutil

out = Path(r"notebooks\Phase_1-2\phase_1_outputs")
prefix = "preprocessed_mort_hosp_los_3_los_7_readmission_30_intervention_vent_intervention_vaso_trends_True_window_24_gap_6_seed_42"
names = [
    "X_train", "X_val", "X_test",
    "y_train", "y_val", "y_test",
    "scaler", "label_encoders", "imputation_values",
    "icustay_ids_train", "icustay_ids_val", "icustay_ids_test",
]

for name in names:
    fixed = out / f"{name}.pkl"
    prefixed = out / f"{prefix}_{name}.pkl"
    if fixed.exists() and not prefixed.exists():
        shutil.copy2(fixed, prefixed)
    elif prefixed.exists() and not fixed.exists():
        shutil.copy2(prefixed, fixed)
'@ | python -
```

After this step, both fixed names and prefixed names should exist. They should refer to the same deterministic split because both conventions are derived from the same Phase 1-2 preprocessing run.

## Step 2: Train Numerical Baselines

The baseline scripts are parameterized through their `Config` classes. To reproduce all six tasks, call `main(config_dict=...)` for each target.

Example XGBoost sweep:

```powershell
@'
import sys
sys.path.insert(0, r"notebooks\Phase_1-2")
import xgboost_analysis

targets = [
    "mort_hosp",
    "los_3",
    "los_7",
    "readmission_30",
    "intervention_vent",
    "intervention_vaso",
]

for target in targets:
    xgboost_analysis.main({
        "TARGET_VARIABLE": target,
        "INPUT_DIR": r"notebooks\Phase_1-2\phase_1_outputs",
        "CALIBRATION_ENABLED": True,
        "CALIBRATION_METHOD": "isotonic",
    })
'@ | python -
```

Example ElasticNet sweep:

```powershell
@'
import sys
sys.path.insert(0, r"notebooks\Phase_1-2")
import elastic_net_analysis

targets = [
    "mort_hosp",
    "los_3",
    "los_7",
    "readmission_30",
    "intervention_vent",
    "intervention_vaso",
]

for target in targets:
    elastic_net_analysis.main({
        "TARGET_VARIABLE": target,
        "INPUT_DIR": r"notebooks\Phase_1-2\phase_1_outputs",
    })
'@ | python -
```

Expected per-task outputs are written under:

```text
notebooks/Phase_1-2/phase_1_outputs/<target>/
```

## Step 3: Generate Text Representations

Confirm `data/Lab_reference_ranges.csv` exists, then run:

```powershell
python notebooks\Phase_3\create_text_dataset.py
```

Expected local outputs:

```text
notebooks/Phase_3/phase_3_serialized_data/
  <target>_train_labels.csv
  <target>_val_labels.csv
  <target>_test_labels.csv
  F1_P0/train/*.txt
  ...
  F3_P5/test/*.txt
```

The F2 representation uses `data/Lab_reference_ranges.csv` for sex-specific high/low/normal flags.

## Step 4: Generate Embeddings

Embedding-generation scripts and batch/API helpers are local/generated in this project and may be ignored. Regenerate or supply embeddings so that Phase 4 configs can find `.npy` arrays in their expected local directories.

Models evaluated in the manuscript result tree include:

- `embedding-001`;
- `text-embedding-004`;
- `text-embedding-005`;
- `text-embedding-large-exp-03-07`;
- `MedEmbed-small`;
- `NeuML_pubmedbert-base-embeddings`.

Embedding arrays are not L2-normalized before XGBoost in the tracked training code.

## Step 5: Train Semantic Models

Run `notebooks/Phase_4/xgboost_embedding_analysis.py` using the desired config import at the top of that file. The script currently runs one embedding-config/target combination at a time.

Before running, confirm the imported config in `xgboost_embedding_analysis.py`, for example:

```python
from config_embedding_analysis_text_embedding_005 import Config
```

Then:

```powershell
python notebooks\Phase_4\xgboost_embedding_analysis.py
```

Expected local outputs:

```text
notebooks/Phase_5/embedding_model_results/<embedding_model>/<target>/
  model_<arm>.pkl
  model_<arm>_calibrated.pkl
  results_<arm>.pkl
  optuna_study_*.pkl
  embedding_analysis_log.txt
```

That whole result tree is ignored by Git. Phase 5 and Phase 6 require it to exist locally.

Champion semantic arms referenced by later analyses:

| Task | Champion arm | Embedding model |
|---|---|---|
| `mort_hosp` | `F3_P5` | `text-embedding-004` |
| `readmission_30` | `F1_P0` | `text-embedding-005` |
| `los_3` | `F3_P1` | `text-embedding-004` |
| `los_7` | `F3_P2` | `text-embedding-005` |
| `intervention_vent` | `F3_P0` | `text-embedding-004` |
| `intervention_vaso` | `F3_P2` | `text-embedding-004` |

## Step 6: Run DeLong Statistical Comparisons

Phase 5 currently contains only the DeLong comparison script. It consumes local ignored result files from:

```text
notebooks/Phase_5/embedding_model_results/
```

Run:

```powershell
python notebooks\Phase_5\delong_statistical_analysis.py
```

Expected local output:

```text
notebooks/Phase_5/statistical_analysis_output/
```

## Step 7: Run Discordance, Archetype, And Meta-Feature Analyses

From the repository root, run the Phase 6 batch script. Edit `PYTHON_EXE` inside `run_exploration.bat` first if your conda environment path differs from the local path in the file.

```powershell
notebooks\Phase_6\run_exploration.bat h2b\config_h2_morthosp.py
```

By default, this runs mortality and then readmission unless `skip_readmin` is passed. It also builds phenotype artifacts if they do not already exist.

Important parameters embedded in the Phase 6 workflow:

- phenotype rules: `notebooks/Phase_6/feature_engineering/feature_rules.csv`;
- number of phenotype rules: `53`;
- subgroup discovery depths: `2,3,4,5,6,7`;
- beam width/result set size: `200`;
- final archetype minimum test coverage: `75` patients;
- final archetype minimum test lift: `1.5`;
- archetype Jaccard deduplication threshold: `0.75`;
- maximum reported archetypes: `8`.

Expected local outputs include:

```text
notebooks/Phase_6/feature_engineering/artifacts/
notebooks/Phase_6/h2a/h2_results/
notebooks/Phase_6/h2b/h2_results/
```

These generated folders are ignored except for intentionally tracked source files.

## Step 8: Regenerate Manuscript Outputs

Selected committed outputs live under:

```text
manuscript_outputs/Figures/
manuscript_outputs/Tables/
```

Useful tracked exporters include:

```powershell
python notebooks\generate_phase3_methodology_tables.py
python notebooks\Phase_4\generate_manuscript_figure3.py
python notebooks\Phase_4\generate_phenotype_visualizations.py
python notebooks\Phase_6\h2b\generate_phase_v_meta_tables.py
python notebooks\Phase_6\h2b\generate_archetype_reports.py
python notebooks\Phase_6\h2b\generate_manuscript_table.py
```

Some exporters depend on local ignored model/result artifacts. Run them after the phase outputs they consume exist.

## Validation Checklist

Before comparing manuscript numbers, confirm:

- `notebooks/Phase_1-2/phase_1_outputs/` contains `X_*`, `y_*`, scaler, label encoders, and split ID pickles;
- task label CSVs exist under `notebooks/Phase_3/phase_3_serialized_data/`;
- embedding `.npy` files exist for all required arms and splits;
- `notebooks/Phase_5/embedding_model_results/` contains `results_<arm>.pkl` for the models/tasks being compared;
- Phase 6 champion model paths in `h2a/config_h2_*.py` resolve to existing local files;
- `data/Lab_reference_ranges.csv` exists locally for F2 reproducibility;
- commands are run from repo root, except the Phase 6 batch file, which changes into `notebooks/Phase_6` internally.

## Privacy And Credentials

Do not commit raw MIMIC data, PHI, API keys, embedding vectors, fitted patient-level artifacts, or `Submission/`. Users are responsible for complying with the PhysioNet data use agreement, HIPAA, institutional policy, and any applicable restrictions before sending text representations to external embedding APIs.

## Citation

Ge, A., Wu, J., Wu, X., et al. Numeric versus Embedding Pipelines For Optimized Clinical Risk Prediction. Manuscript in preparation.
