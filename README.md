# Numeric versus Embedding Pipelines For Optimized Clinical Risk Prediction

**Authors:** Aaron Ge<sup>1,2</sup>, Jialong Wu<sup>3</sup>, Xueyao Wu<sup>4</sup>, Jeya Balaji Balasubramanian<sup>4</sup>, Varun Nautiyal<sup>5</sup>, Rishi Jayakumar<sup>6</sup>, Chy Murali<sup>2</sup>, Angela Lee<sup>2</sup>, Andrew Nguyen<sup>2</sup>, Matthew Allen<sup>7</sup>, Chang Shu<sup>7</sup>, Clayton Brown<sup>1,2</sup>, Shuo Chen<sup>1,2</sup>, Katarina Zeder<sup>1,2</sup>, Jonas De Almeida<sup>4</sup>, Florence Doo<sup>1,2</sup>, Bradley A. Maron<sup>1,2</sup>*

**Affiliations:**
1. University of Maryland-Institute of Health Computing, University of Maryland, School of Medicine, Baltimore, MD, USA
2. University of Maryland, School of Medicine, Baltimore, MD, USA
3. Department of Computer Science, Whiting School of Engineering, Johns Hopkins University, Baltimore, MD, USA
4. Division of Cancer Epidemiology and Genetics, National Cancer Institute, National Institutes of Health, Maryland, USA
5. Department of Computer Science, College of Computer, Mathematical, and Natural Science, University of Maryland, College Park, MD, USA
6. Department of Exercise and Nutrition Sciences, Milken Institute School of Public Health, George Washington University, Foggy Bottom, DC, USA
7. University of California, San Francisco, School of Medicine, San Francisco, CA, USA

*\*Corresponding author: Dr. Bradley A. Maron (BMaron@som.umaryland.edu)*

---

## 📄 Abstract

**Background:** Clinical risk prediction models typically use structured numerical inputs such as lab values and their statistical summaries. Semantic text embeddings offer an alternative: structured data are written out as text and compressed into dense vectors by a foundation model. However, prior head-to-head comparisons of traditional machine learning (ML) and embedding-based clinical prediction models have focused primarily on aggregate metrics such as discrimination, calibration, and fairness. Systematic analyses focusing on model disagreement, differences in the basis for failure, and their respective utility for individual patient-level classification tasks is limited.

**Methods:** We retrospectively analyzed 22,591 intensive care unit (ICU) patients from the MIMIC-III database and compared a **Numerical Model (NM)** trained on 458 structured features with a **Semantic Model (SM)** trained on semantic embeddings using the same XGBoost classifier for both. The models were tested across six prediction tasks: in-hospital mortality, 30-day readmission, mechanical ventilation, vasopressor initiation, and 3-day and 7-day ICU length of stay. We measured model discordance, used linear probes to quantify embedding encoding fidelity, SHAP analysis to characterize predictive features, subgroup discovery to identify clinical error archetypes across 53 phenotypes, and meta-feature analysis to examine whether errors were systematically linked to input data properties.

**Results:** The NM outperformed SM for five acute outcomes including mortality (AUROC 0.901 vs. 0.835, p < 0.001); by contrast, the SM outperformed NM for readmission (0.662 vs. 0.590, p = 0.003). The models failed mostly on non-overlapping patients: 80.7% of mortality false positives and 87.6% of readmission false positives were unique to one model. The embedding preserved measurement frequencies at high fidelity (R² = 0.38 to 0.75) but poorly encoded clinically critical values, notably albumin (R² = 0.231 for mortality, 0.046 for readmission). SM errors concentrated in patients with coagulopathy, hypoalbuminemia, and inflammatory markers, and were consistently associated with high input data density and physiological volatility across all archetypes. NM errors concentrated in patients with creatinine elevation and synthetic hepatic dysfunction, where its acute-severity feature space could not distinguish reversible from irreversible organ dysfunction.

**Conclusions:** Feature representation determines not just aggregate accuracy but which patients a model fails on and why, which are insights otherwise invisible to traditional AUROC comparisons. The embedding's selective degradation of quantitative clinical variables produced predictable, coherent failure modes that were further amplified by input data density. This observation has direct implications for how embedding-based models should be evaluated before clinical deployment.

**Keywords:** Electronic health records; Foundation models; Feature engineering; Clinical risk prediction; Intensive care unit; Model discordance; Subgroup analysis.

---

## 🛠️ Project Structure & Reproduction

This repository contains the code necessary to reproduce the findings of the study. The pipeline is divided into **six sequential phases**.

### 📁 Directory Layout

| Path | Description |
|------|-------------|
| `notebooks/Phase_1-2` | Tabular preprocessing and Numerical Model (NM) baseline training. |
| `notebooks/Phase_3` | Text dataset construction and semantic embedding generation (Vertex/Gemini). |
| `notebooks/Phase_4` | Supervised training and supervised analysis of the Semantic Model (SM). |
| `notebooks/Phase_5` | Model comparison, hyperparameter tuning, and champion model selection. |
| `notebooks/Phase_6` | Analysis of discordance, archetyping, and meta-feature characterization. |
| `manuscript_figures/` | Publication-ready figures and LaTeX tables generated by the pipeline. |
| `data/` | Data staging (MIMIC-III). See `data/README.md` for requirements. |
| `docs/` | Detailed technical layouts and reproducibility documentation. |

---

## 🚀 Getting Started

### 1. Environment Setup

The study was conducted using **Python 3.7.16** and a pinned scientific stack to ensure reproducibility.

**Option A: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate mimic_legacy
```

**Option B: Pip**
```bash
pip install -r requirements.txt
```

### 2. Data Access
This project uses the **MIMIC-III v1.4** database. Users must have credentialed access via [PhysioNet](https://physionet.org/).
- Place raw data (e.g., `all_hourly_data.h5`) in `data/raw/`.
- Refer to `data/README.md` for details on expected files and structure.

### 3. API Configuration
Semantic embeddings were generated using Google Gemini/Vertex AI.
- Create a `.env` file in the root directory.
- Add your API credentials:
  ```env
  GOOGLE_API_KEY=your_actual_api_key_here
  ```

---

## 📊 Analysis Pipeline

To reproduce the study results, follow the phases in order:

1.  **Preprocessing & NM Baselines**: Run scripts in `notebooks/Phase_1-2/` to generate `phase_1_outputs/`.
2.  **Embedding Generation**: Execute `notebooks/Phase_3/` scripts to generate clinical text and fetch embeddings.
3.  **SM Training**: Run Phase_4 configs (`config_embedding_analysis_*.py`) to train embedding-based models.
4.  **Comparison**: Use Phase_5 notebooks to compare NM and SM performance across all tasks.
5.  **Failure Analysis**: Run Phase_6 `h2a` (discordance) and `h2b` (archetyping) modules to characterize why and where models fail.

---

## 📜 Citation

If you use this code or our findings in your research, please cite:

> Ge, A., Wu, J., Wu, X., et al. (2026). Numeric versus Embedding Pipelines For Optimized Clinical Risk Prediction. *Manuscript in Preparation*.

---

## 🔒 Privacy & Security

**Important**: This repository does NOT contain patient data. Users are responsible for complying with HIPAA regulations and the PhysioNet Data Use Agreement when working with MIMIC-III data. Ensure all PHI is removed or appropriately de-identified before using external embedding APIs.
