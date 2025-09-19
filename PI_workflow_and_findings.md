## Comparative Risk Prediction Study: Workflow and Findings (PI Brief)

Intent: Summarize the study design and interpret key archetypes found in the discordance analyses, in plain language for a cardiology audience.

### TL;DR
- **Numerical Model (NM) false alarms** clustered in patients with **AKI**, especially when combined with **synthetic liver dysfunction** or **neutrophilia**.
- **Semantic Model (SM) false alarms** clustered when **anticoagulation derangement** co-occurred with **malnutrition**, **neutrophilia**, or **synthetic liver dysfunction**.
- In the survivors battleground (IVB), subgroups with **invasive hemodynamic monitoring** plus coagulopathy or metabolic derangements showed a **clear NM advantage**.
- Where a section below has “None reported,” it means no statistically significant archetype met criteria in the provided results.

---

### Study Overview (What we built and why)
We compared two complementary approaches to predict in-hospital mortality from EHR data collected in the first 24 hours of ICU care:
- **Numerical Model (NM):** XGBoost trained on curated numerical features (means, trends, counts) engineered from vitals and labs.
- **Semantic Model (SM):** LLM embeddings created from structured text serializations of the same data, optionally with clinical interpretations (e.g., abnormal flags), then fed to a classifier.

Both models were calibrated, evaluated with AUROC/AUPRC/Brier score, and then probed for where they disagree. We used subgroup discovery to find clinically interpretable “archetypes” that explain when each model fails or wins.

```mermaid
flowchart TD
  A[Data: First 24h ICU events<br/>MIMIC-Extract] --> B[Phase I: Feature profiling<br/>and categorization]
  B --> C[Phase II: Curated numerical features<br/>XGBoost (NM)]
  A --> D[Phase III: Text serializations + prompts<br/>LLM embeddings (SM)]
  C --> E[Calibration]
  D --> E
  E --> F[Phase IV: Discordance analysis<br/>(Youden's J thresholds)]
  F --> G[Subgroup discovery (pysubgroup)<br/>Archetype selection]
  G --> H[Phase V: Meta-analysis on data structure]
  H --> I[Reporting]
```

---

### Plain-language glossary (jargon → meaning)
- **LLM embeddings**: Numeric vectors summarizing the meaning of input text; used as model inputs.
- **Calibration**: Aligning predicted probabilities with observed outcomes (important for risk use).
- **Youden’s J threshold**: A cut-point chosen to balance sensitivity and specificity on training data.
- **Concordant**: Both models correct (TP) or both correct negatives (TN).
- **Discordant**: Models disagree; one is right, one is wrong.
- **False alarm (FP)**: Model predicts death, but patient survives.
- **Lift**: How much more common the target is inside a subgroup vs its baseline; 2.0 = twice as likely.
- **WRAcc**: A quality score balancing subgroup size and signal strength; higher is better.
- Clinical terms:
  - **AKI**: Acute kidney injury.
  - **Synthetic liver dysfunction**: Impaired liver synthesis (e.g., abnormal INR/albumin).
  - **Anticoagulation derangement**: Coagulation tests suggesting bleeding/thrombosis risk or drug effect.
  - **Neutrophilia**: Elevated neutrophil count.
  - **Invasive hemodynamic monitoring**: Arterial line/PA catheter data indicating high-acuity management.

---

### Results: H2b Differential – False Alarms vs Concordant True Negatives
Context: For each model separately, we compare its false alarms against cases where both models were correct negatives. “Target share” is the false-alarm rate in the subgroup; “baseline rate” is the overall false-alarm rate in the comparison cohort. Lower q-values indicate stronger statistical evidence.

#### Numerical Model (NM) – False Alarms
- **AKI (any)**
  - Coverage: 1132 patients (22.8%); Lift 2.25; Target 6.4% vs Baseline 2.8%; q 2.17e-13
  - Clinical takeaway: NM overcalls risk in AKI broadly.
- **AKI + Synthetic liver dysfunction**
  - Coverage: 331 (6.7%); Lift 4.60; Target 13.0% vs 2.8%; q 8.51e-18
  - Takeaway: Dual organ dysfunction markedly increases NM false alarms.
- **AKI + Neutrophilia**
  - Coverage: 649 (13.1%); Lift 2.57; Target 7.2% vs 2.8%; q 2.41e-10
  - Takeaway: Inflammatory AKI patterns drive NM overprediction.

#### Semantic Model (SM) – False Alarms
- **Anticoagulation derangement + Malnutrition proxy**
  - Coverage: 739 (15.1%); Lift 4.08; Target 6.5% vs 1.6%; q 6.30e-19
  - Takeaway: SM overcalls risk when coagulopathy co-exists with frailty/malnutrition signals.
- **Anticoagulation derangement + Neutrophilia**
  - Coverage: 1460 (29.8%); Lift 2.49; Target 4.0% vs 1.6%; q 5.94e-15
  - Takeaway: Inflammatory coagulopathy patterns prompt SM false alarms.
- **Anticoagulation derangement + Synthetic liver dysfunction**
  - Coverage: 872 (17.8%); Lift 3.67; Target 5.8% vs 1.6%; q 6.30e-19
  - Takeaway: Hepatic synthetic failure plus coagulopathy strongly biases SM toward overprediction.

Notes:
- No additional false-negative archetypes were reported in the provided file.

---

### Results: IVB Discordance – Survivors Battleground (NM Advantage)
Context: Among survivors where NM and SM disagree, these subgroups are where NM is more often correct than SM (higher “target share” vs battleground baseline).

- **Anticoagulation derangement + Invasive hemodynamic monitoring**
  - Coverage: 72 (33.0% of battleground); Lift 1.75; NM wins 62.5% vs Baseline 35.8%; q 2.29e-06
  - Interpretation: In high-acuity monitored patients with coagulopathy, NM better identifies true survivors (avoids false death calls).
- **Invasive hemodynamic monitoring + Malnutrition proxy**
  - Coverage: 50 (22.9%); Lift 1.90; 68.0% vs 35.8%; q 1.03e-05
  - Interpretation: Frailty signals with invasive monitoring favor NM.
- **Hypocalcemia (ionized) + Invasive hemodynamic monitoring**
  - Coverage: 46 (21.1%); Lift 1.94; 69.6% vs 35.8%; q 1.03e-05
  - Interpretation: Metabolic derangement in closely monitored patients favors NM.

Notes:
- No significant archetypes were reported for other IVB battlegrounds in the provided file.

---

### Readmission-30 Results (Secondary analysis)
Source files:
- H2b false alarms (readmission): `notebooks/Phase 6 - H2 Analysis/h2b/h2_results_readmission_30/final_archetypes.csv`
- IVB survivors battleground (readmission): `notebooks/Phase 6 - H2 Analysis/h2b/h2_results_readmission_30/final_archetypes_ivb.csv`

#### H2b Differential – SM False Alarms (Readmission)
- **Malnutrition proxy**
  - Coverage: 1086 (21.8%); Lift 1.61; Target 20.8% vs 12.9%; q 3.19e-15
  - Takeaway: SM overcalls readmission risk in low-albumin patients.
- **Neutrophilia + Malnutrition**
  - Coverage: 742 (14.9%); Lift 1.78; Target 22.9% vs 12.9%; q 7.24e-15
  - Takeaway: Inflammatory frailty pattern heightens SM false alarms.
- **Coagulopathy + Neutrophilia + Malnutrition**
  - Coverage: 592 (11.9%); Lift 1.81; Target 23.3% vs 12.9%; q 2.74e-12
  - Takeaway: Coagulopathy layered onto inflammation and malnutrition further biases SM upward.

Notes:
- Patterns mirror mortality task: SM overweights frailty/inflammation/coagulopathy signals without clear corroborating trajectories.

#### IVB Discordance – Survivors Battleground (SM Advantage, Readmission)
- **Severe AKI**
  - Coverage: 154 (15.8% of battleground); Lift 1.58; SM wins 53.9% vs Baseline 34.1%; q 7.77e-06
  - Interpretation: For survivors with severe AKI, SM more often avoids false readmission calls by NM (or more often is correct vs NM, per battleground definition).
- **Severe AKI + Coagulopathy**
  - Coverage: 124 (12.7%); Lift 1.54; 52.4% vs 34.1%; q 3.10e-04
  - Interpretation: When renal dysfunction and coagulopathy co-occur, SM has an edge on survivors.
- **AKI Stage 2; Severe AKI + Organ dysfunction score [1–2)**
  - Coverage: 106 (10.9%) and 45 (4.6%); Lifts ~1.60–2.08; q values 1.54e-04 to 2.56e-05
  - Interpretation: SM advantage persists across moderate-to-severe renal dysfunction strata.

Notes:
- Additional findings involve anemia + severe AKI and HR volatility with AKI; both consistent with SM leveraging semantic context in complex renal cases.

---

### Clinical synthesis (why these patterns make sense)
- **NM strengths**: Excels with dense, quantitative signals (trends, counts, extremes) common in AKI and in high-acuity monitoring. It likely captures trajectory features (e.g., rising labs) that are hard to convey in short text.
- **SM sensitivities**: Overweights semantic cues like “coagulopathy” and frailty proxies, which can correlate with illness but do not necessarily imply imminent mortality in isolation—leading to false alarms in survivors.
- **Practical implications**: Combine modalities for safety. Use NM to temper SM overcalls in coagulopathy/frailty patterns; use SM to complement NM where semantics add context (not shown in the provided rows).

---

### Phenotype rule glossary for used features (how subgroups were constructed)
Concise logic for phenotypes appearing in significant archetypes (selected items):
- **is_malnourished_proxy**: albumin measured and < 3.5 g/dL.
- **has_neutrophilia**: neutrophils measured and > 7.7 x10^9/L.
- **has_anticoagulation_derangement**: INR > 1.1 or PTT > 35 sec if measured.
- **sirs_wbc_criterion**: WBC > 12k or < 4k if measured.
- **has_sirs**: ≥2 of temp, HR, RR, WBC SIRS criteria.
- **liver_dysfunction_type == 'Synthetic_Dysfunction'**: INR > 1.5 or albumin < 3.0 if measured.
- **has_any_aki**: creatinine above sex-specific ULN with measurement.
- **has_severe_aki**: creatinine ≥ 2.0 mg/dL with measurement.
- **aki_severity_stage**: Stage_2 if creatinine ≥ 2.0; Stage_3 if ≥ 4.0; Stage_1 if has_any_aki.
- **organ_dysfunction_score**: Count of severe failures among: severe AKI, ventilation, lactic acidosis/hyperlactatemia, hepatic synthetic dysfunction, severe thrombocytopenia, severe GCS impairment.
- **anemia_severity == 'Moderate'**: hemoglobin < 10 g/dL (sex-adjusted lower bounds also considered for Mild).
- **has_invasive_hemo_monitoring**: CVP or PA catheter counts > preset thresholds.
- **has_hypocalcemia_ionized**: ionized calcium < 4.6 mg/dL if measured.
- **hr_volatility**: 24h HR stddev when ≥2 HR measures exist.

Full rule source: `notebooks/Phase 6 - H2 Analysis/feature_engineering/feature_rules.csv`.

---

### Anticipated PI Q&A
- What’s the headline? — NM overcalls mortality in AKI (esp. with hepatic dysfunction/inflammation); SM overcalls in coagulopathy/frailty patterns. In readmission, SM shows advantage among survivors with severe AKI.
- Are these clinically plausible? — Yes: AKI drives lab extremes/trends (NM-friendly); coagulopathy/malnutrition are risk markers but not determinative without trajectories (SM-sensitive cues).
- Did you correct for multiple testing? — Yes, BH-FDR; we report q-values and retain q<0.05 as significant in discovery/Phase V summaries.
- Any data leakage? — No. Thresholds set on full training set; all subgroup discovery and archetype selection used training; final metrics reported on held-out test.
- Why Youden’s J? — Provides a stable, symmetric operating point to define binary cohorts for discordance without optimizing on test.
- Could token count or data density confound SM? — Phase V meta includes density/volatility/imputation/token features; we test for significant shifts between error vs success cohorts.
- How robust are these archetypes? — We pruned by significance, coverage, and redundancy; then validated lift direction on test and, for readmission, observed consistent patterns.
- What’s next to make this actionable? — Calibrate per subgroup, adjust SM prompts to de-emphasize isolated coagulopathy/frailty, and evaluate fusion models within battlegrounds.
- External validity? — Plan to replicate on an external ICU dataset; if consistent, consider prospective silent deployment.
- Deployment concerns? — Guardrails: calibration monitoring, shift detection, fail-safe defaults, clinician-in-the-loop review for high-risk flags.

### Limitations and guardrails
- Single-dataset context (MIMIC-derived); external validity not yet proven.
- Thresholds chosen on full training set to define cohorts (not a trained parameter, but still a design choice).
- Archetype discovery controlled for multiple testing (FDR) but remains post hoc.
- Absent sections mean no significant archetypes passed filters—not proof of absence.

---

### Next steps (actionable)
- Validate archetypes on an external cohort; re-check Lift and calibration.
- Targeted model improvements:
  - For NM: mitigate AKI-driven false alarms (e.g., feature regularization or AKI-aware calibration).
  - For SM: adjust prompts/serialization to de-emphasize isolated coagulopathy and frailty signals unless corroborated by trajectories.
- Evaluate the hybrid fusion models on these subgroups to test synergy in exactly the battlegrounds where each model struggles.

---

### Repro notes
- Protocol detailed in `analysis plan.txt` (Phases I–V).
- Archetype sources:
  - H2b false alarms: `notebooks/Phase 6 - H2 Analysis/h2b/h2_results/final_archetypes.csv`
  - IVB survivors battleground (NM advantage): `notebooks/Phase 6 - H2 Analysis/h2b/h2_results/final_archetypes_ivb.csv`
- Key metrics reported per row: coverage (n, %), lift, subgroup target share vs baseline, q-value.


