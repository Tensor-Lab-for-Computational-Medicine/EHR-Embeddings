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

### Clinical synthesis (why these patterns make sense)
- **NM strengths**: Excels with dense, quantitative signals (trends, counts, extremes) common in AKI and in high-acuity monitoring. It likely captures trajectory features (e.g., rising labs) that are hard to convey in short text.
- **SM sensitivities**: Overweights semantic cues like “coagulopathy” and frailty proxies, which can correlate with illness but do not necessarily imply imminent mortality in isolation—leading to false alarms in survivors.
- **Practical implications**: Combine modalities for safety. Use NM to temper SM overcalls in coagulopathy/frailty patterns; use SM to complement NM where semantics add context (not shown in the provided rows).

---

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


