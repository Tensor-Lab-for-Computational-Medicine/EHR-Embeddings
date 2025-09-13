### Phase 6 - H2 Analysis: Why and How NM vs SM Succeed and Fail

- **Objective**: Move beyond “Which model is better?” to “Why and how do NM and SM succeed and fail?”
- **Phases**:
  - **Phase IV (What)**: Discover clinical failure phenotypes via subgroup discovery (pysubgroup)
  - **Phase V (Why)**: Explain failures via meta-data structure (density, volatility, sparsity)
- **Sources**: `notebooks/Phase 6 - H2 Analysis/h2a/h2_results`, `notebooks/Phase 6 - H2 Analysis/h2b/h2_results`

### How to read the results (plain language)
- **Agreement/Discordance**: How often models give the same answer on the same patient. High discordance means the models “think” differently.
- **McNemar test**: Checks if one model uniquely gets more patients right than the other. Very small p-value → differences are real, not luck.
- **Cohen’s kappa**: How much they agree beyond chance. 0.49 ≈ moderate; expect frequent disagreement.
- **Risk score correlation**: Do risk scores move together patient-by-patient? Moderate → models add different information.
- **AUROC**: Probability a patient who dies is scored higher risk than one who survives. Higher is better.
- **AUPRC**: Precision when the event is rare. Higher → fewer false alarms among flagged patients.
- **Brier score**: Overall risk accuracy (calibration + sharpness). Lower is better.
- **Thresholds**: Risk cutoffs to say “high” vs “low” risk; chosen to balance misses and false alarms.
- **Coverage (rules)**: How many patients match a rule. Think: “How common is this clinical profile?”
- **Lift (rules)**: How concentrated the target is inside the rule vs overall. 2x lift → twice as common in that profile.
- **WRAcc (rules)**: “Usefulness” of a rule; balances strength (lift) with how common it is (coverage). Bigger → more clinically useful slice.
- **Meta features (Phase V)**:
  - **Density**: How much was measured (features and frequency).
  - **Volatility**: How much values change over time (stability vs lability).
  - **Sparsity**: How many features are absent/zero (thin records).

### H2a (Orthogonal Information): Agreement, Discordance, and Performance
- **They think differently** (McNemar statistic 254.0, p ≈ 5.7e-37): NM gets many right that SM misses, and vice versa; not random.
  - Clinical read: The models pick up different clinical cues.
- **Agreement is limited** (Kappa 0.490, 95% CI [0.462, 0.519]) → moderate beyond chance.
  - Clinical read: Expect meaningful disagreement in practice.
- **Risk scores only moderately align** (r = 0.633, 95% CI [0.608, 0.657]).
  - Clinical read: Combining both can add information.
- **Who ranks risk better?** NM > SM
  - NM: AUROC 0.901, AUPRC 0.556, Brier 0.061
  - SM: AUROC 0.835, AUPRC 0.356, Brier 0.075
  - Clinical read: NM separates high- vs low-risk patients better and is more accurate overall.
- **Risk cutoffs**: Youden; NM θ ≈ 0.196, SM θ ≈ 0.216
  - Clinical read: Cutoffs can be tuned per subgroup (below) to trade misses vs false alarms.

Images

![H2a — Probability scatter](<./notebooks/Phase 6 - H2 Analysis/h2a/h2_results/probability_scatter_plot.png>)

![H2a — Disagreement heatmap](<./notebooks/Phase 6 - H2 Analysis/h2a/h2_results/disagreement_heatmap.png>)

### Phase IV (What): Clinical Failure Phenotypes
- Method: pysubgroup (max depth 3, min support 5%, quality = WRAcc)
- Cohorts: TP 315, TN 4049, FN 105, FP 299; SM: FN 82 / FP 544; NM: FN 69 / FP 185

Top interpretable rules and clinical read:

| Slice | Representative rule (truncated) | Coverage | Lift | Clinical interpretation |
|---|---|---|---:|---|
| SM FN vs TP | creatinine urine ∈ [83,84) AND lactate stddev (24h) ∈ [0,0.06) AND atypical lymphocytes = 0 | 125 (31.5%) | 1.86x | “Quiet labs” phenotype with low variability → SM under-calls risk.
| SM FP vs TN | ascites albumin ∈ [1.55,1.60) AND BUN ≥ 26.5 AND ascites creatinine = 1.0 | 919 (20.0%) | 2.62x | Cirrhosis/renal milieu → SM over-calls risk in liver disease profiles.
| NM FN vs TP | fibrinogen slope (24h) ∈ [0,0.67) AND FiO2 stddev (24h) ∈ [0,0.02) AND tidal volume stddev (24h) ∈ [0,17.27) | 122 (31.8%) | 1.92x | “Deceptively stable” ventilator course → NM misses without volatility.
| NM FP vs TN | many PIP counts AND many PEEP counts AND PA systolic slope (6h) = 0 | 555 (13.1%) | 3.22x | High device activity/monitoring → NM over-calls risk despite stability.

Image

![H2b — Feature distributions](<./notebooks/Phase 6 - H2 Analysis/h2b/h2_results/h2b_feature_distributions.png>)

### Phase V (Why): Data-Structural Drivers of Failures
- Files: `phase_v_meta_results.csv`, `phase_v_report.txt` (36 significant contrasts)
- We compare each error group to its matched “success” group on data structure.

Key contrasts (error vs success medians; arrows show direction in errors):

| Slice | Density | Volatility | Sparsity | Example medians (error vs success) |
|---|---|---|---|---|
| SM FN | ↓ fewer measurements | ↓ more stable | ↑ more absent features | features 51 vs 61; events 276.5 vs 341; stddev 2.81 vs 4.12; prop_zero 0.56 vs 0.47 |
| SM FP | ↑ more measurements | ↑ more labile | ↓ fewer absent features | features 57 vs 52; events 305 vs 277; stddev 3.42 vs 2.62; prop_zero 0.51 vs 0.55 |
| NM FN | ↓ fewer measurements | ↓ more stable | ↑ more absent features | features 53.5 vs 61; events 268.5 vs 341; stddev 2.36 vs 4.12; prop_zero 0.54 vs 0.47 |
| NM FP | ↑ more measurements | ↑ more labile | ↓ fewer absent features | features 60 vs 52; events 336.5 vs 277; stddev 4.70 vs 2.62; prop_zero 0.48 vs 0.55 |

### Who each model misclassifies — and why (clinician-facing)
- **SM will miss (false negatives)**: Patients with “quiet” lab profiles (low variability), fewer things measured, and many absent features.
  - Why: SM underreacts when there isn’t much change or data density to trigger it.
- **SM will false-alarm (false positives)**: Patients with liver/renal profiles and dense, volatile labs.
  - Why: SM overreacts when many labs are measured frequently and fluctuate.
- **NM will miss (false negatives)**: Ventilated patients whose respiratory parameters look very stable (low volatility) and sparsely sampled.
  - Why: NM under-detects risk when signals don’t change over time and data are thin.
- **NM will false-alarm (false positives)**: Heavily monitored ICU patients with lots of ventilator/device activity.
  - Why: NM over-calls risk in high-intensity care settings even when patients are stable.

### What to do next (clinically actionable)
- **Route by regime**: Stricter SM thresholds in dense/volatile regimes; favor NM in sparse/stable regimes.
- **Hybridize**: Simple ensemble or rule-based overrides for the Phase IV phenotypes.
- **Data operations**: Close gaps where a model misses (ensure key ventilatory/lab features are recorded).
- **Monitoring**: Flag the phenotypes in real time to inform triage and risk review.

References
- H2a: `h2a_summary_report.txt`, `h2a_metrics.csv`, disagreement/probability figures
- Phase IV: `h2b_detailed_report.txt`, `h2b_summary_table.csv`, pattern CSVs
- Phase V: `phase_v_meta_results.csv` (primary), `phase_v_report.txt`, `phase_v_meta_debug.txt`
