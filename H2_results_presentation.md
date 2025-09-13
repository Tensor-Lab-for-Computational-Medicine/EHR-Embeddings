### Phase 6 - H2 Analysis: Why and How NM vs SM Succeed and Fail

- **Objective**: Move beyond “Which model is better?” to “Why and how do NM and SM succeed and fail?”
- **Plan alignment**:
  - **H1**: Discrimination and calibration comparison (AUROC/AUPRC/Brier).
  - **H2a**: Quantify disagreement (McNemar, Kappa, correlation).
  - **H2b**: Phase IV subgroups = the clinical failure phenotypes (“what”).
  - **H2 (meta)**: Phase V meta-analysis explains failures via data structure (“why”), anchored to Phase IV rules.
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

### Are these subgroups statistically significant (not random)?
- The rules have both **coverage** (common enough to matter) and **lift** (outcome enriched 1.86–3.22x), which would not occur under a random split.
- Independent confirmation from Phase V (non-parametric tests) shows the same subgroups differ on data structure with very small p-values:

| Slice | Significant differences (direction in error group) | p-values (examples) |
|---|---|---| 
| SM FN | Fewer features, fewer events, more absent features, lower volatility | num_features_measured p≈2.16e-11; total_events p≈7.85e-10; unique_families p≈6.33e-12; prop_zero p≈2.16e-11; stddev p≈0.0014 |
| SM FP | More features, more events, fewer absent features, higher volatility | num_features_measured p≈3.05e-28; total_events p≈7.49e-09; unique_families p≈4.91e-35; stddev p≈1.79e-07 |
| NM FN | Fewer features, fewer events, more absent features, lower volatility | num_features_measured p≈2.45e-08; total_events p≈5.27e-09; unique_families p≈4.41e-09; stddev p≈1.95e-04 |
| NM FP | More features, more events, fewer absent features, higher volatility | num_features_measured p≈1.05e-18; total_events p≈6.17e-12; unique_families p≈6.20e-23; stddev p≈2.29e-10 |

- Clinical read: These p-values show the phenotypes are **systematically different**, not artifacts. The same directions repeat across multiple features and both models.

### Phase V (Why): Data-Structural Drivers Anchored to Phase IV Rules
- Files: `phase_v_meta_results.csv`, `phase_v_report.txt` (36 significant contrasts)
- Analysis is performed WITHIN each discovered Phase IV rule-defined subgroup vs its corresponding concordant success cohort (not broad FN/FP alone).

Per-rule meta-feature contrasts (medians in error subgroup vs success; p-values):

- **SM FN — Rule #1** (creatinine urine [83–84), lactate stddev 24h [0–0.06), atypical lymphocytes = 0)
  - Density: features 51 vs 61 (p≈2.16e-11), events 276.5 vs 341 (p≈7.85e-10), families 45 vs 55 (p≈6.33e-12)
  - Sparsity: prop_zero 0.56 vs 0.47 (p≈2.16e-11)
  - Volatility: stddev 2.81 vs 4.12 (p≈0.0014)
- **SM FN — Rule #2** (AST mean [40–59), creatinine urine [83–84), LDH slope 24h [0–0.19))
  - Density: features 53 vs 61 (p≈6.96e-12), events 284 vs 341 (p≈1.58e-07), families 47 vs 55 (p≈7.78e-12)
  - Sparsity: prop_zero 0.54 vs 0.47 (p≈6.96e-12)
  - Volatility: stddev 3.32 vs 4.12 (p≈0.039)
- **SM FP — Rule #1** (ascites albumin [1.55–1.60), BUN ≥ 26.5, ascites creatinine = 1.0)
  - Density: features 57 vs 52 (p≈3.05e-28), events 305 vs 277 (p≈7.49e-09), families 51 vs 45 (p≈4.91e-35)
  - Sparsity: prop_zero 0.51 vs 0.55 (p≈3.05e-28)
  - Volatility: stddev 3.42 vs 2.62 (p≈1.79e-07)
- **NM FN — Rule #1** (fibrinogen slope 24h [0–0.67), FiO2 stddev 24h [0–0.02), tidal volume stddev 24h [0–17.27))
  - Density: features 53.5 vs 61 (p≈2.45e-08), events 268.5 vs 341 (p≈5.27e-09), families 46.5 vs 55 (p≈4.41e-09)
  - Sparsity: prop_zero 0.54 vs 0.47 (p≈2.45e-08)
  - Volatility: stddev 2.36 vs 4.12 (p≈1.95e-04)
- **NM FP — Rule #1** (PIP count ≥ 4, PEEP count ≥ 5, PA systolic slope 6h = 0)
  - Density: features 60 vs 52 (p≈1.05e-18), events 336.5 vs 277 (p≈6.17e-12), families 54 vs 45 (p≈6.20e-23)
  - Sparsity: prop_zero 0.48 vs 0.55 (p≈1.05e-18)
  - Volatility: stddev 4.70 vs 2.62 (p≈2.29e-10)

Plain-language takeaway: Phase V confirms, per specific Phase IV rules, that failures arise in distinct data regimes (sparse/stable vs dense/labile). This explains “why” the clinical phenotypes are misclassified.

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
