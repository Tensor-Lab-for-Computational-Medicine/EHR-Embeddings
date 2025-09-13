### Phase 6 - H2 Analysis: Why and How NM vs SM Succeed and Fail

- **Objective**: Move beyond “Which model is better?” to “Why and how do NM and SM succeed and fail?”
- **Phases**:
  - **Phase IV (What)**: Discover clinical failure phenotypes via subgroup discovery (pysubgroup)
  - **Phase V (Why)**: Explain failures via meta-data structure (density, volatility, sparsity)
- **Sources**: `notebooks/Phase 6 - H2 Analysis/h2a/h2_results`, `notebooks/Phase 6 - H2 Analysis/h2b/h2_results`

### How to read the results (plain language)
- **Agreement/Discordance**: How often models give the same answer on the same patient. High discordance means the models "think" differently.
- **McNemar test**: Asks if one model uniquely gets more patients right than the other. If yes (very small p-value), they succeed on different types of patients.
- **Cohen’s kappa**: How often they agree beyond chance. 0.49 ≈ moderate; they sometimes agree, often they don’t.
- **Risk score correlation**: Do their predicted risks move together patient-by-patient? Moderate correlation means they are not redundant.
- **AUROC**: Chance that a patient who dies is ranked higher risk than one who survives (higher is better).
- **AUPRC**: How precise alerts are when the outcome is rare (higher means fewer false alarms among positives).
- **Brier score**: Overall risk accuracy; lower means probabilities match reality better (calibration + sharpness).
- **Thresholds**: Risk cutoffs used to say "high" vs "low" risk; here chosen to balance sensitivity and specificity.
- **WRAcc / Lift / Coverage (Phase IV rules)**:
  - **WRAcc**: How strongly a rule isolates the target group beyond average.
  - **Lift**: How many times more common the target is inside the rule vs overall.
  - **Coverage**: How much of the cohort the rule describes.
- **Meta features (Phase V)**:
  - **Density**: How much data is available (how many things were measured, how often).
  - **Volatility**: How much those measurements changed over time (clinical stability vs lability).
  - **Sparsity**: How many features are zero/absent (thin records, missing signals).

### H2a (Orthogonal Information): Agreement, Discordance, and Performance
- **They think differently** (McNemar statistic 254.0, p ≈ 5.7e-37): One model (NM) gets many patients right that the other (SM) misses, and vice versa; this is not random.
  - Clinical read: The models capture different clinical cues.
- **Agreement is limited** (Kappa 0.490, 95% CI [0.462, 0.519]): Moderate agreement beyond chance.
  - Clinical read: Expect meaningful disagreement in day-to-day use.
- **Risk scores are only moderately aligned** (r = 0.633, 95% CI [0.608, 0.657]).
  - Clinical read: Combining both can add information.
- **Who is better at ranking risk?** NM > SM
  - NM: AUROC 0.901, AUPRC 0.556, Brier 0.061
  - SM: AUROC 0.835, AUPRC 0.356, Brier 0.075
  - Clinical read: NM separates high- vs low-risk patients better and is better calibrated.
- **Risk cutoffs**: Youden; NM θ ≈ 0.196, SM θ ≈ 0.216
  - Clinical read: Thresholds balance misses vs false alarms; they can be tuned per subgroup (see below).

Images

![H2a — Probability scatter](notebooks/Phase%206%20-%20H2%20Analysis/h2a/h2_results/probability_scatter_plot.png)

![H2a — Disagreement heatmap](notebooks/Phase%206%20-%20H2%20Analysis/h2a/h2_results/disagreement_heatmap.png)

### Phase IV (What): Clinical Failure Phenotypes
- Method: pysubgroup (max depth 3, min support 5%, quality = WRAcc)
- Cohorts: TP 315, TN 4049, FN 105, FP 299; SM: FN 82 / FP 544; NM: FN 69 / FP 185

Top interpretable rules and clinical read:

| Slice | Representative rule (truncated) | Coverage | Lift | Clinical interpretation |
|---|---|---|---:|---|
| SM FN vs TP | creatinine urine ∈ [83,84) AND lactate stddev (24h) ∈ [0,0.06) AND atypical lymphocytes = 0 | 125 (31.5%) | 1.86x | "Quiet labs" phenotype with low variability; SM under-calls risk when labs look stable.
| SM FP vs TN | ascites albumin ∈ [1.55,1.60) AND BUN ≥ 26.5 AND ascites creatinine = 1.0 | 919 (20.0%) | 2.62x | Cirrhosis/renal milieu; SM over-calls risk in advanced liver disease profiles.
| NM FN vs TP | fibrinogen slope (24h) ∈ [0,0.67) AND FiO2 stddev (24h) ∈ [0,0.02) AND tidal volume stddev (24h) ∈ [0,17.27) | 122 (31.8%) | 1.92x | "Deceptively stable" on ventilator (little change); NM misses deteriorations without volatility.
| NM FP vs TN | many PIP counts AND many PEEP counts AND PA systolic slope (6h) = 0 | 555 (13.1%) | 3.22x | Heavy monitoring/intervention; NM over-calls risk in high-intensity care even when stable.

Image

![H2b — Feature distributions](notebooks/Phase%206%20-%20H2%20Analysis/h2b/h2_results/h2b_feature_distributions.png)

### Phase V (Why): Data-Structural Drivers of Failures
- Files: `phase_v_meta_results.csv`, `phase_v_report.txt` (36 significant contrasts)
- We compare each error group to its matched "success" group on meta features that reflect how data are recorded.

Key contrasts (error vs success medians; arrows show direction in errors):

| Slice | Density | Volatility | Sparsity | Example medians (error vs success) |
|---|---|---|---|---|
| SM FN | ↓ fewer measurements | ↓ more stable | ↑ more absent features | features 51 vs 61; events 276.5 vs 341; stddev 2.81 vs 4.12; prop_zero 0.56 vs 0.47 |
| SM FP | ↑ more measurements | ↑ more labile | ↓ fewer absent features | features 57 vs 52; events 305 vs 277; stddev 3.42 vs 2.62; prop_zero 0.51 vs 0.55 |
| NM FN | ↓ fewer measurements | ↓ more stable | ↑ more absent features | features 53.5 vs 61; events 268.5 vs 341; stddev 2.36 vs 4.12; prop_zero 0.54 vs 0.47 |
| NM FP | ↑ more measurements | ↑ more labile | ↓ fewer absent features | features 60 vs 52; events 336.5 vs 277; stddev 4.70 vs 2.62; prop_zero 0.48 vs 0.55 |

Plain-language takeaways (linking Phase IV → V):
- **SM false alarms** in cirrhosis-like profiles happen with dense, volatile labs → SM is reactive when data are busy.
- **SM misses** when labs look quiet and sparse → SM underreacts without variability.
- **NM false alarms** in heavily monitored ventilated patients with high-intensity care → NM overreacts to density and device activity.
- **NM misses** when the ventilatory course looks very stable and data are thin → NM under-detects risk without change over time.

### What to do next (clinically actionable)
- **Route by regime**: Use stricter SM thresholds in dense/volatile regimes; favor NM in sparse/stable regimes.
- **Hybridize**: Simple ensemble or rule-based overrides for the Phase IV phenotypes above.
- **Data operations**: Close gaps where a model misses (e.g., ensure key ventilatory/lab features are recorded).
- **Monitoring**: Flag the phenotypes in real time to inform triage and risk review.

References
- H2a: `h2a_summary_report.txt`, `h2a_metrics.csv`, disagreement/probability figures
- Phase IV: `h2b_detailed_report.txt`, `h2b_summary_table.csv`, pattern CSVs
- Phase V: `phase_v_meta_results.csv` (primary), `phase_v_report.txt`, `phase_v_meta_debug.txt`
