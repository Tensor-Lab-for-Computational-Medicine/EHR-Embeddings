### Phase 6 - H2 Analysis: Updated Archetypes (from latest CSVs)

- **Objective**: Summarize current Phase IV/IVB archetypes driving NM vs SM successes/failures.
- **Sources**:
  - Main: `h2b/h2_results/final_archetypes.csv`, `h2b/h2_results/final_archetypes_ivb.csv`
  - Readmission-30: `h2b/h2_results_readmission_30/final_archetypes.csv`, `h2b/h2_results_readmission_30/final_archetypes_ivb.csv`

### Phase IV (What): Representative archetypes — Main dataset

| Slice | Representative rule | Coverage | Lift |
|---|---|---:|---:|
| NM FP vs TN | positive end-expiratory pressure set_count ≥ 5 AND pulmonary artery systolic slope (6h) = 0 | 688 | 2.76 |
| NM FN vs TP | FiO2 stddev (24h) ∈ [0, 0.02) AND tidal volume stddev (24h) ∈ [0, 17.27) | 133 | 1.80 |
| SM FP vs TN | ascites albumin ∈ [1.55, 1.60) AND BUN ≥ 26.5 | 920 | 2.62 |
| SM FN vs TP | creatinine urine ∈ [83, 84) AND lactate stddev (24h) ∈ [0, 0.06) | 130 | 1.79 |

Clinical read (main): NM over-calls with high device intensity + flat PA slope; SM over-calls in hepatic/renal profiles; both miss when volatility is low and records are sparse.

### Phase IVB (Discordance domains) — Main dataset

| Battleground | Advantaged model | Representative rule | Coverage | Lift |
|---|---|---|---:|---:|
| deaths | Numerical Model | basophils = 0.30 AND CVP slope (6h) ∈ [0,0.09) AND atypical lymphocytes count = 0 AND PA systolic slope ∈ [0,0.30) | 54 | 1.60 |
| deaths | Semantic Model | cardiac output (Fick) stddev (24h) ∈ [0,0.38) AND creatinine ≥ 2.0 | 30 | 1.82 |
| survivors | Semantic Model | alkaline phosphatase slope (24h) = 0 AND lactic acid ≈ 1.8 | 230 | 1.75 |

### Phase IV (What): Representative archetypes — Readmission-30

| Slice | Representative rule | Coverage | Lift |
|---|---|---:|---:|
| NM FP vs TN | BUN last ≥ 30.0 AND cardiac output (Fick) mean ≈ 5.75–5.76 | 884 | 2.02 |
| SM FP vs TN | bilirubin last ≥ 0.70 AND LDL last ≈ 87–88 | 903 | 1.81 |
| NM FN vs TP | ascites albumin = 1.55 AND PEEP count = 0 AND troponin-T count = 0 (with demographic interaction) | 26 | 1.26 |

### Phase IVB — Readmission-30

| Battleground | Advantaged model | Representative rule | Coverage | Lift |
|---|---|---|---:|---:|
| deaths | Numerical Model | MCHC stddev (24h) ∈ [0,0.28) AND RR set ≈ 14 | 28 | 1.80 |
| deaths | Semantic Model | chloride urine ∈ [46.5,83] AND creatinine/lactate stddev (24h) low | 28 | 1.29 |
| survivors | Semantic Model | ALT count = 0–1 AND eosinophils last = 2.0 | 482 | 1.28 |

### Takeaways (consistent across datasets)

- **Device/monitoring intensity + flat hemodynamics → NM false alarms.**
- **Sparse/low-volatility regimes → both models can miss.**
- **Hepatic/renal profiles → SM false alarms; renal/liver severity sharpens lift.**

Actionable: calibrate thresholds by regime; consider simple overrides for these phenotypes; ensure key ventilatory/lab features are captured to reduce misses.
