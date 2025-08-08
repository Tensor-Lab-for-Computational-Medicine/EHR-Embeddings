# XGBoost MODEL4B Performance Summary
## Clinical Prediction Task Results

This table presents the performance of XGBoost MODEL4B across different clinical prediction tasks. All metrics are reported with 95% confidence intervals.

**Performance Metrics:**
- **AUROC**: Area Under the Receiver Operating Characteristic Curve
- **AUPRC**: Area Under the Precision-Recall Curve

| Clinical Prediction Task   | Model Configuration   | AUROC (95% CI)         | AUPRC (95% CI)         |
|:---------------------------|:----------------------|:-----------------------|:-----------------------|
| Hospital Mortality         | MODEL4B_P0            | 0.7079 (0.6883-0.7276) | 0.2062 (0.1843-0.2330) |
| Length-of-Stay > 7 Days    | MODEL4B_P0            | 0.6477 (0.6216-0.6736) | 0.1300 (0.1131-0.1526) |
| Length-of-Stay > 3 Days    | MODEL4B_P0            | 0.6305 (0.6160-0.6439) | 0.5459 (0.5235-0.5673) |
| Mechanical Ventilation     | MODEL4B_P0            | 0.6013 (0.5674-0.6300) | 0.1064 (0.0882-0.1285) |
| 30-Day Readmission         | MODEL4B_P0            | 0.5893 (0.5563-0.6210) | 0.0697 (0.0578-0.0904) |
| Vasopressor Administration | MODEL4B_P0            | 0.5879 (0.5536-0.6227) | 0.0847 (0.0703-0.1045) |

*Generated on: 2025-07-29 08:08:48*
