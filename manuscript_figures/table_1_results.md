## Table 1: Full Experimental Results

| Representation         | Prompt                        | AUROC (95% CI)           | AUPRC (95% CI)           |
|:-----------------------|:------------------------------|:-------------------------|:-------------------------|
| Baseline (Numeric)     | Elastic Net                   | 0.8983 (0.8864 - 0.9093) | 0.5549 (0.8864 - 0.9093) |
| Baseline (Numeric)     | XGBoost                       | 0.9080 (0.8968 - 0.9193) | 0.6193 (0.8968 - 0.9193) |
| F1 (Uninterpreted)     | P0 (Control)                  | 0.7781 (0.7577 - 0.7963) | 0.2891 (0.7577 - 0.7963) |
| F1 (Uninterpreted)     | P1 (Task-Specific)            | 0.7786 (0.7602 - 0.7961) | 0.2762 (0.7602 - 0.7961) |
| F1 (Uninterpreted)     | P2 (Persona-Driven)           | 0.7770 (0.7578 - 0.7957) | 0.2982 (0.7578 - 0.7957) |
| F1 (Uninterpreted)     | P3 (Relational-Focus)         | 0.7734 (0.7554 - 0.7918) | 0.2688 (0.7554 - 0.7918) |
| F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | 0.7718 (0.7532 - 0.7895) | 0.2647 (0.7532 - 0.7895) |
| F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | 0.7751 (0.7568 - 0.7935) | 0.2791 (0.7568 - 0.7935) |
| F2 (Interpreted)       | P0 (Control)                  | 0.7833 (0.7662 - 0.8021) | 0.3005 (0.7662 - 0.8021) |
| F2 (Interpreted)       | P1 (Task-Specific)            | 0.7989 (0.7818 - 0.8168) | 0.3133 (0.7818 - 0.8168) |
| F2 (Interpreted)       | P2 (Persona-Driven)           | 0.8023 (0.7841 - 0.8198) | 0.3234 (0.7841 - 0.8198) |
| F2 (Interpreted)       | P3 (Relational-Focus)         | 0.7866 (0.7673 - 0.8045) | 0.3095 (0.7673 - 0.8045) |
| F2 (Interpreted)       | P4 (Acute Dysregulation)      | 0.7930 (0.7741 - 0.8108) | 0.3216 (0.7741 - 0.8108) |
| F2 (Interpreted)       | P5 (Dominant Pathophysiology) | 0.7837 (0.7644 - 0.8014) | 0.3089 (0.7644 - 0.8014) |
| F3 (Narrative Summary) | P0 (Control)                  | 0.8321 (0.8154 - 0.8475) | 0.3767 (0.8154 - 0.8475) |
| F3 (Narrative Summary) | P1 (Task-Specific)            | 0.8336 (0.8172 - 0.8497) | 0.3910 (0.8172 - 0.8497) |
| F3 (Narrative Summary) | P2 (Persona-Driven)           | 0.8384 (0.8217 - 0.8541) | 0.4043 (0.8217 - 0.8541) |
| F3 (Narrative Summary) | P3 (Relational-Focus)         | 0.8338 (0.8177 - 0.8498) | 0.3752 (0.8177 - 0.8498) |
| F3 (Narrative Summary) | P4 (Acute Dysregulation)      | 0.8382 (0.8219 - 0.8554) | 0.4010 (0.8219 - 0.8554) |
| F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | 0.8348 (0.8183 - 0.8502) | 0.3745 (0.8183 - 0.8502) |