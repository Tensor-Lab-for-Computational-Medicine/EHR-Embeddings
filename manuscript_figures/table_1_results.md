## Table 1: Full Experimental Results

| Representation         | Prompt                        | Model              | AUROC (95% CI)           | AUPRC (95% CI)           |
|:-----------------------|:------------------------------|:-------------------|:-------------------------|:-------------------------|
| Baseline (Numeric)     | Elastic Net                   | ElasticNet         | 0.8983 (0.8864 - 0.9093) | 0.5549 (0.5189 - 0.5932) |
| Baseline (Numeric)     | XGBoost                       | XGBoost            | 0.9080 (0.8968 - 0.9193) | 0.6193 (0.5850 - 0.6559) |
| F1 (Uninterpreted)     | P0 (Control)                  | embedding-001      | 0.7788 (0.7612 - 0.7956) | 0.2755 (0.2462 - 0.3136) |
| F1 (Uninterpreted)     | P0 (Control)                  | text-embedding-004 | 0.7781 (0.7577 - 0.7963) | 0.2891 (0.2575 - 0.3262) |
| F1 (Uninterpreted)     | P0 (Control)                  | text-embedding-005 | 0.7035 (0.6820 - 0.7247) | 0.2156 (0.1902 - 0.2451) |
| F1 (Uninterpreted)     | P1 (Task-Specific)            | embedding-001      | 0.7798 (0.7608 - 0.7979) | 0.2804 (0.2485 - 0.3157) |
| F1 (Uninterpreted)     | P1 (Task-Specific)            | text-embedding-004 | 0.7786 (0.7602 - 0.7961) | 0.2762 (0.2464 - 0.3110) |
| F1 (Uninterpreted)     | P1 (Task-Specific)            | text-embedding-005 | 0.7372 (0.7156 - 0.7581) | 0.2507 (0.2198 - 0.2827) |
| F1 (Uninterpreted)     | P2 (Persona-Driven)           | embedding-001      | 0.7753 (0.7571 - 0.7920) | 0.2814 (0.2507 - 0.3170) |
| F1 (Uninterpreted)     | P2 (Persona-Driven)           | text-embedding-004 | 0.7770 (0.7578 - 0.7957) | 0.2982 (0.2653 - 0.3362) |
| F1 (Uninterpreted)     | P2 (Persona-Driven)           | text-embedding-005 | 0.7357 (0.7143 - 0.7566) | 0.2505 (0.2171 - 0.2851) |
| F1 (Uninterpreted)     | P3 (Relational-Focus)         | embedding-001      | 0.7727 (0.7548 - 0.7902) | 0.2893 (0.2583 - 0.3235) |
| F1 (Uninterpreted)     | P3 (Relational-Focus)         | text-embedding-004 | 0.7734 (0.7554 - 0.7918) | 0.2688 (0.2397 - 0.3040) |
| F1 (Uninterpreted)     | P3 (Relational-Focus)         | text-embedding-005 | 0.7333 (0.7114 - 0.7541) | 0.2505 (0.2176 - 0.2873) |
| F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | embedding-001      | 0.7707 (0.7523 - 0.7884) | 0.2731 (0.2430 - 0.3102) |
| F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | text-embedding-004 | 0.7718 (0.7532 - 0.7895) | 0.2647 (0.2360 - 0.3000) |
| F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | text-embedding-005 | 0.7399 (0.7198 - 0.7590) | 0.2558 (0.2243 - 0.2897) |
| F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | embedding-001      | 0.7651 (0.7461 - 0.7832) | 0.2875 (0.2565 - 0.3244) |
| F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | text-embedding-004 | 0.7751 (0.7568 - 0.7935) | 0.2791 (0.2483 - 0.3153) |
| F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | text-embedding-005 | 0.7302 (0.7084 - 0.7519) | 0.2558 (0.2220 - 0.2928) |
| F2 (Interpreted)       | P0 (Control)                  | embedding-001      | 0.7974 (0.7790 - 0.8161) | 0.3216 (0.2871 - 0.3596) |
| F2 (Interpreted)       | P0 (Control)                  | text-embedding-004 | 0.7833 (0.7662 - 0.8021) | 0.3005 (0.2675 - 0.3394) |
| F2 (Interpreted)       | P0 (Control)                  | text-embedding-005 | 0.7466 (0.7254 - 0.7672) | 0.2465 (0.2184 - 0.2808) |
| F2 (Interpreted)       | P1 (Task-Specific)            | embedding-001      | 0.7989 (0.7809 - 0.8161) | 0.3217 (0.2882 - 0.3623) |
| F2 (Interpreted)       | P1 (Task-Specific)            | text-embedding-004 | 0.7989 (0.7818 - 0.8168) | 0.3133 (0.2798 - 0.3515) |
| F2 (Interpreted)       | P1 (Task-Specific)            | text-embedding-005 | 0.7530 (0.7331 - 0.7728) | 0.2466 (0.2216 - 0.2800) |
| F2 (Interpreted)       | P2 (Persona-Driven)           | embedding-001      | 0.7921 (0.7730 - 0.8102) | 0.3065 (0.2730 - 0.3443) |
| F2 (Interpreted)       | P2 (Persona-Driven)           | text-embedding-004 | 0.8023 (0.7841 - 0.8198) | 0.3234 (0.2888 - 0.3652) |
| F2 (Interpreted)       | P2 (Persona-Driven)           | text-embedding-005 | 0.7436 (0.7219 - 0.7636) | 0.2578 (0.2274 - 0.2933) |
| F2 (Interpreted)       | P3 (Relational-Focus)         | embedding-001      | 0.7974 (0.7794 - 0.8146) | 0.3229 (0.2878 - 0.3635) |
| F2 (Interpreted)       | P3 (Relational-Focus)         | text-embedding-004 | 0.7866 (0.7673 - 0.8045) | 0.3095 (0.2724 - 0.3515) |
| F2 (Interpreted)       | P3 (Relational-Focus)         | text-embedding-005 | 0.7548 (0.7341 - 0.7746) | 0.2653 (0.2338 - 0.2995) |
| F2 (Interpreted)       | P4 (Acute Dysregulation)      | embedding-001      | 0.7894 (0.7711 - 0.8076) | 0.3164 (0.2818 - 0.3583) |
| F2 (Interpreted)       | P4 (Acute Dysregulation)      | text-embedding-004 | 0.7930 (0.7741 - 0.8108) | 0.3216 (0.2860 - 0.3619) |
| F2 (Interpreted)       | P4 (Acute Dysregulation)      | text-embedding-005 | 0.7333 (0.7109 - 0.7551) | 0.2626 (0.2302 - 0.3004) |
| F2 (Interpreted)       | P5 (Dominant Pathophysiology) | embedding-001      | 0.7906 (0.7724 - 0.8084) | 0.3087 (0.2768 - 0.3467) |
| F2 (Interpreted)       | P5 (Dominant Pathophysiology) | text-embedding-004 | 0.7837 (0.7644 - 0.8014) | 0.3089 (0.2728 - 0.3506) |
| F2 (Interpreted)       | P5 (Dominant Pathophysiology) | text-embedding-005 | 0.7424 (0.7210 - 0.7638) | 0.2574 (0.2259 - 0.2925) |
| F3 (Narrative Summary) | P0 (Control)                  | embedding-001      | 0.8131 (0.7970 - 0.8288) | 0.3223 (0.2903 - 0.3615) |
| F3 (Narrative Summary) | P0 (Control)                  | text-embedding-004 | 0.8321 (0.8154 - 0.8475) | 0.3767 (0.3389 - 0.4171) |
| F3 (Narrative Summary) | P0 (Control)                  | text-embedding-005 | 0.8129 (0.7958 - 0.8302) | 0.3431 (0.3075 - 0.3838) |
| F3 (Narrative Summary) | P1 (Task-Specific)            | embedding-001      | 0.8254 (0.8092 - 0.8415) | 0.3671 (0.3303 - 0.4081) |
| F3 (Narrative Summary) | P1 (Task-Specific)            | text-embedding-004 | 0.8336 (0.8172 - 0.8497) | 0.3910 (0.3531 - 0.4338) |
| F3 (Narrative Summary) | P1 (Task-Specific)            | text-embedding-005 | 0.8140 (0.7987 - 0.8320) | 0.3344 (0.3019 - 0.3754) |
| F3 (Narrative Summary) | P2 (Persona-Driven)           | embedding-001      | 0.8161 (0.7996 - 0.8330) | 0.3378 (0.3022 - 0.3776) |
| F3 (Narrative Summary) | P2 (Persona-Driven)           | text-embedding-004 | 0.8384 (0.8217 - 0.8541) | 0.4043 (0.3657 - 0.4480) |
| F3 (Narrative Summary) | P2 (Persona-Driven)           | text-embedding-005 | 0.8007 (0.7836 - 0.8182) | 0.3198 (0.2869 - 0.3602) |
| F3 (Narrative Summary) | P3 (Relational-Focus)         | embedding-001      | 0.8288 (0.8135 - 0.8449) | 0.3503 (0.3141 - 0.3902) |
| F3 (Narrative Summary) | P3 (Relational-Focus)         | text-embedding-004 | 0.8338 (0.8177 - 0.8498) | 0.3752 (0.3384 - 0.4145) |
| F3 (Narrative Summary) | P3 (Relational-Focus)         | text-embedding-005 | 0.8139 (0.7962 - 0.8315) | 0.3409 (0.3066 - 0.3817) |
| F3 (Narrative Summary) | P4 (Acute Dysregulation)      | embedding-001      | 0.8223 (0.8060 - 0.8381) | 0.3394 (0.3038 - 0.3797) |
| F3 (Narrative Summary) | P4 (Acute Dysregulation)      | text-embedding-004 | 0.8382 (0.8219 - 0.8554) | 0.4010 (0.3602 - 0.4426) |
| F3 (Narrative Summary) | P4 (Acute Dysregulation)      | text-embedding-005 | 0.8139 (0.7968 - 0.8314) | 0.3377 (0.3013 - 0.3768) |
| F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | embedding-001      | 0.8264 (0.8102 - 0.8423) | 0.3446 (0.3088 - 0.3856) |
| F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | text-embedding-004 | 0.8348 (0.8183 - 0.8502) | 0.3745 (0.3370 - 0.4167) |
| F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | text-embedding-005 | 0.8141 (0.7969 - 0.8312) | 0.3202 (0.2891 - 0.3584) |