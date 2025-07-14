## Table 1: Full Experimental Results

| Task              | Representation         | Prompt                        | Model                   | AUROC (95% CI)           | AUPRC (95% CI)           |
|:------------------|:-----------------------|:------------------------------|:------------------------|:-------------------------|:-------------------------|
| mort_hosp         | Baseline (Numeric)     | XGBoost                       | XGBoost                 | 0.9107 (0.8998 - 0.9213) | 0.6143 (0.5802 - 0.6484) |
| mort_hosp         | Baseline (Numeric)     | Elastic Net                   | ElasticNet              | 0.8989 (0.8874 - 0.9104) | 0.5604 (0.5253 - 0.5979) |
| mort_hosp         | F1 (Uninterpreted)     | P0 (Control)                  | text-embedding-004      | 0.7780 (0.7589 - 0.7968) | 0.2882 (0.2560 - 0.3255) |
| mort_hosp         | F1 (Uninterpreted)     | P1 (Task-Specific)            | text-embedding-004      | 0.7743 (0.7570 - 0.7941) | 0.2824 (0.2520 - 0.3213) |
| mort_hosp         | F1 (Uninterpreted)     | P2 (Persona-Driven)           | text-embedding-004      | 0.7823 (0.7640 - 0.8005) | 0.2847 (0.2537 - 0.3241) |
| mort_hosp         | F1 (Uninterpreted)     | P3 (Relational-Focus)         | text-embedding-004      | 0.7745 (0.7575 - 0.7929) | 0.2701 (0.2415 - 0.3078) |
| mort_hosp         | F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | text-embedding-004      | 0.7759 (0.7575 - 0.7934) | 0.2818 (0.2495 - 0.3197) |
| mort_hosp         | F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.7752 (0.7566 - 0.7930) | 0.2812 (0.2473 - 0.3169) |
| mort_hosp         | F2 (Interpreted)       | P0 (Control)                  | text-embedding-004      | 0.7998 (0.7822 - 0.8176) | 0.3214 (0.2879 - 0.3616) |
| mort_hosp         | F2 (Interpreted)       | P1 (Task-Specific)            | text-embedding-004      | 0.7887 (0.7700 - 0.8070) | 0.3069 (0.2709 - 0.3462) |
| mort_hosp         | F2 (Interpreted)       | P3 (Relational-Focus)         | text-embedding-004      | 0.7905 (0.7721 - 0.8086) | 0.3150 (0.2805 - 0.3551) |
| mort_hosp         | F2 (Interpreted)       | P4 (Acute Dysregulation)      | text-embedding-004      | 0.7814 (0.7635 - 0.8001) | 0.3058 (0.2702 - 0.3481) |
| mort_hosp         | F2 (Interpreted)       | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.7928 (0.7734 - 0.8102) | 0.3108 (0.2749 - 0.3511) |
| mort_hosp         | F3 (Narrative Summary) | P0 (Control)                  | text-embedding-004      | 0.8379 (0.8225 - 0.8538) | 0.3857 (0.3478 - 0.4245) |
| mort_hosp         | F3 (Narrative Summary) | P1 (Task-Specific)            | text-embedding-004      | 0.8350 (0.8184 - 0.8508) | 0.3828 (0.3463 - 0.4234) |
| mort_hosp         | F3 (Narrative Summary) | P2 (Persona-Driven)           | text-embedding-004      | 0.8220 (0.8049 - 0.8385) | 0.3602 (0.3230 - 0.4000) |
| mort_hosp         | F3 (Narrative Summary) | P3 (Relational-Focus)         | text-embedding-004      | 0.8372 (0.8216 - 0.8533) | 0.4034 (0.3650 - 0.4443) |
| mort_hosp         | F3 (Narrative Summary) | P4 (Acute Dysregulation)      | text-embedding-004      | 0.8332 (0.8160 - 0.8489) | 0.3820 (0.3448 - 0.4241) |
| mort_hosp         | F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.8389 (0.8236 - 0.8554) | 0.3872 (0.3510 - 0.4258) |
| los_3             | Baseline (Numeric)     | XGBoost                       | XGBoost                 | 0.7265 (0.7140 - 0.7397) | 0.6627 (0.6461 - 0.6808) |
| los_3             | Baseline (Numeric)     | Elastic Net                   | ElasticNet              | 0.7187 (0.7056 - 0.7327) | 0.6561 (0.6390 - 0.6738) |
| los_3             | F1 (Uninterpreted)     | P0 (Control)                  | text-embedding-004      | 0.6557 (0.6412 - 0.6696) | 0.5778 (0.5575 - 0.6004) |
| los_3             | F1 (Uninterpreted)     | P1 (Task-Specific)            | text-embedding-004      | 0.6590 (0.6442 - 0.6727) | 0.5783 (0.5585 - 0.6020) |
| los_3             | F1 (Uninterpreted)     | P2 (Persona-Driven)           | text-embedding-004      | 0.6514 (0.6363 - 0.6649) | 0.5680 (0.5487 - 0.5906) |
| los_3             | F1 (Uninterpreted)     | P3 (Relational-Focus)         | text-embedding-004      | 0.6511 (0.6369 - 0.6651) | 0.5723 (0.5527 - 0.5956) |
| los_3             | F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6567 (0.6424 - 0.6708) | 0.5803 (0.5602 - 0.6040) |
| los_3             | F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6538 (0.6391 - 0.6684) | 0.5716 (0.5523 - 0.5958) |
| los_3             | F2 (Interpreted)       | P0 (Control)                  | text-embedding-004      | 0.6582 (0.6434 - 0.6720) | 0.5792 (0.5587 - 0.6031) |
| los_3             | F2 (Interpreted)       | P1 (Task-Specific)            | text-embedding-004      | 0.6597 (0.6456 - 0.6739) | 0.5792 (0.5606 - 0.6036) |
| los_3             | F2 (Interpreted)       | P2 (Persona-Driven)           | text-embedding-004      | 0.6568 (0.6422 - 0.6709) | 0.5781 (0.5587 - 0.6021) |
| los_3             | F2 (Interpreted)       | P3 (Relational-Focus)         | text-embedding-004      | 0.6513 (0.6367 - 0.6647) | 0.5740 (0.5543 - 0.5972) |
| los_3             | F2 (Interpreted)       | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6558 (0.6413 - 0.6694) | 0.5741 (0.5546 - 0.5984) |
| los_3             | F2 (Interpreted)       | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6548 (0.6401 - 0.6682) | 0.5758 (0.5561 - 0.6003) |
| los_3             | F3 (Narrative Summary) | P0 (Control)                  | text-embedding-004      | 0.6629 (0.6494 - 0.6769) | 0.5810 (0.5623 - 0.6029) |
| los_3             | F3 (Narrative Summary) | P1 (Task-Specific)            | text-embedding-004      | 0.6701 (0.6572 - 0.6840) | 0.5959 (0.5753 - 0.6176) |
| los_3             | F3 (Narrative Summary) | P2 (Persona-Driven)           | text-embedding-004      | 0.6666 (0.6535 - 0.6812) | 0.5963 (0.5762 - 0.6171) |
| los_3             | F3 (Narrative Summary) | P3 (Relational-Focus)         | text-embedding-004      | 0.6634 (0.6504 - 0.6774) | 0.5908 (0.5713 - 0.6135) |
| los_3             | F3 (Narrative Summary) | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6684 (0.6555 - 0.6820) | 0.5944 (0.5741 - 0.6163) |
| los_3             | F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6652 (0.6522 - 0.6794) | 0.5855 (0.5653 - 0.6079) |
| los_7             | Baseline (Numeric)     | XGBoost                       | XGBoost                 | 0.7413 (0.7178 - 0.7665) | 0.1829 (0.1614 - 0.2127) |
| los_7             | Baseline (Numeric)     | Elastic Net                   | ElasticNet              | 0.7438 (0.7203 - 0.7682) | 0.1859 (0.1650 - 0.2175) |
| los_7             | F1 (Uninterpreted)     | P0 (Control)                  | text-embedding-004      | 0.6432 (0.6142 - 0.6696) | 0.1190 (0.1037 - 0.1407) |
| los_7             | F1 (Uninterpreted)     | P1 (Task-Specific)            | text-embedding-004      | 0.6695 (0.6425 - 0.6965) | 0.1271 (0.1105 - 0.1487) |
| los_7             | F1 (Uninterpreted)     | P2 (Persona-Driven)           | text-embedding-004      | 0.6667 (0.6385 - 0.6930) | 0.1304 (0.1130 - 0.1550) |
| los_7             | F1 (Uninterpreted)     | P3 (Relational-Focus)         | text-embedding-004      | 0.6782 (0.6503 - 0.7032) | 0.1324 (0.1161 - 0.1575) |
| los_7             | F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6846 (0.6592 - 0.7098) | 0.1256 (0.1110 - 0.1497) |
| los_7             | F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6829 (0.6567 - 0.7080) | 0.1334 (0.1171 - 0.1593) |
| los_7             | F2 (Interpreted)       | P0 (Control)                  | text-embedding-004      | 0.6777 (0.6493 - 0.7051) | 0.1325 (0.1164 - 0.1571) |
| los_7             | F2 (Interpreted)       | P2 (Persona-Driven)           | text-embedding-004      | 0.6793 (0.6531 - 0.7060) | 0.1315 (0.1153 - 0.1546) |
| los_7             | F2 (Interpreted)       | P3 (Relational-Focus)         | text-embedding-004      | 0.6758 (0.6498 - 0.7018) | 0.1366 (0.1179 - 0.1641) |
| los_7             | F2 (Interpreted)       | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6825 (0.6558 - 0.7092) | 0.1351 (0.1171 - 0.1606) |
| los_7             | F2 (Interpreted)       | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6794 (0.6520 - 0.7055) | 0.1304 (0.1145 - 0.1553) |
| los_7             | F3 (Narrative Summary) | P0 (Control)                  | text-embedding-004      | 0.6882 (0.6613 - 0.7159) | 0.1472 (0.1248 - 0.1766) |
| los_7             | F3 (Narrative Summary) | P1 (Task-Specific)            | text-embedding-004      | 0.6959 (0.6702 - 0.7234) | 0.1460 (0.1265 - 0.1763) |
| los_7             | F3 (Narrative Summary) | P2 (Persona-Driven)           | text-embedding-004      | 0.6860 (0.6576 - 0.7138) | 0.1435 (0.1238 - 0.1735) |
| los_7             | F3 (Narrative Summary) | P3 (Relational-Focus)         | text-embedding-004      | 0.6934 (0.6665 - 0.7210) | 0.1434 (0.1241 - 0.1695) |
| los_7             | F3 (Narrative Summary) | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6967 (0.6702 - 0.7239) | 0.1441 (0.1248 - 0.1720) |
| los_7             | F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6953 (0.6686 - 0.7228) | 0.1456 (0.1247 - 0.1732) |
| readmission_30    | Baseline (Numeric)     | XGBoost                       | XGBoost                 | 0.5973 (0.5636 - 0.6305) | 0.0657 (0.0579 - 0.0779) |
| readmission_30    | Baseline (Numeric)     | Elastic Net                   | ElasticNet              | 0.5759 (0.5419 - 0.6097) | 0.0565 (0.0511 - 0.0648) |
| readmission_30    | F1 (Uninterpreted)     | P0 (Control)                  | text-embedding-004      | 0.5674 (0.5315 - 0.6013) | 0.0570 (0.0478 - 0.0690) |
| readmission_30    | F1 (Uninterpreted)     | P1 (Task-Specific)            | text-embedding-004      | 0.5540 (0.5200 - 0.5860) | 0.0569 (0.0481 - 0.0708) |
| readmission_30    | F1 (Uninterpreted)     | P2 (Persona-Driven)           | text-embedding-004      | 0.5656 (0.5316 - 0.5994) | 0.0579 (0.0491 - 0.0738) |
| readmission_30    | F1 (Uninterpreted)     | P3 (Relational-Focus)         | text-embedding-004      | 0.5654 (0.5327 - 0.5974) | 0.0601 (0.0487 - 0.0752) |
| readmission_30    | F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | text-embedding-004      | 0.5625 (0.5297 - 0.5945) | 0.0590 (0.0485 - 0.0741) |
| readmission_30    | F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.5601 (0.5234 - 0.5935) | 0.0632 (0.0511 - 0.0840) |
| readmission_30    | F2 (Interpreted)       | P0 (Control)                  | text-embedding-004      | 0.5572 (0.5208 - 0.5911) | 0.0563 (0.0480 - 0.0684) |
| readmission_30    | F2 (Interpreted)       | P1 (Task-Specific)            | text-embedding-004      | 0.5178 (0.4824 - 0.5517) | 0.0498 (0.0427 - 0.0608) |
| readmission_30    | F2 (Interpreted)       | P2 (Persona-Driven)           | text-embedding-004      | 0.5406 (0.5045 - 0.5766) | 0.0550 (0.0467 - 0.0679) |
| readmission_30    | F2 (Interpreted)       | P3 (Relational-Focus)         | text-embedding-004      | 0.5785 (0.5468 - 0.6124) | 0.0597 (0.0508 - 0.0727) |
| readmission_30    | F2 (Interpreted)       | P4 (Acute Dysregulation)      | text-embedding-004      | 0.5736 (0.5373 - 0.6045) | 0.0586 (0.0501 - 0.0705) |
| readmission_30    | F2 (Interpreted)       | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.5287 (0.4941 - 0.5633) | 0.0496 (0.0429 - 0.0594) |
| readmission_30    | F3 (Narrative Summary) | P0 (Control)                  | text-embedding-004      | 0.5626 (0.5285 - 0.5990) | 0.0553 (0.0477 - 0.0667) |
| readmission_30    | F3 (Narrative Summary) | P1 (Task-Specific)            | text-embedding-004      | 0.5813 (0.5472 - 0.6157) | 0.0645 (0.0542 - 0.0792) |
| readmission_30    | F3 (Narrative Summary) | P2 (Persona-Driven)           | text-embedding-004      | 0.5774 (0.5430 - 0.6122) | 0.0631 (0.0533 - 0.0781) |
| readmission_30    | F3 (Narrative Summary) | P3 (Relational-Focus)         | text-embedding-004      | 0.5672 (0.5330 - 0.6007) | 0.0584 (0.0503 - 0.0715) |
| readmission_30    | F3 (Narrative Summary) | P4 (Acute Dysregulation)      | text-embedding-004      | 0.5599 (0.5269 - 0.5940) | 0.0558 (0.0484 - 0.0690) |
| readmission_30    | F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.5800 (0.5453 - 0.6147) | 0.0632 (0.0539 - 0.0766) |
| intervention_vent | Baseline (Numeric)     | XGBoost                       | XGBoost                 | 0.6939 (0.6602 - 0.7271) | 0.1995 (0.1624 - 0.2486) |
| intervention_vent | Baseline (Numeric)     | Elastic Net                   | ElasticNet              | 0.6714 (0.6357 - 0.7063) | 0.1507 (0.1263 - 0.1846) |
| intervention_vent | F1 (Uninterpreted)     | P0 (Control)                  | text-embedding-004      | 0.5579 (0.5194 - 0.5987) | 0.0917 (0.0764 - 0.1132) |
| intervention_vent | F1 (Uninterpreted)     | P1 (Task-Specific)            | text-embedding-004      | 0.5830 (0.5488 - 0.6197) | 0.0878 (0.0743 - 0.1047) |
| intervention_vent | F1 (Uninterpreted)     | P2 (Persona-Driven)           | text-embedding-004      | 0.5960 (0.5629 - 0.6321) | 0.0948 (0.0810 - 0.1174) |
| intervention_vent | F1 (Uninterpreted)     | P3 (Relational-Focus)         | text-embedding-004      | 0.5758 (0.5403 - 0.6145) | 0.0951 (0.0797 - 0.1192) |
| intervention_vent | F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6036 (0.5708 - 0.6414) | 0.0961 (0.0815 - 0.1174) |
| intervention_vent | F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.5996 (0.5648 - 0.6355) | 0.0977 (0.0812 - 0.1208) |
| intervention_vent | F2 (Interpreted)       | P0 (Control)                  | text-embedding-004      | 0.5817 (0.5491 - 0.6209) | 0.0949 (0.0804 - 0.1171) |
| intervention_vent | F2 (Interpreted)       | P1 (Task-Specific)            | text-embedding-004      | 0.5305 (0.4916 - 0.5697) | 0.0773 (0.0659 - 0.0923) |
| intervention_vent | F2 (Interpreted)       | P2 (Persona-Driven)           | text-embedding-004      | 0.5812 (0.5466 - 0.6161) | 0.0923 (0.0774 - 0.1121) |
| intervention_vent | F2 (Interpreted)       | P3 (Relational-Focus)         | text-embedding-004      | 0.5974 (0.5607 - 0.6345) | 0.0996 (0.0838 - 0.1234) |
| intervention_vent | F2 (Interpreted)       | P4 (Acute Dysregulation)      | text-embedding-004      | 0.5257 (0.4902 - 0.5621) | 0.0853 (0.0701 - 0.1087) |
| intervention_vent | F2 (Interpreted)       | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.5809 (0.5429 - 0.6150) | 0.0943 (0.0798 - 0.1153) |
| intervention_vent | F3 (Narrative Summary) | P0 (Control)                  | text-embedding-004      | 0.6280 (0.5915 - 0.6606) | 0.1062 (0.0905 - 0.1322) |
| intervention_vent | F3 (Narrative Summary) | P1 (Task-Specific)            | text-embedding-004      | 0.5900 (0.5547 - 0.6240) | 0.0958 (0.0807 - 0.1165) |
| intervention_vent | F3 (Narrative Summary) | P2 (Persona-Driven)           | text-embedding-004      | 0.5852 (0.5483 - 0.6214) | 0.1006 (0.0838 - 0.1277) |
| intervention_vent | F3 (Narrative Summary) | P3 (Relational-Focus)         | text-embedding-004      | 0.6162 (0.5805 - 0.6515) | 0.1069 (0.0906 - 0.1306) |
| intervention_vent | F3 (Narrative Summary) | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6109 (0.5726 - 0.6448) | 0.1038 (0.0871 - 0.1278) |
| intervention_vent | F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6133 (0.5760 - 0.6488) | 0.1033 (0.0878 - 0.1264) |
| intervention_vaso | Baseline (Numeric)     | XGBoost                       | XGBoost                 | 0.7724 (0.7405 - 0.8025) | 0.2166 (0.1748 - 0.2748) |
| intervention_vaso | Baseline (Numeric)     | Elastic Net                   | ElasticNet              | 0.7719 (0.7394 - 0.8035) | 0.2331 (0.1878 - 0.2902) |
| intervention_vaso | F1 (Uninterpreted)     | P0 (Control)                  | text-embedding-004      | 0.6754 (0.6386 - 0.7143) | 0.1131 (0.0912 - 0.1504) |
| intervention_vaso | F1 (Uninterpreted)     | P1 (Task-Specific)            | text-embedding-004      | 0.6717 (0.6355 - 0.7091) | 0.1025 (0.0836 - 0.1330) |
| intervention_vaso | F1 (Uninterpreted)     | P2 (Persona-Driven)           | text-embedding-004      | 0.6740 (0.6342 - 0.7116) | 0.1013 (0.0839 - 0.1270) |
| intervention_vaso | F1 (Uninterpreted)     | P3 (Relational-Focus)         | text-embedding-004      | 0.6532 (0.6136 - 0.6940) | 0.1002 (0.0799 - 0.1284) |
| intervention_vaso | F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6298 (0.5955 - 0.6660) | 0.0859 (0.0702 - 0.1098) |
| intervention_vaso | F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6186 (0.5806 - 0.6574) | 0.0828 (0.0688 - 0.1053) |
| intervention_vaso | F2 (Interpreted)       | P0 (Control)                  | text-embedding-004      | 0.6705 (0.6310 - 0.7089) | 0.1121 (0.0902 - 0.1471) |
| intervention_vaso | F2 (Interpreted)       | P1 (Task-Specific)            | text-embedding-004      | 0.6547 (0.6132 - 0.6930) | 0.1115 (0.0873 - 0.1508) |
| intervention_vaso | F2 (Interpreted)       | P2 (Persona-Driven)           | text-embedding-004      | 0.6753 (0.6392 - 0.7131) | 0.1197 (0.0944 - 0.1627) |
| intervention_vaso | F2 (Interpreted)       | P3 (Relational-Focus)         | text-embedding-004      | 0.6499 (0.6116 - 0.6896) | 0.1023 (0.0829 - 0.1384) |
| intervention_vaso | F2 (Interpreted)       | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6736 (0.6378 - 0.7116) | 0.1095 (0.0876 - 0.1417) |
| intervention_vaso | F2 (Interpreted)       | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6688 (0.6319 - 0.7086) | 0.1205 (0.0958 - 0.1607) |
| intervention_vaso | F3 (Narrative Summary) | P0 (Control)                  | text-embedding-004      | 0.6764 (0.6399 - 0.7120) | 0.1255 (0.1000 - 0.1659) |
| intervention_vaso | F3 (Narrative Summary) | P1 (Task-Specific)            | text-embedding-004      | 0.6677 (0.6305 - 0.7049) | 0.1075 (0.0877 - 0.1367) |
| intervention_vaso | F3 (Narrative Summary) | P2 (Persona-Driven)           | text-embedding-004      | 0.6925 (0.6560 - 0.7292) | 0.1258 (0.1018 - 0.1611) |
| intervention_vaso | F3 (Narrative Summary) | P3 (Relational-Focus)         | text-embedding-004      | 0.6412 (0.6063 - 0.6770) | 0.0906 (0.0752 - 0.1174) |
| intervention_vaso | F3 (Narrative Summary) | P4 (Acute Dysregulation)      | text-embedding-004      | 0.6653 (0.6292 - 0.7031) | 0.1115 (0.0905 - 0.1443) |
| intervention_vaso | F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | text-embedding-004      | 0.6738 (0.6377 - 0.7090) | 0.1135 (0.0926 - 0.1492) |
| nan               | F1 (Uninterpreted)     | P0 (Control)                  | embedding_model_results | 0.7788 (0.7612 - 0.7956) | 0.2755 (0.2462 - 0.3136) |
| nan               | F1 (Uninterpreted)     | P0 (Control)                  | embedding_model_results | 0.7035 (0.6820 - 0.7247) | 0.2156 (0.1902 - 0.2451) |
| nan               | F1 (Uninterpreted)     | P1 (Task-Specific)            | embedding_model_results | 0.7798 (0.7608 - 0.7979) | 0.2804 (0.2485 - 0.3157) |
| nan               | F1 (Uninterpreted)     | P1 (Task-Specific)            | embedding_model_results | 0.7372 (0.7156 - 0.7581) | 0.2507 (0.2198 - 0.2827) |
| nan               | F1 (Uninterpreted)     | P2 (Persona-Driven)           | embedding_model_results | 0.7753 (0.7571 - 0.7920) | 0.2814 (0.2507 - 0.3170) |
| nan               | F1 (Uninterpreted)     | P2 (Persona-Driven)           | embedding_model_results | 0.7357 (0.7143 - 0.7566) | 0.2505 (0.2171 - 0.2851) |
| nan               | F1 (Uninterpreted)     | P3 (Relational-Focus)         | embedding_model_results | 0.7727 (0.7548 - 0.7902) | 0.2893 (0.2583 - 0.3235) |
| nan               | F1 (Uninterpreted)     | P3 (Relational-Focus)         | embedding_model_results | 0.7333 (0.7114 - 0.7541) | 0.2505 (0.2176 - 0.2873) |
| nan               | F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | embedding_model_results | 0.7707 (0.7523 - 0.7884) | 0.2731 (0.2430 - 0.3102) |
| nan               | F1 (Uninterpreted)     | P4 (Acute Dysregulation)      | embedding_model_results | 0.7399 (0.7198 - 0.7590) | 0.2558 (0.2243 - 0.2897) |
| nan               | F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | embedding_model_results | 0.7651 (0.7461 - 0.7832) | 0.2875 (0.2565 - 0.3244) |
| nan               | F1 (Uninterpreted)     | P5 (Dominant Pathophysiology) | embedding_model_results | 0.7302 (0.7084 - 0.7519) | 0.2558 (0.2220 - 0.2928) |
| nan               | F2 (Interpreted)       | P0 (Control)                  | embedding_model_results | 0.7974 (0.7790 - 0.8161) | 0.3216 (0.2871 - 0.3596) |
| nan               | F2 (Interpreted)       | P0 (Control)                  | embedding_model_results | 0.7466 (0.7254 - 0.7672) | 0.2465 (0.2184 - 0.2808) |
| nan               | F2 (Interpreted)       | P1 (Task-Specific)            | embedding_model_results | 0.7989 (0.7809 - 0.8161) | 0.3217 (0.2882 - 0.3623) |
| nan               | F2 (Interpreted)       | P1 (Task-Specific)            | embedding_model_results | 0.7530 (0.7331 - 0.7728) | 0.2466 (0.2216 - 0.2800) |
| nan               | F2 (Interpreted)       | P2 (Persona-Driven)           | embedding_model_results | 0.7921 (0.7730 - 0.8102) | 0.3065 (0.2730 - 0.3443) |
| nan               | F2 (Interpreted)       | P2 (Persona-Driven)           | embedding_model_results | 0.7436 (0.7219 - 0.7636) | 0.2578 (0.2274 - 0.2933) |
| nan               | F2 (Interpreted)       | P3 (Relational-Focus)         | embedding_model_results | 0.7974 (0.7794 - 0.8146) | 0.3229 (0.2878 - 0.3635) |
| nan               | F2 (Interpreted)       | P3 (Relational-Focus)         | embedding_model_results | 0.7548 (0.7341 - 0.7746) | 0.2653 (0.2338 - 0.2995) |
| nan               | F2 (Interpreted)       | P4 (Acute Dysregulation)      | embedding_model_results | 0.7894 (0.7711 - 0.8076) | 0.3164 (0.2818 - 0.3583) |
| nan               | F2 (Interpreted)       | P4 (Acute Dysregulation)      | embedding_model_results | 0.7333 (0.7109 - 0.7551) | 0.2626 (0.2302 - 0.3004) |
| nan               | F2 (Interpreted)       | P5 (Dominant Pathophysiology) | embedding_model_results | 0.7906 (0.7724 - 0.8084) | 0.3087 (0.2768 - 0.3467) |
| nan               | F2 (Interpreted)       | P5 (Dominant Pathophysiology) | embedding_model_results | 0.7424 (0.7210 - 0.7638) | 0.2574 (0.2259 - 0.2925) |
| nan               | F3 (Narrative Summary) | P0 (Control)                  | embedding_model_results | 0.8131 (0.7970 - 0.8288) | 0.3223 (0.2903 - 0.3615) |
| nan               | F3 (Narrative Summary) | P0 (Control)                  | embedding_model_results | 0.8129 (0.7958 - 0.8302) | 0.3431 (0.3075 - 0.3838) |
| nan               | F3 (Narrative Summary) | P1 (Task-Specific)            | embedding_model_results | 0.8254 (0.8092 - 0.8415) | 0.3671 (0.3303 - 0.4081) |
| nan               | F3 (Narrative Summary) | P1 (Task-Specific)            | embedding_model_results | 0.8140 (0.7987 - 0.8320) | 0.3344 (0.3019 - 0.3754) |
| nan               | F3 (Narrative Summary) | P2 (Persona-Driven)           | embedding_model_results | 0.8161 (0.7996 - 0.8330) | 0.3378 (0.3022 - 0.3776) |
| nan               | F3 (Narrative Summary) | P2 (Persona-Driven)           | embedding_model_results | 0.8007 (0.7836 - 0.8182) | 0.3198 (0.2869 - 0.3602) |
| nan               | F3 (Narrative Summary) | P3 (Relational-Focus)         | embedding_model_results | 0.8288 (0.8135 - 0.8449) | 0.3503 (0.3141 - 0.3902) |
| nan               | F3 (Narrative Summary) | P3 (Relational-Focus)         | embedding_model_results | 0.8139 (0.7962 - 0.8315) | 0.3409 (0.3066 - 0.3817) |
| nan               | F3 (Narrative Summary) | P4 (Acute Dysregulation)      | embedding_model_results | 0.8223 (0.8060 - 0.8381) | 0.3394 (0.3038 - 0.3797) |
| nan               | F3 (Narrative Summary) | P4 (Acute Dysregulation)      | embedding_model_results | 0.8139 (0.7968 - 0.8314) | 0.3377 (0.3013 - 0.3768) |
| nan               | F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | embedding_model_results | 0.8264 (0.8102 - 0.8423) | 0.3446 (0.3088 - 0.3856) |
| nan               | F3 (Narrative Summary) | P5 (Dominant Pathophysiology) | embedding_model_results | 0.8141 (0.7969 - 0.8312) | 0.3202 (0.2891 - 0.3584) |