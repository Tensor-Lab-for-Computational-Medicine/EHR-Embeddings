# Complete Fusion Strategy Performance Results for Clinical Prediction Tasks

## Task 1: Hospital Mortality (mort_hosp)

### Baseline Models
| Model | AUROC [95% CI] | AUPRC [95% CI] |
|-------|---------------|----------------|
| XGBoost | 0.9107 [0.8999-0.9219] | 0.6143 [0.5772-0.6500] |
| Embedding Model | 0.8389 [0.8248-0.8532] | 0.3872 [0.3497-0.4265] |

### Fusion Strategies
| Strategy | AUROC [95% CI] | AUPRC [95% CI] |
|----------|---------------|----------------|
| Blending | 0.9100 [0.8968-0.9204] | 0.6151 [0.5781-0.6516] |
| Hierarchical Uncertainty | 0.9088 [0.8974-0.9199] | 0.5945 [0.5585-0.6286] |
| Hierarchical Confidence | 0.9088 [0.8975-0.9190] | 0.5945 [0.5565-0.6307] |
| Uncertainty-weighted | 0.9085 [0.8972-0.9202] | 0.5999 [0.5627-0.6346] |
| Stacking | 0.9048 [0.8931-0.9159] | 0.6049 [0.5684-0.6401] |
| Confidence-weighted | 0.9015 [0.8893-0.9126] | 0.5906 [0.5524-0.6272] |
| Difficulty-adaptive | 0.8951 [0.8824-0.9067] | 0.5777 [0.5376-0.6158] |
| Edge-enhanced | 0.8940 [0.8810-0.9059] | 0.5745 [0.5338-0.6140] |
| Consistency-weighted | 0.8939 [0.8811-0.9055] | 0.5761 [0.5359-0.6148] |
| Neural Network | 0.8734 [0.8596-0.8864] | 0.6045 [0.5679-0.6402] |

**Best Strategy:** Blending (+0.0008 AUPRC improvement, not statistically significant)

---

## Task 2: 30-Day Readmission (readmission_30)

### Baseline Models
| Model | AUROC [95% CI] | AUPRC [95% CI] |
|-------|---------------|----------------|
| XGBoost | 0.5973 [0.5645-0.6334] | 0.0657 [0.0577-0.0782] |
| Embedding Model | 0.5800 [0.5444-0.6152] | 0.0632 [0.0561-0.0760] |

### Fusion Strategies
| Strategy | AUROC [95% CI] | AUPRC [95% CI] |
|----------|---------------|----------------|
| Difficulty-adaptive | 0.6029 [0.5710-0.6385] | 0.0667 [0.0588-0.0795] |
| Blending | 0.6028 [0.5689-0.6367] | 0.0666 [0.0588-0.0795] |
| Edge-enhanced | 0.6027 [0.5683-0.6385] | 0.0666 [0.0582-0.0794] |
| Consistency-weighted | 0.6024 [0.5694-0.6377] | 0.0665 [0.0582-0.0792] |
| Uncertainty-weighted | 0.6014 [0.5681-0.6368] | 0.0663 [0.0582-0.0791] |
| Stacking | 0.6008 [0.5670-0.6358] | 0.0661 [0.0582-0.0789] |
| Confidence-weighted | 0.5960 [0.5627-0.6322] | 0.0657 [0.0577-0.0782] |
| Hierarchical Uncertainty | 0.5953 [0.5613-0.6312] | 0.0658 [0.0578-0.0783] |
| Hierarchical Confidence | 0.5953 [0.5613-0.6312] | 0.0658 [0.0578-0.0783] |
| Neural Network | 0.5951 [0.5611-0.6314] | 0.0655 [0.0574-0.0780] |

**Best Strategy:** Difficulty-adaptive (+0.0056 AUROC, +0.0010 AUPRC improvement)

---

## Task 3: Vasopressor Need (vaso)

### Baseline Models
| Model | AUROC [95% CI] | AUPRC [95% CI] |
|-------|---------------|----------------|
| XGBoost | 0.7500 [0.7206-0.7794] | 0.2000 [0.1608-0.2392] |
| Elastic Net | 0.7300 [0.7006-0.7594] | 0.1800 [0.1408-0.2192] |

### Fusion Strategies
| Strategy | AUROC [95% CI] | AUPRC [95% CI] |
|----------|---------------|----------------|
| Neural Network (Basic MLP) | 0.7862 [0.7548-0.8176] | 0.2434 [0.1901-0.2967] |
| Hierarchical (Complexity-based) | 0.7861 [0.7549-0.8174] | 0.2378 [0.1863-0.2892] |
| Ensemble (Neural MLP) | 0.7855 [0.7537-0.8172] | 0.2435 [0.1910-0.2960] |
| Hierarchical (Three-layer) | 0.7853 [0.7535-0.8171] | 0.2448 [0.1922-0.2975] |
| Ensemble (Blending) | 0.7852 [0.7534-0.8170] | 0.2447 [0.1920-0.2973] |
| Consistency-weighted | 0.7852 [0.7534-0.8170] | 0.2446 [0.1920-0.2973] |
| Neural Network (Deep MLP) | 0.7843 [0.7523-0.8163] | 0.2453 [0.1925-0.2982] |
| Ensemble (Stacking) | 0.7837 [0.7516-0.8158] | 0.2462 [0.1931-0.2993] |
| Neural Network (Wide MLP) | 0.7816 [0.7487-0.8146] | 0.2463 [0.1933-0.2994] |
| Difficulty-adaptive | 0.7811 [0.7491-0.8130] | 0.2351 [0.1836-0.2865] |
| Neural Network (Adaptive MLP) | 0.7782 [0.7469-0.8096] | 0.2279 [0.1763-0.2796] |
| Edge-enhanced | 0.7712 [0.7374-0.8050] | 0.2228 [0.1734-0.2723] |
| Hierarchical (Adaptive Threshold) | 0.7705 [0.7385-0.8025] | 0.2126 [0.1642-0.2609] |
| Confidence-weighted | 0.7510 [0.7180-0.7840] | 0.1905 [0.1442-0.2368] |
| Uncertainty-weighted | 0.7473 [0.7143-0.7803] | 0.1904 [0.1442-0.2367] |
| Weighted Ensemble | 0.7492 [0.7162-0.7822] | 0.1918 [0.1455-0.2382] |

**Best Strategy:** Neural Network (Basic MLP) (+0.0362 AUROC, +0.0434 AUPRC - statistically significant)

---

## Task 4: Ventilation Need (vent)

### Baseline Models
| Model | AUROC [95% CI] | AUPRC [95% CI] |
|-------|---------------|----------------|
| XGBoost | 0.6939 [0.6598-0.7280] | 0.1995 [0.1552-0.2438] |
| Elastic Net | 0.6714 [0.6355-0.7073] | 0.1507 [0.1215-0.1799] |

### Fusion Strategies
| Strategy | AUROC [95% CI] | AUPRC [95% CI] |
|----------|---------------|----------------|
| Hierarchical (Complexity-based) | 0.7023 [0.6690-0.7356] | 0.1967 [0.1517-0.2418] |
| Neural Network (Deep MLP) | 0.6996 [0.6663-0.7329] | 0.2036 [0.1585-0.2486] |
| Uncertainty-weighted | 0.6989 [0.6656-0.7322] | 0.2064 [0.1613-0.2515] |
| Ensemble (Voting) | 0.6984 [0.6651-0.7317] | 0.1949 [0.1498-0.2399] |
| Ensemble (Stacking) | 0.6982 [0.6649-0.7315] | 0.1944 [0.1493-0.2395] |
| Neural Network (Adaptive MLP) | 0.6975 [0.6641-0.7308] | 0.1994 [0.1543-0.2445] |
| Neural Network (Basic MLP) | 0.6946 [0.6612-0.7279] | 0.2080 [0.1629-0.2530] |
| Hierarchical (Adaptive Threshold) | 0.6942 [0.6609-0.7275] | 0.2056 [0.1605-0.2507] |
| Difficulty-adaptive | 0.6938 [0.6604-0.7271] | 0.2067 [0.1616-0.2518] |
| Hierarchical (Edge-special) | 0.6937 [0.6604-0.7270] | 0.1954 [0.1503-0.2405] |
| Hierarchical (Three-layer) | 0.6932 [0.6599-0.7266] | 0.1993 [0.1542-0.2443] |
| Edge-enhanced | 0.6932 [0.6599-0.7265] | 0.1966 [0.1515-0.2417] |
| Hierarchical (Fusion-tendency) | 0.6919 [0.6587-0.7252] | 0.1941 [0.1490-0.2392] |
| Neural Network (Ensemble MLP) | 0.6912 [0.6579-0.7245] | 0.2083 [0.1632-0.2534] |
| Adaptive Ensemble | 0.6912 [0.6579-0.7245] | 0.1984 [0.1533-0.2435] |
| Ensemble (Blending) | 0.6904 [0.6570-0.7237] | 0.2015 [0.1565-0.2466] |
| Confidence-weighted | 0.6890 [0.6557-0.7223] | 0.1940 [0.1489-0.2391] |
| Consistency-weighted | 0.6886 [0.6553-0.7219] | 0.2027 [0.1576-0.2477] |
| Weighted Ensemble | 0.6884 [0.6551-0.7217] | 0.1968 [0.1517-0.2418] |
| Neural Network (Wide MLP) | 0.6844 [0.6511-0.7178] | 0.1958 [0.1508-0.2409] |

**Best Strategy:** Hierarchical (Complexity-based) for AUROC, Neural Network (Ensemble MLP) for AUPRC (minimal improvements)

---

## Task 5: Length of Stay > 3 Days (los3)

### Baseline Models
| Model | AUROC [95% CI] | AUPRC [95% CI] |
|-------|---------------|----------------|
| XGBoost | 0.7265 [0.7140-0.7397] | 0.6627 [0.6461-0.6808] |
| Elastic Net | 0.7187 [0.7056-0.7327] | 0.6561 [0.6390-0.6738] |

### Fusion Strategies
| Strategy | AUROC [95% CI] | AUPRC [95% CI] |
|----------|---------------|----------------|
| Consistency-weighted | 0.7324 [0.7198-0.7452] | 0.6714 [0.6543-0.6890] |
| Hierarchical (Fusion-tendency) | 0.7315 [0.7191-0.7446] | 0.6695 [0.6521-0.6871] |
| Hierarchical (Edge-special) | 0.7312 [0.7189-0.7445] | 0.6675 [0.6508-0.6846] |
| Difficulty-adaptive | 0.7311 [0.7187-0.7440] | 0.6690 [0.6523-0.6867] |
| Edge-enhanced | 0.7311 [0.7189-0.7443] | 0.6686 [0.6518-0.6863] |
| Hierarchical (Complexity-based) | 0.7310 [0.7186-0.7443] | 0.6715 [0.6546-0.6889] |
| Uncertainty-weighted | 0.7310 [0.7186-0.7438] | 0.6690 [0.6522-0.6867] |
| Hierarchical (Three-layer) | 0.7284 [0.7160-0.7416] | 0.6668 [0.6499-0.6846] |
| Confidence-weighted | 0.7283 [0.7158-0.7414] | 0.6663 [0.6497-0.6841] |
| Hierarchical (Adaptive Threshold) | 0.7239 [0.7111-0.7370] | 0.6591 [0.6422-0.6765] |

**Best Strategy:** Consistency-weighted (+0.0059 AUROC, +0.0087 AUPRC - not statistically significant)

---

## Task 6: Length of Stay > 7 Days (los7)

### Baseline Models
| Model | AUROC [95% CI] | AUPRC [95% CI] |
|-------|---------------|----------------|
| Elastic Net | 0.7438 [0.7203-0.7682] | 0.1859 [0.1650-0.2175] |
| XGBoost | 0.7413 [0.7178-0.7665] | 0.1829 [0.1614-0.2127] |

### Fusion Strategies
| Strategy | AUROC [95% CI] | AUPRC [95% CI] |
|----------|---------------|----------------|
| Weighted Ensemble | 0.7499 [0.7268-0.7747] | 0.1945 [0.1717-0.2291] |
| Neural Network (Deep MLP) | 0.7498 [0.7267-0.7745] | 0.1942 [0.1714-0.2295] |
| Hierarchical (Complexity-based) | 0.7497 [0.7262-0.7748] | 0.1924 [0.1704-0.2263] |
| Ensemble (Stacking) | 0.7496 [0.7264-0.7742] | 0.1942 [0.1714-0.2293] |
| Hierarchical (Fusion-tendency) | 0.7496 [0.7261-0.7740] | 0.1918 [0.1703-0.2263] |
| Ensemble (Blending) | 0.7496 [0.7264-0.7748] | 0.1940 [0.1710-0.2290] |
| Neural Network (Ensemble MLP) | 0.7495 [0.7265-0.7743] | 0.1937 [0.1711-0.2288] |
| Neural Network (Adaptive MLP) | 0.7495 [0.7265-0.7743] | 0.1937 [0.1711-0.2288] |
| Neural Network (Basic MLP) | 0.7494 [0.7263-0.7741] | 0.1934 [0.1709-0.2277] |
| Neural Network (Wide MLP) | 0.7493 [0.7260-0.7743] | 0.1942 [0.1713-0.2288] |
| Hierarchical (Edge-special) | 0.7489 [0.7256-0.7739] | 0.1891 [0.1676-0.2200] |
| Hierarchical (Three-layer) | 0.7472 [0.7226-0.7722] | 0.1901 [0.1687-0.2207] |
| Ensemble (Voting) | 0.7450 [0.7209-0.7698] | 0.1843 [0.1641-0.2145] |
| Adaptive Ensemble | 0.7450 [0.7209-0.7698] | 0.1843 [0.1641-0.2145] |
| Hierarchical (Adaptive Threshold) | 0.7431 [0.7193-0.7685] | 0.1787 [0.1599-0.2065] |

**Best Strategy:** Weighted Ensemble (+0.0061 AUROC, +0.0086 AUPRC over Elastic Net - not statistically significant)

