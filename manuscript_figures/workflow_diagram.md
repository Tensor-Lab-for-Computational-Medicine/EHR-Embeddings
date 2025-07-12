---
marp: true
---

# Figure: End-to-End Experimental Workflow

```mermaid
graph LR
    %% Phase 1: Data Preparation & Feature Engineering
    subgraph "Phase 1: Data Prep & Feature Engineering"
        direction TB
        A["MIMIC-III ICU Data<br>(34,472 Stays)"] --> B{"Filtering & Cohort Selection<br>(24h data window)"};
        B --> C["Analysis Cohort<br>(23,944 Stays)"];
        C --> D{"Feature Engineering<br>(478 Structured Features)"};
    end

    %% Phase 2: Modeling Pipelines
    subgraph "Phase 2: Predictive Modeling"
        direction TB
        subgraph "A: XGBoost on Structured Data"
            D --> E{"Train XGBoost Classifier"};
        end

        subgraph "B: Classifier on Text Data"
            D --> F{Data-to-Text Serialization};
            F --> G{Embedding Generation};
            G --> H{"Train Classifier on Embeddings"};
        end
    end

    %% Phase 3: Evaluation & Interpretation
    subgraph "Phase 3: Evaluation & Interpretation"
        direction TB
        subgraph "Performance Comparison"
            E --> I[XGBoost Predictions];
            H --> J[LLM-based Predictions];
            I & J --> K{"Model Evaluation<br>(AUROC & AUPRC)"};
        end
        subgraph "Results & Interpretation"
            K --> L["<b>Final Result</b><br>XGBoost AUROC: 0.908"];
            E --> M{"SHAP Analysis"};
            M --> N[Feature Importance];
        end
    end

    %% Styling
    classDef data fill:#fef0de,stroke:#c7882a,stroke-width:2px;
    classDef model fill:#e3eefc,stroke:#3671c6,stroke-width:2px;
    classDef llm fill:#e4f2e3,stroke:#559a53,stroke-width:2px;
    classDef eval fill:#f1e3f2,stroke:#8d559a,stroke-width:2px;

    class A,B,C,D data;
    class E,I,M,N model;
    class F,G,H,J llm;
    class K,L eval;
```
