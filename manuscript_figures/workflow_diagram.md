---
marp: true
---

# Figure: End-to-End Experimental Workflow

```mermaid
graph LR
    %% Phase 1: Data Preparation
    subgraph "Phase 1: Data Preparation"
        direction TB
        A["MIMIC-III ICU Data"] --> B{"Cohort Selection<br>(23,944 Stays)"};
        B --> C{"Feature Engineering<br>(478 Features)"};
    end

    %% Phase 2: Modeling Pipelines
    subgraph "Phase 2: Modeling"
        direction TB
        
        subgraph "A: XGBoost Pipeline"
            C -- Structured Data --> D{"Train XGBoost Classifier"};
        end

        subgraph "B: LLM-based Pipeline"
            C -- Serialized Text --> E{Text-to-Embedding};
            E --> F{"Train Classifier on Embeddings"};
        end
    end

    %% Phase 3: Evaluation
    subgraph "Phase 3: Evaluation"
        direction TB
        D & F --> G{"Performance Comparison<br>(AUROC & AUPRC)"};
        G --> H["<b>Final Result</b><br>XGBoost Outperforms<br>AUROC: 0.908"];
    end

    %% Styling
    classDef data fill:#fef0de,stroke:#c7882a,stroke-width:2px;
    classDef model fill:#e3eefc,stroke:#3671c6,stroke-width:2px;
    classDef llm fill:#e4f2e3,stroke:#559a53,stroke-width:2px;
    classDef eval fill:#f1e3f2,stroke:#8d559a,stroke-width:2px;

    class A,B,C data;
    class D model;
    class E,F llm;
    class G,H eval;
```
