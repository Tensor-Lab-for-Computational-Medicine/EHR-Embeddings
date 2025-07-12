---
marp: true
---

# Figure: End-to-End Experimental Workflow

```mermaid
graph LR
    %% Phase 1: Data Preparation
    subgraph "Phase 1: Data Preparation"
        direction TB
        A[Raw Structured Data] --> B{Preprocessing};
        B --> C[Processed Data];
    end

    %% Phase 2: Parallel Modeling Pipelines
    subgraph "Phase 2: Modeling"
        direction TB
        
        subgraph "A: Baseline Pipeline"
            C --> D{Baseline Model Training};
            D --> E[Baseline Models];
        end

        subgraph "B: LLM Pipeline"
            C --> F{Textual Representation};
            F --> G{Prompt Application};
            G --> H{Embedding Generation};
            H --> I[Embeddings];
        end
    end
    
    %% Phase 3: Evaluation
    subgraph "Phase 3: Evaluation & Results"
        direction TB
        I --> J{Classification Layer};
        J --> K[LLM-based Predictions];
        E --> L[Baseline Predictions];
        
        subgraph "Final Comparison"
            K & L --> M{Performance Evaluation};
            M -- AUROC & AUPRC --> N[Final Results];
        end
    end

    %% Styling
    classDef data fill:#fef0de,stroke:#c7882a,stroke-width:2px,color:#333;
    classDef baseline fill:#e3eefc,stroke:#3671c6,stroke-width:2px,color:#333;
    classDef llm fill:#e4f2e3,stroke:#559a53,stroke-width:2px,color:#333;
    classDef eval fill:#f1e3f2,stroke:#8d559a,stroke-width:2px,color:#333;

    class A,B,C data;
    class D,E,L baseline;
    class F,G,H,I,J,K llm;
    class M,N eval;
