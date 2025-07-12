---
marp: true
---

# Figure: End-to-End Experimental Workflow

```mermaid
graph TD
    subgraph "Phase 1: Data & Baseline Models"
        direction LR
        A[Raw Structured EHR Data] --> B{Preprocessing};
        B --> C[Processed Data];
        C --> D{Baseline Model Training};
        D --> E[Baseline Models];
    end

    subgraph "Phase 2: Text & Embedding Generation"
        direction LR
        C --> G{Textual Representation};
        G --> K{Prompt Application};
        K --> S{Embedding Generation};
        S --> W[Embeddings];
    end

    subgraph "Phase 3: Modeling & Evaluation"
        direction LR
        W --> X{Classification Layer};
        X --> Y[LLM-based Predictions]
        E --> Z[Baseline Predictions]
        Y & Z --> AA{Performance Evaluation};
        AA -- "AUROC & AUPRC" --> BB[Final Results];
    end

    %% Styling
    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style C fill:#f9f9f9,stroke:#333,stroke-width:2px
    style D fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
    style E fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
    style G fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    style K fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    style S fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style W fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style X fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style Y fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style Z fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style AA fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style BB fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
``` 
