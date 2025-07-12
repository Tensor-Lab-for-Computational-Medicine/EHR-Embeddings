---
marp: true
---

# Figure: End-to-End Experimental Workflow

```mermaid
graph TD
    subgraph "Phase 1: Data Preparation & Baseline Modeling"
        direction LR
        A[Raw Structured<br>EHR Data] --> B{Data Preprocessing};
        B --> C[Processed<br>Structured Data];
        subgraph "Baseline Models"
            direction TB
            D[Training]
        end
        C --> D;
        D --> E[XGBoost];
        D --> F[Elastic Net];
    end

    subgraph "Phase 2: Textual Data Transformation"
        direction LR
        C --> G{Textual Representation<br>Generation};
        subgraph "Textual Formats"
            direction TB
            G --> H[F1: Uninterpreted];
            G --> I[F2: Interpreted];
            G --> J[F3: Narrative Summary];
        end
    end

    subgraph "Phase 3: Prompting Strategies"
        direction LR
        K_input(Textual Formats) --> K{Prompt Application};
        subgraph "Prompts"
            direction TB
            K --> L[P0: Control];
            K --> M[P1: Task-Specific];
            K --> N[P2: Persona-Driven];
            K --> O[P3: Relational-Focus];
            K --> P[P4: Acute Dysregulation];
            K --> Q[P5: Dominant Pathophysiology];
        end
    end

    subgraph "Phase 4: Embedding Generation & Modeling"
        direction TB
        R_input(Prompted Text) --> S{Embedding Models};
        subgraph "Embedding Models"
            direction LR
            S --> T[embedding-001];
            S --> U[text-embedding-004];
            S --> V[... etc.];
        end
        W[Generated Embeddings] --> X{Classification Layer};
        S --> W;
    end

    subgraph "Phase 5: Evaluation & Comparison"
        direction LR
        Y_input(Model Predictions) --> Z{Performance Evaluation};
        Z -- "AUROC & AUPRC" --> AA[Final Results Comparison];
    end

    H --> K_input;
    I --> K_input;
    J --> K_input;

    L --> R_input;
    M --> R_input;
    N --> R_input;
    O --> R_input;
    P --> R_input;
    Q --> R_input;

    E --> Y_input;
    F --> Y_input;
    X --> Y_input;


    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style C fill:#f9f9f9,stroke:#333,stroke-width:2px
    style D fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
    style E fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
    style F fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
    style G fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    style H fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    style I fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    style J fill:#d5e8d4,stroke:#82b366,stroke-width:2px
    style K fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    style L fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    style M fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    style N fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    style O fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    style P fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    style Q fill:#ffe6cc,stroke:#d79b00,stroke-width:2px
    style S fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style T fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style U fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style V fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style W fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style X fill:#f8cecc,stroke:#b85450,stroke-width:2px
    style Z fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style AA fill:#e1d5e7,stroke:#9673a6,stroke-width:2px
    style K_input fill:#fff,stroke:#fff
    style R_input fill:#fff,stroke:#fff
    style Y_input fill:#fff,stroke:#fff
``` 
