---
marp: true
---

# Figure: End-to-End Experimental Workflow

```mermaid
graph TD
    subgraph Phase 1: Data Preparation & Baseline Modeling
        A[Raw Structured EHR Data] --> B{Data Preprocessing};
        B --> C[Processed Structured Data];
        C --> D[Baseline Model Training];
        D --> E{XGBoost};
        D --> F{Elastic Net};
    end

    subgraph Phase 2: Textual Data Transformation
        C --> G{Textual Representation Generation};
        G --> H[F1: Uninterpreted];
        G --> I[F2: Interpreted];
        G --> J[F3: Narrative Summary];
    end

    subgraph Phase 3: Prompting Strategies
        H --> K{Prompt Application};
        I --> K;
        J --> K;
        K --> L[P0: Control];
        K --> M[P1: Task-Specific];
        K --> N[P2: Persona-Driven];
        K --> O[P3: Relational-Focus];
        K --> P[P4: Acute Dysregulation];
        K --> Q[P5: Dominant Pathophysiology];
    end

    subgraph Phase 4: Embedding Generation & Modeling
        R[Prompted Text Data]
        subgraph For Each Representation F1, F2, F3 and Each Prompt P0-P5
            direction LR
            R --> S{Embedding Models};
        end
        S --> T[embedding-001];
        S --> U[text-embedding-004];
        S --> V[... etc.];
        W[Generated Embeddings] --> X{Classification Layer};
    end
    
    subgraph Phase 5: Evaluation & Comparison
        Y[Model Predictions];
        E --> Y;
        F --> Y;
        X --> Y;
        Y --> Z{Performance Evaluation};
        Z -- AUROC & AUPRC --> AA[Final Results Comparison];
    end

    L --> R; M --> R; N --> R; O --> R; P --> R; Q --> R;
``` 