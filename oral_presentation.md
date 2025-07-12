# Oral Presentation: The Complementary Value of Semantic and Numerical Representations in Clinical Risk Prediction

---

## 1. Title of Project

The Complementary Value of Semantic and Numerical Representations in Clinical Risk Prediction

---

## 2. Broad Scope of Project (Introduction & Background)

*   **The Challenge of Clinical Risk Prediction:** Predicting patient outcomes, like in-hospital mortality, is a critical task in modern medicine. It helps clinicians make timely decisions and allocate resources effectively.
*   **Traditional Approaches:** For decades, we've relied on structured, numerical data from patient records – lab values, vital signs, etc. – to build predictive models. These models, like the one we've built using XGBoost, are powerful and well-established.
*   **The Rise of LLMs:** Recently, Large Language Models (LLMs) have shown incredible capabilities in understanding and processing human language. This opens up a new frontier in clinical informatics. Can we leverage the rich, unstructured text in clinical notes (physician's notes, discharge summaries) to improve our predictions?
*   **The Core Question:** Do these new "semantic" representations of clinical data offer information that is complementary to, or even better than, the traditional numerical data? That's the central question our research aims to answer.

---

## 3. Specific Aims

Our study has two primary aims:

1.  **To systematically compare** the performance of mortality prediction models built on semantic text embeddings against a strong baseline model that uses traditional, engineered numerical features.
2.  **To investigate the impact of data representation** on model performance. Specifically, we wanted to see how different ways of serializing clinical data (as raw values, interpreted values, or narrative summaries) and different prompting techniques affect the quality of the resulting text embeddings.

---

## 4. Methods

*   **Dataset:** We used the MIMIC-III database, a large, publicly available dataset of de-identified ICU patient data. Our cohort consisted of 22,591 ICU patients.
*   **Numerical Baseline Model:**
    *   We built a powerful XGBoost model using 458 engineered features from the first 24 hours of each patient's ICU stay. This represents a very strong and realistic clinical baseline.
*   **Semantic Models:**
    *   **Data Serialization:** We experimented with three formats to convert patient data into text:
        *   **F1: Uninterpreted Values:** Just the raw numbers and labels.
        *   **F2: Interpreted Values:** Adding clinical context, like "High" or "Low" to values.
        *   **F3: Narrative Summaries:** Generating a paragraph-style clinical summary.
    *   **Prompt Engineering:** For each serialization format, we used six different prompting strategies (P0-P5) to guide the embedding model. This included zero-shot prompts and persona-driven prompts (e.g., asking the model to act as a clinician).
    *   **Embedding Models:** We tested three of Google's text embedding models (`text-embedding-004`, `embedding-001`, and `text-embedding-005`) to generate the final semantic representations.
*   **Evaluation:** Our primary performance metrics were AUROC (Area Under the Receiver Operating Characteristic curve) and AUPRC (Area Under the Precision-Recall curve).

---

## 5. Preliminary Data and Results

*(Here, you could show `manuscript_figures/figure_1_auroc_heatmap.png`)*

*   **The Numerical Baseline is Still Superior:** Our XGBoost numerical model performed the best, achieving an AUROC of **0.908**.
*   **Narrative is Better for Text:** Among the semantic models, the narrative summaries (F3) consistently outperformed the more structured formats.
*   **Best Semantic Model:** The best-performing semantic model used a narrative summary with a persona-driven prompt (F3+P2) and the `text-embedding-004` model. It achieved an AUROC of **0.838**.
    *   This represents a significant performance gap of about 0.07 AUROC points compared to the numerical baseline.
*   **Prompting Matters (a little):** Prompt engineering had a modest but consistent effect. The persona-driven prompt (P2) was generally the most effective.
    *(Here, you could show `manuscript_figures/figure_2_interaction_plot.png`)*
*   **Not All Embeddings are Equal:** The choice of embedding model had a significant impact on performance. We found a clear ranking: `text-embedding-004` was the best, followed by `embedding-001`, and `text-embedding-005` was the worst.
    *(Here, you could show `manuscript_figures/figure_3_performance_lift.png`)*

---

## 6. Summary and Conclusions

*   **Summary:** In a head-to-head comparison for in-hospital mortality prediction, a traditional numerical model outperformed models based on semantic embeddings, even when using sophisticated prompt engineering and various data serialization techniques.
*   **Key Finding:** The way we format data for an LLM matters. Narrative representations (F3) appear to be a more effective way to capture clinically relevant information than structured lists of values.
*   **Challenges:** The primary challenge is the performance gap. While promising, semantic embeddings are not yet a direct replacement for carefully engineered numerical features in this high-stakes prediction task.
*   **Future Directions:**
    *   **Hybrid Models:** The most promising path forward is likely not an "either/or" approach, but a "both/and". Our findings suggest that semantic data may offer *complementary* information. The next logical step is to build hybrid models that combine numerical features with text embeddings to see if we can get the best of both worlds and push performance beyond what either can achieve alone.
    *   **Exploring Different Architectures:** Further research could explore different model architectures for combining these data types.

---

**Thank you. Any questions?** 