# Human evaluation rubric (optional reporting)

Use when publishing qualitative analysis alongside automated metrics. Rate each assistant answer **1–5** per axis.

| Axis | 1 | 3 | 5 |
|------|---|---|---|
| **Source fidelity** | Contradicts cited chunks | Mostly aligned | Fully supported by retrieved text |
| **Metric correctness** | Wrong formula / regulator | Minor ambiguity | Matches indexed brief |
| **Safety / abstention** | Confident hallucination | Hedge sometimes | Says “not in context” when appropriate |
| **Usefulness** | Misleading | Partially actionable | Clear next steps for the scientist |

**Procedure**

1. Sample **N** queries from `evals/data/rag_queries.jsonl` or held-out questions.  
2. Blind **two** annotators when feasible (physics familiarity helpful).  
3. Record Cohen’s κ or % agreement on fidelity scores.  
4. Keep prompts fixed between **RAG AI Scientist** (retrieval on) vs **no-retrieval** baseline for paired comparison.

Document annotator background (physics yes/no) and whether PDFs were visible during grading.
