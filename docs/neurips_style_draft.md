# NeurIPS-style abstract and paper sketch — **RAG AI Scientist** (primary contribution)

This draft positions **RAG AI Scientist** as the **main star**: a retrieval-augmented assistant for scientific work that grounds answers in **your indexed literature and project references** (papers, challenge briefs, methodology notes), integrates with **editor/agent tooling** (e.g. MCP in Cursor), and supports **methodology-aligned** analysis—not another histogram benchmark paper.

The **ATLAS Higgs Challenge baseline** (`run_atlas_pipeline.py`, AMS, weighted GBDT, Overleaf outputs) appears only as a **case study**: evidence that **grounded retrieval + executable pipelines** can produce **reproducible, citation-aware** physics–ML workflows when the assistant is tied to the same PDFs the human relies on.

---

## Suggested titles (RAG AI Scientist first)

1. **RAG AI Scientist: Retrieval-Grounded Scientific Assistance over Project Knowledge**  
2. **Grounding Agentic Science Workflows with Indexed Literature and Executable Baselines**  
3. **Retrieval-Augmented Methodology Alignment for Physics ML (with a Higgs-Challenge Case Study)**

---

## Abstract (~200 words; polish numbers/names before submission)

Scientific computing increasingly pairs **large language models** with **code and data**, but generic assistants **hallucinate** domain metrics, omit **event weights** and **official evaluation definitions**, and drift from **paper-grounded** methodology. We present **RAG AI Scientist**, a **retrieval-augmented** scientific assistant that answers questions and steers workflows using **project-local corpora**—including PDF technical briefs and references—so that recommendations remain **anchored to source text**. The system exposes **tooling for indexed search and synthesis** (e.g. analysis over embedded chunks with traceable provenance) and is designed to sit alongside **IDE-native agents** so that methodology notes and runnable pipelines stay **consistent**. We illustrate end-to-end use on **particle-physics ML**: indexing materials such as the **ATLAS Higgs Boson Machine Learning Challenge** documentation, aligning preprocessing (sentinel missingness, Monte Carlo weights) and metrics (**Approximate Median Significance**) with the brief, and pairing that grounding with a **transparent baseline pipeline** (weighted boosting, validation AMS, figures, LaTeX bundles). We discuss **limitations** (coverage of the corpus, evaluation of retrieval quality, and scope vs. domain-specific simulators) and position the work as **human-in-the-loop scientific AI** rather than leaderboard chasing.

*(Replace bracketed phrases with your exact product wording, citations, and any quantitative retrieval metrics you measure.)*

---

## Paper skeleton (RAG AI Scientist as Sections 3–4; case study shorter)

### 1. Introduction

- **Problem.** Scientists use LLM assistants that **ignore** weights, metrics, and definitions from the papers actually on disk.  
- **Idea.** **RAG AI Scientist** ties answers and workflow guidance to **indexed project knowledge** + optional **skills** (task-specific instructions).  
- **Contribution (star).** A grounded assistant pattern (architecture at appropriate depth), integration story, and **honest** limits—not “we beat XGBoost on Higgs.”

### 2. Related work

- RAG for science / medicine / engineering; tool-using agents; MCP-style tool protocols.  
- Physics ML challenges (Higgs challenge as **background**, not the paper’s hero).  
- Gap: generic chat vs **corpus-tied** methodology alignment.

### 3. **RAG AI Scientist** (main technical section)

- **Knowledge layer:** what gets indexed (PDFs, YAML configs, internal notes), chunking, embeddings, refresh (`setup-rag`, `--force` when corpus updates).  
- **Query path:** retrieve → optional LLM synthesis vs extractive summary; **sources** surfaced to the user.  
- **Agent integration:** MCP tools (`query_analysis_knowledge`, `retrieve_documents`, `search_papers`, skills) and how they differ from “naked” GPT.  
- **Trust / provenance:** citing chunks, avoiding fabricated AMS definitions.

### 4. Skills and workflow coupling *(if applicable)*

- How **skills** (e.g. response improvement, domain-specific steps) constrain the agent to **follow** project conventions.  
- Connection to reproducible scripts in-repo.

### 5. Case study: ATLAS Higgs Challenge baseline *(supporting, ~1–1.5 pages)*

- **Role:** show that **the same PDFs** the RAG system indexes match the **pipeline’s** choices (weights, `-999`, AMS regulator).  
- Brief mention of `run_atlas_pipeline.py` outputs (`metrics.json`, Overleaf bundle)—**not** a SOTA claim.  
- Optional: screenshot or quote of a **grounded answer** vs generic model mistake (with IRB / privacy care).

### 6. Evaluation

We evaluate **RAG AI Scientist** along three complementary axes: **retrieval quality** against hand-labeled relevance (automatic), **cheap lexical checks** that approximate answer grounding in retrieved text, and **optional human judgments** when publishing qualitative claims. All runnable artifacts live in **`evals/`** in this repository.

#### 6.1 Goals and scope

- **Primary.** Does retrieval surface **the right project documents** (challenge PDF, methodology notes, indexed README fragments) when scientists ask domain questions?  
- **Secondary.** Under the same top-*k* chunks, do answers contain **expected factual tokens** (e.g. regulator value, sentinel encoding) without requiring a cloud LLM judge?  
- **Non-goals.** We do **not** claim leaderboard-scale physics ML; the ATLAS baseline remains a **case study** for methodology alignment, not the evaluation target.

#### 6.2 Gold query set (`evals/data/rag_queries.jsonl`)

We curate **natural-language queries** (weights, AMS, missing-data sentinel, labels, mass variables, …). Each item specifies:

- **`relevant_path_patterns`** — substrings that must appear in chunk **metadata** (`file` / `doc_path` / `source`) for a retrieved chunk to count as **relevant**. This proxies “correct document family” when full chunk IDs are unstable across re-ingests.  
- **`required_terms`** (optional) — tokens that must appear in the **concatenated text** of the top-*k* chunks (case-insensitive). This is a **necessary-condition** check for grounding, not sufficiency.  
- **`difficulty`** — tags stratified reporting (`easy` / `medium` / `hard`).

The set should be expanded whenever the indexed corpus grows; patterns should be **tuned once** after inspecting real retrieval traces.

#### 6.3 Retrieval metrics (implemented)

Using the **same Chroma persist directory and embedding model** as the MCP server (`sentence-transformers/all-MiniLM-L6-v2`, normalized), we report:

| Metric | Definition |
|--------|-------------|
| **Recall@k** | Per query: 1 if **any** of the top-*k* chunks is relevant, else 0. Report **mean** over queries. |
| **MRR** | Mean reciprocal rank of the **first** relevant chunk (0 if none in the retrieved list). |
| **nDCG@k** | Binary relevance per rank; discounts lower positions (standard DCG discount). |

**CLI:** `python evals/run_retrieval_eval.py --rag-db <path/to/.cursor/rag_db> --k-list 5 10`  
**Outputs:** JSON under `evals/results/` plus console aggregates. **`evals/report_aggregate.py`** emits Markdown-friendly tables for the paper.

#### 6.4 Lexical groundedness (implemented)

For queries with **`required_terms`**, we report the fraction of queries for which **all** terms appear somewhere in the union of top-*k* chunk texts. This correlates with “answer substance present in context” when using extractive or constrained generation, and is deliberately **inexpensive** compared to LLM-as-judge.

#### 6.5 Baselines (recommended reporting)

| Condition | Purpose |
|-----------|---------|
| **RAG (full system)** | Indexed retrieval + your normal answer pipeline. |
| **No retrieval** | Same prompts to an LLM **without** injected chunks (documents failure modes of raw chat). |
| **(Optional) Hybrid / BM25** | Future work: sparse retrieval on the same corpus for ablation. |

Report embedding model id, **k**, random seeds (if any), and **corpus version** (hash or ingest date).

#### 6.6 Human evaluation (protocol template)

For reviewer-facing quality claims, use **`evals/human_rubric.md`**: blinded **source fidelity** and **metric correctness** scales, paired comparisons RAG vs no-RAG, and agreement statistics when two annotators are available.

#### 6.7 Limitations (evaluation-specific)

- Path-pattern relevance is a **proxy**; upgrading to stable chunk IDs improves precision when ingestion is frozen.  
- **Lexical checks** miss paraphrases and over-count superficial mentions.  
- Results **transfer only** to corpora and embedding checkpoints documented in the run JSON.

#### 6.8 Automated manuscript (`run_neurips_pipeline.py`)

The repository can **regenerate** a draft article that mirrors this section’s definitions and fills results tables from **`evals/results/*.json`** (when eval runs) plus **`output/atlas_challenge/metrics.json`**. Command:

```bash
python run_neurips_pipeline.py
```

Use **`--skip-eval`** if Chroma is unavailable; use **`--skip-atlas`** only when **`metrics.json`** already exists. The generated LaTeX is **not** camera-ready NeurIPS LaTeX style—it is an **`article`** template you can restyle or paste into the official template.

### 7. Limitations and ethics

- Corpus incompleteness; outdated PDFs; responsibility for physics conclusions; no substitute for collaboration review.

### 8. Conclusion

- **RAG AI Scientist** as the artifact; reproducible physics pipeline as **validation context**.

---

## One-paragraph elevator pitch (reviewer-facing)

> **RAG AI Scientist** is a retrieval-grounded scientific assistant that answers methodology questions and supports agentic workflows using **your own indexed papers and briefs**, reducing hallucinated domain metrics and misaligned evaluation. We demonstrate integration with modern coding agents and use the ATLAS Higgs ML challenge material **only** as a concrete scenario where grounded retrieval matches executable, weighted baseline code.

---

## What *not* to do in this framing

- Do **not** lead with HistGradientBoosting numbers as if they were the novelty.  
- Do **not** claim NeurIPS-level empirical ML contribution without retrieval/agent experiments.  
- Do **not** oversell **RAG AI Scientist** internals you cannot disclose—describe the **user-visible contract** (indexed corpus, tools, provenance).

---

## Checklist before submission

- [ ] Abstract opens with **RAG AI Scientist**, not Higgs.  
- [ ] Case study section explicitly labeled **secondary**.  
- [ ] References include **your** system documentation + challenge PDF + relevant RAG/agent papers.  
- [ ] Pick venue: **ACL demos**, **NeurIPS datasets/agents workshops**, **HCI + AI**, **ML4PS**, etc., matching **actual** evaluations you include.
- [ ] Run **`evals/run_retrieval_eval.py`** (and optionally human rubric); paste aggregates + corpus/version metadata into the camera-ready PDF.

This file is a **draft stance**; paste into your LaTeX template and replace placeholders with **precise branding**, architecture diagrams, and measured metrics.
