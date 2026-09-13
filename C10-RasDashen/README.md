# C10-RasDashen: Optimizing RAG Document Retrieval for Agronomic Advice

**Cohort:** 10
**Team:** RasDashen
**Competition:** [Agricultural Extension RAG — Smart Retrieval for Farmers](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers)

## Problem

Given a smallholder farmer's question, return the top-5 most relevant documents from a
knowledge base of agricultural extension material (crop diseases, pests, nutrient
deficiencies, soil management, climate adaptation, and fertiliser advice). Scored on
**nDCG@5** against expert relevance judgements. See `doc/problem_statement.pdf` for the
full values-led problem statement behind this project.

---

## Dataset

The knowledge base and labelled query set are provided by the competition (not
redistributed in this repo — access them directly on Kaggle by joining the competition
and attaching its dataset in the notebook's Input panel). At a high level:

- **Documents**: agricultural extension reference material — crop disease guidance,
  pest management, nutrient deficiency symptoms, soil management, climate adaptation,
  and fertiliser advice.
- **Queries**: short, plain-language farmer questions, a subset of which are labelled
  with known relevant document ids for local evaluation.

Full dataset design, sourcing, labelling strategy, and ethical/bias considerations are
documented in `doc/data_card.pdf`.

## Training Pipeline

This project does not train or fine-tune a model — it implements and compares four
**retrieval** methods, all running on CPU:

1. **TF-IDF cosine similarity** — reproduces the competition's provided baseline.
2. **BM25** — a stronger sparse retriever for short, keyword-heavy queries.
3. **Dense embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) — catches semantic/
   paraphrase matches sparse methods miss.
4. **Hybrid fusion** — BM25 + dense rankings combined via Reciprocal Rank Fusion (RRF),
   with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranking the fused
   shortlist for a final precision boost.

All four methods are implemented in `agronomic-rag-retrieval.ipynb`.

## Evaluation

Before generating the final submission, all four methods are scored with **nDCG@5** on
the labelled training queries, and the method with the best local score is used to
generate the final submission — the choice is backed by measured evidence rather than
assumption. See the notebook's evaluation section for the comparison table and the
`FINAL_RANK_FN` selection logic.

## Reproduction

1. Open `agronomic-rag-retrieval.ipynb` on Kaggle with the competition data attached
   (Input panel), or locally with the competition CSVs in the working directory.
2. Run all cells top to bottom. Dependencies (`rank_bm25`, `sentence-transformers`) are
   installed automatically in the first code cell; see `requirements.txt` for a local
   environment.
3. If the auto-detected file/column names don't match the real competition files,
   adjust the `CONFIG` dictionary near the top of the notebook.
4. Check the printed nDCG@5 comparison across TF-IDF / BM25 / dense / hybrid.
5. `submission.csv` is written and schema-validated automatically at the end of the
   notebook.

## Appendix

- `doc/problem_statement.pdf` — Challenge 1: values-led problem statement.
- `doc/data_card.pdf` — Challenge 2: dataset design and ethical considerations.
- `doc/impact_statement_card.pdf` — Challenge 3: broader social/ethical impact
  assessment.
- `doc/stakeholder_engagement.pdf` — Challenge 4: stakeholder engagement plan.
- `requirements.txt` — Python dependencies for local runs.

**Note on submission consistency:** the code and files in this repository are kept in
sync with the notebook actually submitted to the Kaggle competition. If you update one,
update the other.
