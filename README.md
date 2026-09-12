# Optimizing RAG Document Retrieval for Agronomic Advice

**Competition:** [Agricultural Extension RAG — Smart Retrieval for Farmers](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers)

## Problem

Given a smallholder farmer's question, return the top-5 most relevant documents
from a knowledge base of agricultural extension material (crop diseases, pests,
nutrient deficiencies, soil management, climate adaptation, and fertiliser advice).
Scored on **nDCG@5** against expert relevance judgements.

## Approach

Rather than relying on a single retrieval method, this solution builds and compares
four retrievers, then submits whichever wins on local validation:

1. **TF-IDF cosine similarity** — the baseline provided by the competition.
2. **BM25** — a stronger sparse retriever for short, keyword-heavy farmer queries.
3. **Dense embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) — catches
   semantic/paraphrase matches that sparse methods miss (e.g. "corn" vs "maize").
4. **Hybrid fusion** — BM25 and dense rankings are combined with **Reciprocal
   Rank Fusion (RRF)**, then the fused shortlist is reranked with a
   **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for a final
   precision boost.

All retrieval runs on **CPU** — no GPU or model training required.

## Validation

Before generating the final submission, all four methods are scored with
**nDCG@5** on the labelled training queries, so the choice of final method is
backed by measured evidence rather than assumption.

## Repo contents

| File | Description |
|---|---|
| `agronomic-rag-retrieval.ipynb` | Full pipeline: data loading, all four retrievers, local nDCG@5 evaluation, and submission generation. |
| `requirements.txt` | Python dependencies. |

## How to run

1. Open the notebook on Kaggle with the competition data attached (see the
   Input panel), or locally with the competition CSVs in the working directory.
2. Run all cells top to bottom.
3. Check the printed nDCG@5 comparison across TF-IDF / BM25 / dense / hybrid.
4. `submission.csv` is written and validated automatically.

## Notes

- No dataset is bundled in this repo — the competition's data is not
  redistributable and must be accessed directly on Kaggle.
- The notebook auto-detects common file/column naming patterns for the
  competition data; see the `CONFIG` cell near the top if you need to adjust
  file or column names.
