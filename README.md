# C10-RasDashen: Optimizing RAG Document Retrieval for Agronomic Advice

**Cohort:** 10 | **Team:** RasDashen
**Competition:** [Agricultural Extension RAG — Smart Retrieval for Farmers](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers)

Given a smallholder farmer's question, return the top-5 most relevant documents from
a knowledge base of agricultural extension material (crop diseases, pests, nutrient
deficiencies, soil management, climate adaptation, fertiliser advice), scored on
**nDCG@5**. Full values-led motivation in `docs/problem_statement.pdf`.

## Dataset

The document corpus and query set are provided by the competition. Documents are
agricultural extension reference material (crop disease guidance, pest management,
nutrient deficiency symptoms, soil management, climate adaptation, fertiliser advice).
Queries are short, plain-language farmer questions; a labelled subset includes known
relevant document ids for local evaluation. The dataset is not redistributed in this
repo per Kaggle's competition rules — `data/download_data.py` fetches it directly via
the Kaggle API. Full sourcing, labelling strategy, and bias/ethics considerations are
in `docs/data_card.pdf`.

## Training Pipeline

No model is trained or fine-tuned — this is a **retrieval** pipeline. Four methods are
implemented and compared, all CPU-only:

1. **TF-IDF cosine similarity** — reproduces the competition's baseline.
2. **BM25** — stronger sparse retrieval for short, keyword-heavy queries.
3. **Dense embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) — catches semantic/
   paraphrase matches sparse methods miss.
4. **Hybrid fusion** — BM25 + dense combined via Reciprocal Rank Fusion (RRF), with a
   cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranking the fused shortlist.

Preprocessing is minimal by design (lowercasing, punctuation stripping, tokenisation)
since the core design choice is *which retriever*, not feature engineering. No
hyperparameter search is performed beyond RRF's damping constant and candidate pool
size, both set to standard defaults (`k=60`, pool=20) and left as clearly labelled
constants in the notebook for easy tuning.

## Evaluation

All four methods are scored with **nDCG@5** on the labelled training queries before
generating the final submission, and the best-scoring method (`FINAL_RANK_FN`) is used
for the actual submission — so the choice is backed by measured local evidence, not
assumption. The notebook prints a side-by-side comparison table for all four methods.

## Reproduction

Run from the repository root:

1. `pip install -r requirements.txt`
2. `python data/download_data.py` (requires a Kaggle API token — see the script's
   docstring; you must first join the competition on Kaggle)
3. Open `scripts/agronomic-rag-retrieval.ipynb` and run all cells top to bottom. If
   running on Kaggle instead, attach the competition dataset in the Input panel and
   skip step 2.
4. If auto-detected file/column names don't match, adjust the `CONFIG` dict near the
   top of the notebook.
5. Check the printed nDCG@5 comparison, then `submission.csv` is written and
   schema-validated automatically at the end of the notebook.

## Appendix

**Team members:** Olumide Adekunle, Misturah, Sabona, Ogwuche Moses

**Mentor:** [Insert mentor's name]

**Repository contents:**
- `README.md` — this file
- `docs/` — all four Cohort Challenge submissions (problem statement, data card,
  impact statement, stakeholder engagement plan)
- `scripts/agronomic-rag-retrieval.ipynb` — full retrieval pipeline and evaluation
- `data/download_data.py` — fetches the competition dataset via the Kaggle API
- `requirements.txt` — Python dependencies

**Submission consistency:** the notebook in `scripts/` is kept identical to the
version submitted on the Kaggle competition page. If one is updated, the other is
updated to match.
