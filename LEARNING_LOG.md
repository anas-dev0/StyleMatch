# Learning log

One entry per module: what it does, why this approach, alternatives rejected. At the end of
each phase Anas answers the interview questions **in his own words**. Claude never writes the
answers.

## About this file

The original was never committed and could not be recovered. Rebuilt on 2026-11-23.
- **[RECONSTRUCTED]** notes for Phases 0-2 describe what the code does. "Why" appears only
  where the repo records it (see `DECISIONS.md`), otherwise it says *not recorded*.
- Anas's original interview answers are **lost**. The questions are listed again with empty
  answers. They are not rewritten.
- **[NEW]** notes are for Phase 3 modules, written when the module was built.

---

## Phase 0: setup [RECONSTRUCTED]

- `pyproject.toml` / `uv.lock` / CI: Python 3.11, uv, ruff, pytest, pre-commit. Heavy packages
  are optional groups so CI stays fast (R5).
- `config.py`: loads YAML; every path and hyper-parameter lives in `configs/`.

**Interview questions** (answers: Anas)
- Why group-aware splits?
  > _answer lost; to be rewritten by Anas_
- What does MLflow track here?
  > _answer lost; to be rewritten by Anas_

## Phase 1: subset and splits [RECONSTRUCTED]

- `data/subset.py`: cleans attributes and drops non-fashion groups, items without an image,
  and types under 30 items. Caps 300 per type (R1).
- `data/relevance.py`: proxy grade 0-3 from type, colour group and graphical appearance.
  Missing values never match.
- `data/dedupe.py`: pairs with cosine > threshold on frozen CLIP. The similarity matrix is
  computed one block of rows at a time to bound memory. Threshold 0.98 (D16).
- `data/splits.py`: union-find over `product_code` plus near-duplicate edges. 20% of types
  held out; mixed groups dropped; 70/10/20 (R2). Leakage checks run before writing.
- Alternatives rejected: *not recorded*.

**Interview questions** (answers: Anas)
- What leaks if you split by `article_id`?
  > _answer lost; to be rewritten by Anas_
- Why is relevance only a proxy?
  > _answer lost; to be rewritten by Anas_

## Phase 2: baselines and harness [RECONSTRUCTED]

- `embed/models.py`: one wrapper for ResNet-50, CLIP, DINOv2 and FashionCLIP. Pinned commit,
  fp16 on GPU (R3, R4).
- `embed/extract.py`: resumable sharded extraction. Each shard is written to a temp file and
  then renamed, so a crash never leaves a half-written shard. The encoder is injected, so
  it is testable without torch.
- `eval/metrics.py`: hit, recall, MRR, AP, nDCG (linear gain), bootstrap and paired
  bootstrap (D9, D10).
- `eval/harness.py`: cosine ranking with the own product excluded from results and from the
  ideal (D11).
- `eval/run_baselines.py`: per model and split, CSV plus MLflow, with a random floor and an
  attributes reference (R8).
- Alternatives rejected: *not recorded*.

**Interview questions** (answers: Anas)
- What is an embedding?
  > _pending_
- Why cosine similarity?
  > _pending_
- What does nDCG add over precision@k?
  > _pending_

---

## Phase 3: Qdrant and hybrid search [NEW]

**`embed/text.py`**: one caption per item (default `detail_desc`), embedded with FashionCLIP's
*text* tower at the same pinned revision as the image tower. Image and text vectors therefore
share a space. Captions are cut at CLIP's 77 tokens. It reuses `run_extraction`, so shards and
resume work the same way as for images. Rejected: an attribute caption as the main signal,
because it contains the three labels the proxy grades on and would grade itself (D20).

**`index/sparse.py`**: BM25 implemented in about 60 lines. The document vector holds
`idf * saturated tf`; the query vector is 1 per distinct term; their dot product is exactly
the BM25 score. Because the full weight is stored, Qdrant's sparse dot product returns the
same number, and a test checks it. k1 controls how fast repeated words stop adding score;
b controls how much long descriptions are penalised. Rejected: fastembed BM25 (a different
tokeniser, an extra download) and SPLADE (a learned model) (D21).

**`index/faiss_baseline.py`**: `IndexFlatIP` on unit vectors means exact cosine search,
the ground truth for approximate search. FAISS cannot filter, so `search_excluding` fetches
extra results and drops the query's own product.

**`eval/fusion.py`**: Reciprocal Rank Fusion, `sum 1/(60 + rank)`. It uses ranks, not scores,
so a cosine in [-1, 1] and an unbounded BM25 score can be combined without calibration.
Rejected: a weighted score sum (needs normalisation) and learned weights (the only labels are
the proxy) (D22). Qdrant counts ranks from 0, so it is sent `k - 1`. A test checks that both
give the same list.

**`eval/rankings.py`**: scores a ranking computed elsewhere with the Phase 2 metrics, and
refuses any ranking that returns the query's own product (D11). A test checks it agrees
exactly with `harness.evaluate_embeddings`.

**`index/qdrant_store.py`**: one point per article with named vectors `image`, `text` and
`bm25`, plus a payload for filters. Hybrid search runs one prefetch per signal, then RRF.
Local mode (`url=None`) is a full scan: fine for tests, useless for measuring HNSW.

**`eval/run_hybrid.py`**: exact ranking per signal over each split, fused offline, scored
like Phase 2, with a paired bootstrap vs image-only. Systems that use attribute words are
flagged `circular` (D23).

**`index/hnsw_recall.py`**: HNSW recall@10 vs exact FAISS for several `ef` values, with
latency. `ef` is how many candidates HNSW keeps while walking the graph: higher means better
recall and slower queries. Thresholds are lowered so the graph is really built and used, and
`indexed_vectors_count` proves it (D24).

**Interview questions** (answers: Anas)
- Exact vs approximate search?
  > _pending_
- What is RRF and why use it?
  > _pending_
