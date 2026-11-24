# Progress report

What has been done so far, in order, with the reasoning behind each step. Numbers come from real
runs on the H&M data; where something was not measured, this file says so. Decisions and their
alternatives are in `DECISIONS.md` (D-numbers are referenced below).

Contents: [Phase 0](#phase-0-project-skeleton) · [Phase 1](#phase-1-data-subset-and-leakage-safe-splits) ·
[Phase 2](#phase-2-frozen-baselines-and-evaluation-harness) · [Text-search app](#text-search-app-side-project) ·
[Open items](#open-items)

---

## Phase 0: project skeleton

**Goal.** A repository where every later step is reproducible and checked automatically.

- Package `lookalike` under `src/` (repo folder is `StyleMatch`, D3), Python 3.11 pinned with `uv` (D1),
  dependencies locked in `uv.lock`. `requires-python` was later relaxed to `>=3.11` because Kaggle
  runs Python 3.13 (D7).
- `ruff` (lint + format) and `pytest`, run locally and in GitHub Actions on every push.
- Heavy dependencies are optional groups (`tracking` for MLflow, `ml` for torch, `app` for the web
  app), so CI stays fast (D2).
- Working rules in `CLAUDE.md`: seeds and YAML configs instead of hard-coded paths, tests for every
  module, no data or secrets in git, decisions written down.

**Why it matters.** Without a locked environment and automatic checks, results cannot be
reproduced and small breakages go unnoticed.

---

## Phase 1: data subset and leakage-safe splits

**Goal.** A manageable, clean set of fashion items, split so that test results cannot be inflated
by near-copies of training items.

### 1. Subset (`src/lookalike/data/subset.py`, `configs/data/subset.yaml`)
- The H&M catalogue has 105,542 articles and 131 product types.
- Non-fashion groups (cosmetics, furniture, stationery, soft toys, ...) were excluded (D8), because
  a clothing look-alike search should not return toys.
- Product types with fewer than 30 items were dropped (cannot be split), and each type was capped at
  300 items so a few huge types (like trousers) do not dominate.
- Result: **18,123 items**. Items without an image were removed.

### 2. Graded relevance (`relevance.py`)
Image-to-image "is this similar?" has no ground truth in the dataset, so a proxy is built from
catalogue attributes: grade 3 = same product type, colour group and graphical appearance; 2 = same
type plus one of the other two; 1 = same type; 0 = otherwise. This is a **proxy**, not human
judgement, and every result table says so.

### 3. Near-duplicate detection (`dedupe.py`)
- Colour variants of one product share a `product_code`, but different products can also look almost
  identical. If one of them is in train and its twin in test, the test is too easy ("leakage").
- Items whose CLIP image embeddings have cosine similarity above a threshold are linked, and linked
  items are grouped (union-find) so a group never straddles two splits.
- First threshold 0.97 chained generic look-alike basics into giant groups (largest group 1,328
  items, 23% of items dropped). A sweep over 0.97-0.995 on the real data led to **0.98**: 3,055 linked
  pairs, largest group 231, 742 items dropped (D16). The threshold is a judgement call; I did not
  inspect pairs visually.

### 4. Splits (`splits.py`, `splits/splits_v1.csv`)
- Split by `product_code`, never by `article_id`, so colour variants stay together.
- About 20% of product types are held out entirely as `test_unseen`, to test generalisation to
  clothing types never seen in training. Groups that mix held-out and seen types are dropped (D4).
- Final sizes (17,381 items kept of 18,123): **train 9,414 · val 1,345 · test 2,689 · test_unseen 3,933**.
- Two guards run on the real data before the file is written: no `product_code` in two splits, and
  no near-duplicate pair (> 0.98) straddling splits (`check_near_duplicates_together`).

---

## Phase 2: frozen baselines and evaluation harness

**Goal.** Measure how good off-the-shelf models already are, with an evaluation setup that can be
trusted, so later improvements have something honest to be compared against.

### 1. Metrics (`src/lookalike/eval/metrics.py`)
nDCG@10 (linear gain, D9), hit@k, recall@k, MRR, mAP@10, bootstrap confidence intervals and a paired
bootstrap for model differences. Each is tested on tiny hand-computed examples. Binary metrics treat
grade >= 2 as relevant (D10).

### 2. Harness (`harness.py`)
For each query item, rank the gallery by cosine similarity and score the ranking against the proxy
relevance. A query's own colour variants are removed from the results **and** from the ideal ranking
(D11); otherwise scores would be capped below 1 by unreachable items.

### 3. Embedding extraction (`src/lookalike/embed/`)
- Four frozen models, each pinned to an exact Hugging Face commit in `configs/embed/baselines.yaml`:
  ResNet-50 (supervised ImageNet), CLIP ViT-B/16, DINOv2-base (self-supervised), FashionCLIP.
- Run on a Kaggle GPU in resumable shards (a crash resumes where it stopped). fp16 inference,
  float32 storage (D14).
- Bug found and fixed on the real setup: the library no longer exposed the model's commit hash, so
  it was recorded as null. Fix: resolve it from the Hub API and load the model at that revision
  (D15).
- Because these models are frozen, embedding the whole subset before splitting does not leak. Fine-tuned
  models (Phase 5) will need re-extraction.

### 4. Baseline runner (`run_baselines.py`)
Evaluates random, an attribute-based reference, and the four models on `test` and `test_unseen`,
writes `results/baselines_phase2.csv`, and logs one MLflow run per model and split.
Problems met: MLflow 3 rejects a file store (use SQLite) and `@` in metric names (logged as
`ndcg_at_10`). My MLflow test had been skipped locally so I missed the first; installing the
tracking group locally fixed that.

### 5. Results (full tables with intervals and MLflow run ids are in `README.md`)
- FashionCLIP had the highest nDCG@10 of the four real models on both splits; the confidence
  intervals do not overlap with CLIP or ResNet-50 on either split, and DINOv2 is second on `test`.
- The attribute row is **circular** (it encodes the labels the proxy is built from), so it is a
  reference, not a competitor.
- Random scores differ per split (0.012 vs 0.038), so models can be compared **within** a split only.
- Intervals are bootstrap over items and likely understate uncertainty, since colour variants are
  correlated.

---

## Text-search app (side project)

**Goal.** A small, finished software project on the same data: type a description, get matching
catalogue photos. Built separately from the research track so it could be completed quickly.

### Design (D17, D18)
1. **Model.** FashionCLIP text tower at the same pinned revision that produced the image vectors.
   If text and image vectors came from different checkpoints they would live in different spaces and
   similarity would be meaningless.
2. **Images.** The API returns image *URLs*, not image bytes. Thumbnails (256 px JPEG) are static files
   served by FastAPI at `/thumbs/...`, so browsers can cache them. The URL scheme lives in one
   function (`thumbnail_url`), so moving images to object storage later changes one place. MinIO was
   rejected: an extra service with no benefit on one machine.
3. **Data preparation on Kaggle** (`app/tools/export_assets.py`), because the original images are
   there and the laptop is short on disk. It checks the item ids match the vectors, writes
   `metadata.csv`, makes the thumbnails, and bundles everything. It is a CPU job (image decoding and
   resizing); no neural network runs in it. It is resumable.
4. **Vector database: Qdrant** (`qdrant_store.py`). Cosine distance, point id = article id, payload =
   name / type / colour / description. Type and colour filters run *inside* the search. FAISS was
   rejected because it has no payload or filtering. Two modes behind one client API: embedded storage
   in a folder (used here, no Docker) or a Qdrant server. An exact NumPy `BruteForceStore` is kept as
   the reference to test against.
5. **Backend** (`api.py`, FastAPI). `POST /search`, `GET /facets` (feeds the dropdowns), `GET /health`.
   The app is built by a factory function that receives the store and encoder, so tests can use a
   fake encoder and swap vector stores. Input is validated (empty query, `k < 1` give HTTP 422).
6. **Frontend** (`app/frontend/index.html`). One plain HTML/JS page: search box, type and colour
   dropdowns, card grid. It builds cards with `textContent`, not `innerHTML`, so catalogue text
   cannot inject HTML.
7. **Environment.** Uses the existing conda `pytorch` environment so torch is not installed twice
   into the project environment (disk space).

### Later changes
- **One-command start.** `python -m app --data-dir ... --hf-home ...` (`app/__main__.py`) replaces
  the long `uvicorn app.backend.main:build_app --factory --port 8000` command. The factory exists so
  that importing the module does not load a 600 MB model.
- **Minimum score** (D19). Nearest-neighbour search always returns k items, so "laptop" returned the
  least-bad bags. I measured the best-result score for 30 fashion and 30 non-fashion queries of my own
  on the real index: fashion 0.270-0.384 (median 0.339), non-fashion 0.193-0.340 (median 0.264). The
  groups overlap, so no threshold is clean. I proposed 0.30; after using the app, Anas lowered it to
  **0.275** because 0.30 cut relevant items. On my 60 queries that keeps 29/30 fashion queries and lets
  10/30 non-fashion ones through (several of those match real catalogue items such as a takeaway mug
  or banana-print socks). It is a demo setting, not a validated optimum.

### What was verified
- Qdrant returns the same ranking and scores as exact brute force on 300 random vectors, with and
  without filters. Changing the distance from cosine to dot product made 4 of these tests fail, which
  shows the tests really check it.
- The API tests run against both stores. The full suite passed at the last run (126 tests) and ruff is
  clean.
- The real app was run on the real 18,123-item index: "laptop", "the capital of france" and
  "asdfghjkl" returned nothing at 0.30, while "red floral summer dress" and "black leather boots"
  returned items. Anas also ran the app himself and reported that it works.

### What was NOT verified
- Search quality. There is no labelled text-query set, so no accuracy number exists; only hand-tried
  examples.
- Qdrant in Docker server mode (the Docker daemon was not running when I tried).
- Speed under load.

---

## Open items

- Phase 0-2 interview questions are waiting for Anas's own answers in `LEARNING_LOG.md`.
- Check that CI is green on GitHub after the latest pushes.
- Re-read the competition rules for any clause on redistributing derived data (D6).
- Optional Phase 2 extras: product-level (cluster) bootstrap, text-only baseline.
- App: small labelled query set to measure text-search quality; try the Docker server mode.
- Research track: Phase 3 (Qdrant hybrid search for image look-alikes) through Phase 8 per `PLAN.md`.
