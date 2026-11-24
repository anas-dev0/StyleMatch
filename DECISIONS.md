# Decisions

Format: what was decided, why, and the alternatives considered.

## About this file

The original `DECISIONS.md` was never committed (it was gitignored) and could not
be recovered. On 2026-11-23 it was rebuilt from the code, configs, README and git
history. Three kinds of entry:

- **[RECONSTRUCTED]**: the decision is visible in the code or configs. The "why" is
  quoted only when the repo itself records it; otherwise it says *not recorded*.
- **[LOST]**: the number is cited in the repo but its text is gone.
- **[NEW]**: written on 2026-11-23 for Phase 3, not yet reviewed by Anas.

Numbers D1-D8 and D12-D15 existed (the numbering reaches D16), but nothing in the
repo cites them and their content is unknown. They are left empty and are **not
reused**. D17-D19 are the text-search app's decisions (see below). Phase 3
entries were first numbered D17-D22 and were moved to D20-D25 on 2026-11-24, before they
were ever committed, to avoid clashing with the app. New entries continue from D26.

---

## Numbered decisions recovered from citations

### D9 [LOST, partially recoverable]: metric conventions
Cited with D10 in `src/lookalike/eval/metrics.py` ("Conventions (see DECISIONS.md
D9-D10)"). It is not known which of the conventions below D9 covered, or why.
What the code implements:
- A query with no relevant item gets NaN for binary metrics (undefined, not zero), so
  those queries are skipped when averaging.
- nDCG uses linear gain: `DCG = sum(grade_i / log2(i + 1))`.
- AP@k divides by `min(R, k)`.
Why / alternatives: not recorded.

### D10 [RECONSTRUCTED]: binary relevance threshold
Grade >= 2 counts as "relevant" for hit, recall, MRR and AP
(`configs/eval/baselines.yaml: relevance_threshold: 2`). nDCG uses the graded 0-3
values directly. Why / alternatives: not recorded.

### D11 [RECONSTRUCTED]: exclude the query's own product
Gallery items sharing the query's `product_code` (its colour variants) can never be
returned. They are also removed from the "all relevant items" set used by the ideal
ranking and by recall. Why (recorded in `eval/harness.py`): "Otherwise scores would be
inflated or capped below 1." Alternatives: not recorded.

### D16 [RECONSTRUCTED]: near-duplicate threshold 0.98
Cosine similarity above 0.98 on frozen CLIP embeddings marks a near-duplicate pair.
Each pair is merged into one split group. Recorded in `configs/data/subset.yaml`:
"swept 0.97-0.995 on real data". Commit `ab67e18` changed it from PLAN's ~0.97 to
0.98. The sweep's results and why 0.98 was picked: not recorded.

---

## Unnumbered decisions recovered from code (original numbers unknown)

**R1 [RECONSTRUCTED]: subset filtering.** Exclude non-fashion product groups (Unknown,
Cosmetic, Items, Furniture, Garment and Shoe care, Stationery, Interior textile, Fun)
and the type Soft Toys. Why (config): "a look-alike search for clothes should not return
toys". Cap 300 items per type, "chosen from the real distribution (105,542 articles, 131
types); target ~20k". Drop types with fewer than 30 items: "types with fewer items cannot
be split sensibly". Items without an image, or with a missing or "unknown" key
attribute, are dropped.

**R2 [RECONSTRUCTED]: split construction.** Union-find groups items by
`product_code`, plus near-duplicate edges (D16). 20% of product types are held out as
`test_unseen`. Groups that mix held-out and seen types are dropped, "so `test_unseen`
stays clean" (`data/splits.py`). The remaining groups are assigned whole to train/val/test
70/10/20 in seeded random order. Output is `splits/splits_v1.csv` (ids only).

**R3 [RECONSTRUCTED]: pinned model revisions.** Each baseline encoder is loaded at a
fixed Hub commit (`configs/embed/baselines.yaml`, commit `ec320ae`), "the commit that
produced the Phase 2 embeddings". Without a pin, the resolved hash is written to
`meta.json`.

**R4 [RECONSTRUCTED]: embedding definitions.** resnet = global-average-pooled final
feature map (2048-d); clip = `get_image_features` projection (512-d); dinov2 = pooled CLS
token (768-d) (`embed/models.py`). Default Hugging Face preprocessing. fp16 on GPU because
"T4 has no bf16".

**R5 [RECONSTRUCTED]: heavy dependencies are optional.** torch, transformers, pillow
and mlflow live in optional dependency groups and are imported inside functions, "so CI
stays fast" (`pyproject.toml`) and the test suite runs without them.

**R6 [RECONSTRUCTED]: experiment tracking.** MLflow logs to a local SQLite file
because "MLflow 3.x no longer accepts the plain-folder backend"
(`configs/eval/baselines.yaml`). The store is not committed, so the committed CSV in
`results/` is the record (README).

**R7 [RECONSTRUCTED]: data licence.** The H&M data is non-commercial / academic only,
so no images, embeddings or weights are published (README).

**R8 [RECONSTRUCTED]: evaluation reporting.**
- Queries = gallery = one split.
- Mean with a 95% percentile bootstrap over queries (1000 resamples, seed 42), and a
  paired bootstrap against a reference model (CLIP).
- A random floor, plus an `attributes` one-hot row that is "a reference, not a
  competitor" because it is circular with the proxy.
- Compare within a split only, because base rates differ.
- The README states that the item-level bootstrap likely understates uncertainty.

---

## Text-search app decisions (D17-D19)

These were written by Anas while building `app/` and were never committed. They are cited in
`PROGRESS.md`, `README.md` and `configs/app/app.yaml`. The text below is only what
`PROGRESS.md` records.

### D17, D18 [LOST, partially recoverable]: text-search app design
**What (from `PROGRESS.md`, "Design (D17, D18)"):** FashionCLIP text tower at the same pinned
revision as the image vectors. The API returns thumbnail URLs served as static files (MinIO
rejected). Data is prepared on Kaggle (`app/tools/export_assets.py`). Qdrant vector store with
cosine distance and payload filters (FAISS rejected: no payload or filtering), embedded or server
mode, checked against an exact NumPy `BruteForceStore`. FastAPI backend built by a factory;
plain HTML/JS frontend using `textContent`.
**Not recorded:** which of these points is D17 and which is D18.

### D19 [LOST, partially recoverable]: app minimum score 0.275
**What (from `PROGRESS.md`):** results below `min_score` are dropped. Best-result scores on 30
fashion and 30 non-fashion hand-written queries overlapped (fashion median 0.339, non-fashion
median 0.264). 0.30 was proposed; Anas lowered it to 0.275 after using the app. It is a demo
setting, not a validated optimum.

---

## Phase 3 decisions

### D20 [NEW]: text signal = FashionCLIP text tower on the product description
**What:** A dense text vector per item from FashionCLIP's text encoder
(`get_text_features`), loaded at the **same pinned revision** as the FashionCLIP image
encoder (`7e3ba62...`). The input is `detail_desc` only, truncated to CLIP's 77 tokens.
A second variant prefixes type, colour and graphical appearance. It is a circular
reference only, like the `attributes` row.
**Why:** FashionCLIP is the strongest frozen image baseline (README, Phase 2). Using its
text tower keeps one model and one pinned revision, and image and text share an embedding
space (which allows cross-modal search later). `detail_desc` is free text written for
customers, not the three attributes the proxy is built from, so it does not grade itself.
**Alternatives:**
- Generic CLIP text: weaker domain fit.
- A sentence-transformer: a second model to pin and justify.
- Attribute captions as the main signal: circular with the proxy.

### D21 [NEW]: sparse signal = BM25 computed in the repo
**What:** Tokenise (lowercase `[a-z0-9]+`, small English stopword list, no stemming), with
k1 = 1.2 and b = 0.75. IDF `ln((N - n + 0.5) / (n + 0.5) + 1)` is fitted on the
gallery being searched. Document weights include the IDF. The query vector is 1 per
distinct known term, so a dot product equals the BM25 score. Default field:
`detail_desc`. A BM25 vector over the attribute words is a circular reference only.
**Why:** It is deterministic and testable. The same weights are uploaded to Qdrant
(no server-side IDF modifier), so offline and Qdrant scores are identical.
**Alternatives:**
- fastembed `Qdrant/bm25`: an extra dependency and model download, and a different
  tokeniser.
- SPLADE: a learned sparse model, heavier and another model to justify.

### D22 [NEW]: fusion = Reciprocal Rank Fusion, k = 60, top-100 per signal
**What:** Each signal returns its top 100 candidates (after D11 exclusion).
`score = sum over signals of 1 / (k + rank)` with 1-based rank and k = 60, the value from
Cormack et al. 2009. Ties go to the item with the better best rank, then the lower index.
Offline fusion is what gets evaluated. Qdrant's native RRF is used for serving.
Observed in Qdrant local mode (qdrant-client 1.20.0): the top item gets 1/k, so Qdrant
counts ranks from 0. Passing `k - 1` to Qdrant reproduces the offline formula, and this
is asserted in `tests/test_qdrant_store.py`.
**Why:** RRF uses ranks only, so the cosine scores of the dense signals and the unbounded
BM25 scores need no normalisation. It has no trained weights, so it cannot overfit the
proxy labels.
**Alternatives:**
- Weighted score sum: needs score calibration across signals.
- Learned fusion: needs labels. The only labels are the proxy, so it would learn the
  attributes.

### D23 [NEW]: hybrid evaluation uses exact retrieval per signal
**What:** For the hybrid-vs-single-signal table, each signal is ranked exactly (cosine
or BM25 over the whole split), then fused offline. Queries = gallery = one split.
Own-product exclusion (D11) is applied per signal before fusion. Metrics, threshold
(D10), bootstrap and paired bootstrap follow Phase 2. The reference system is image-only
FashionCLIP.
**Why:** It keeps the fusion comparison free of ANN error and does not need a Qdrant
server on Kaggle. ANN error is measured separately (D24). Caution: `detail_desc` often
names the garment type ("Jersey top with ..."), so the text signals are partly aligned with
the proxy too. A text or hybrid gain on the proxy is only provisional until the gold set. Systems that use the attribute
signal, or a payload filter on product type, are circular and are labelled as references
until the Phase 4 gold set.
**Alternatives:** Evaluate through Qdrant end to end. That mixes two error sources and
makes CI depend on a server.

### D24 [NEW]: HNSW recall measured on a Qdrant server against FAISS flat
**What:**
- Gallery = FashionCLIP image vectors of the whole subset. Queries = 1000 random gallery
  items (seed 42), with no exclusion.
- Exact top-10 from FAISS `IndexFlatIP` on L2-normalised vectors. Approximate top-10
  from Qdrant HNSW (m = 16, ef_construct = 100).
- Sweep `hnsw_ef`, and report recall@10 plus p50/p95 latency per value.
- `indexing_threshold` and `full_scan_threshold` are lowered so the HNSW graph is
  really used, and `indexed_vectors_count` is reported as proof.
**Why:** Qdrant's local (embedded) mode does a brute-force scan, so it cannot show HNSW
loss. A small collection can also silently fall back to full scan. 10k-1M vectors are
Phase 6.
**Alternatives:** hnswlib or FAISS HNSW alone: they don't measure the engine actually
served.

### D25 [NEW]: `index` dependency group
**What:** `faiss-cpu` and `qdrant-client` go in a new optional `index` group (locked in
`uv.lock`), imported lazily. Their tests use `pytest.importorskip`.
**Why:** Same reasoning as R5.
**Open:** CI installs only the base group, so these tests are skipped there. Adding
`--group index` to the CI sync step would run them; it needs Anas's agreement.
