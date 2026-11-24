# LookAlike

Fashion search on the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
catalogue, built around CLIP-family image and text embeddings. The repository has two parts:

| Part | What it is | Status |
|---|---|---|
| **Text-search app** (`app/`) | Type a description such as "black leather boots" and get the closest catalogue photos. FashionCLIP text encoder, Qdrant vector database, FastAPI backend, plain HTML page. | Working locally |
| **Visual look-alike research** (`src/lookalike/`) | Image-to-image look-alike search with frozen baselines, a leakage-safe evaluation harness, and (planned) LoRA fine-tuning and hybrid search. See `PLAN.md`. | Phases 0-2 done |

Data licence: the H&M data is for non-commercial / academic use. No images, thumbnails,
embeddings or model weights are published in this repository, and the app is meant to run on
your own machine only. See `DECISIONS.md` D6 if you have the local copy.

## Text-search app

```
 browser  --POST /search {"query": "red floral dress"}-->  FastAPI (app/backend/api.py)
                                                              |
                          FashionCLIP text tower  <-----------+  turns the text into a 512-number vector
                                                              |
                          Qdrant (cosine similarity)  <-------+  nearest image vectors, optional type/colour filter
                                                              |
 browser  <--JSON: rank, score, name, type, colour, image_url--+
 browser  --GET /thumbs/0108775015.jpg-->  FastAPI static files (256 px thumbnails)
```

CLIP-style models put text and images in the same vector space, so the vector of a sentence lands
near the vectors of photos that match it. The photos were embedded once, offline, on Kaggle; at
query time only the (small) text tower runs. Results below `min_score` (in `configs/app/app.yaml`)
are dropped, so an unrelated query such as "laptop" returns nothing.

### Run it

1. **Prepare the data once, on Kaggle** (the images live there): run
   `python -m app.tools.export_assets --config configs/app/app.yaml --out /kaggle/working/app_data`,
   zip the result, download it and extract it to a folder outside OneDrive, for example
   `C:\lookalike_data\app_data`. It contains the FashionCLIP vectors, `metadata.csv` and `thumbs/`.
2. **Load the vectors into Qdrant** (embedded mode, no Docker needed):
   `python -m app.tools.ingest` with `LOOKALIKE_DATA_DIR` pointing at that folder.
3. **Start the app**, in an environment that has PyTorch, transformers and the `app` dependency group:
   ```
   python -m app --data-dir C:\lookalike_data\app_data --hf-home C:\lookalike_data\hf_cache
   ```
   Then open http://127.0.0.1:8000. Interactive API docs are at `/docs`.

More detail (Docker Qdrant server, options): `app/README.md`.

### Honest limits

- Text-search quality has **not** been evaluated with a labelled query set, so this README
  reports no accuracy number for it. It has only been tried by hand.
- `min_score` is a hand-chosen demo setting: fashion and non-fashion query scores overlap
  (`DECISIONS.md` D19).
- Embedded Qdrant allows one process per storage folder and is meant for development.
  The Docker server mode is implemented but has not been run end to end.
- Only 18,123 catalogue items (the project subset) are searchable, not the full 105,542 articles.

## Repository layout

```
app/                text-search app (backend/, frontend/, tools/ for data export and ingest)
src/lookalike/      research code: data/ (subset, dedupe, splits, relevance), embed/, eval/
configs/            YAML settings for every script (no hard-coded paths)
tests/              pytest tests (metrics are checked on small hand-computed examples)
splits/             product-level train/val/test splits (ids only)
results/            baseline results CSV
PLAN.md             8-phase research plan
PROGRESS.md         what has been done so far, step by step
```

## Development

```
uv sync --locked --group app        # project environment
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

CI (GitHub Actions) runs ruff and pytest. Splits are made by `product_code`, never by
`article_id`, and a test asserts that no product appears in two splits.

## Phase 2: frozen baselines

Queries and gallery are the items of one split; a query's own product (colour variants) is
removed from its results. Relevance is a **proxy** graded 0-3 from product type, colour group
and graphical appearance (not human judgement). hit@k counts grade >= 2 as relevant. Intervals
are 95% bootstrap over queries (items), which likely understates uncertainty because colour
variants of a product are correlated. Compare models **within** a split only: the two splits
have different base rates (random scores 0.012 vs 0.038 nDCG@10).

**Seen product types (`test`), 2689 queries**

| Model | nDCG@10 [95% CI] | hit@1 | hit@10 | MLflow run id |
|---|---|---|---|---|
| random (floor) | 0.012 [0.011, 0.013] | 0.007 | 0.079 | `f5e9d28a9f464b9caf8f99878fe95d41` |
| ResNet-50 | 0.392 [0.381, 0.403] | 0.447 | 0.766 | `3e2670e8827142d39ef40ec9b14dcce4` |
| CLIP ViT-B/16 | 0.457 [0.446, 0.468] | 0.506 | 0.828 | `527334079ad44a0aa1fd12b6ee71a102` |
| DINOv2-base | 0.488 [0.477, 0.499] | 0.500 | 0.807 | `39ac70c949cb42319cb45aef79a1832d` |
| FashionCLIP | 0.515 [0.504, 0.525] | 0.550 | 0.868 | `aaabd5dba27c46a098b03d4856e92bd1` |
| attributes (circular reference) | 0.678 [0.668, 0.687] | 0.809 | 0.976 | `d24cc3c367844580adae0f741c112770` |

**Held-out product types (`test_unseen`), 3933 queries**

| Model | nDCG@10 [95% CI] | hit@1 | hit@10 | MLflow run id |
|---|---|---|---|---|
| random (floor) | 0.038 [0.036, 0.040] | 0.025 | 0.245 | `5bdf80785ec544ada815f75d9bc772f8` |
| ResNet-50 | 0.564 [0.556, 0.570] | 0.613 | 0.894 | `dba1a23982e845e1a23ccdeaf1db1a89` |
| CLIP ViT-B/16 | 0.632 [0.625, 0.640] | 0.674 | 0.920 | `6edcaa7ee4f24aa58ed6d59744533682` |
| DINOv2-base | 0.606 [0.599, 0.612] | 0.631 | 0.911 | `eb85d364bfb74dde823b27bd4d0ce8a8` |
| FashionCLIP | 0.653 [0.645, 0.660] | 0.689 | 0.939 | `7cc03f50f6294958a3f064792d6aad0f` |
| attributes (circular reference) | 0.926 [0.922, 0.930] | 0.968 | 0.998 | `9a0918c73605425d94dd72552a070a55` |

Raw results: `results/baselines_phase2.csv`. The `attributes` row one-hot encodes the same
labels the proxy is built from, so it is a reference, not a competitor. Embeddings are from the
exact model commits pinned in `configs/embed/baselines.yaml` (fp16, default preprocessing).
Run ids refer to a local MLflow SQLite store that is not committed; the CSV is the record.
