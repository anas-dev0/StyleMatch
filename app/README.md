# Text search app

Type a description, get the closest catalogue photos. FashionCLIP (text tower) -> Qdrant
(vectors + item fields, optional type/colour filter) -> FastAPI -> plain HTML page.

Data is derived from H&M images (non-commercial / academic licence): keep `app_data/` local.

## One-time data preparation
1. On Kaggle: `python -m app.tools.export_assets --config configs/app/app.yaml --out /kaggle/working/app_data`
   then zip and download it; extract to e.g. `C:\lookalike_data\app_data`.
2. Point the app at it: `$env:LOOKALIKE_DATA_DIR="C:\lookalike_data\app_data"`
   (and `$env:HF_HOME="C:\lookalike_data\hf_cache"` to keep the model cache off OneDrive).

## Vector database
Option A, embedded (no server): nothing to start.
Option B, Qdrant server in Docker:
```
docker run -d --name qdrant -p 6333:6333 -v C:\lookalike_data\qdrant_storage:/qdrant/storage qdrant/qdrant
$env:LOOKALIKE_QDRANT_URL="http://localhost:6333"
```
Load the vectors (rerun any time; add `--recreate` to start from empty):
```
python -m app.tools.ingest
```

## Run
From the repo root, in an environment with torch + transformers + the `app` group:
```
python -m app --data-dir C:\lookalike_data\app_data --hf-home C:\lookalike_data\hf_cache
```
Open http://127.0.0.1:8000. API docs at /docs. Endpoints: `POST /search`
(`query`, optional `k`, `product_type`, `colour`), `GET /facets`, `GET /health`.
