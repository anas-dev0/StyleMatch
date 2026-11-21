# LookAlike: implementation plan

Visual look-alike search for fashion items. Given a product image, return the
most similar catalogue items. The project compares embedding models, fine-tunes
one with LoRA, serves it through Qdrant hybrid search, and evaluates everything
with an independent gold set.

Target audience: a reviewer for the "AI-Assisted New Item Forecasting: Finding
Look-Alike Products with Images and Attributes" end-of-studies offer, plus
general applied-ML interviews.

## 1. What makes it stand out

- A fine-tuned model (LoRA contrastive), not only pretrained parts glued together.
- A domain baseline (FashionCLIP) to beat, not only generic CLIP.
- An independent, TREC-style pooled gold set, so the model is not graded only on
  the labels it trained on.
- A held-out-product-type split, to test generalisation beyond training types.
- Industry tooling: Hugging Face + PEFT, MLflow, Qdrant (named vectors, sparse
  vectors, RRF fusion, payload filters), FastAPI, Docker Compose, CI.
- A scalability benchmark (build time, latency, recall vs HNSW settings).
- A small demand-transfer check showing why look-alikes matter for new items.

## 2. Scope guardrails

In scope: retrieval, fine-tuning, evaluation, serving, demand-transfer check.
Out of scope: a full forecasting system, user accounts, mobile apps, any
claim of production deployment. Keep every claim defensible.

## 3. Stack

| Area | Tool |
|---|---|
| Language / env | Python 3.11, `uv`, ruff, pytest, pre-commit |
| Deep learning | PyTorch, Hugging Face `transformers` (use `CLIPModel`), `peft` |
| Model zoo | ResNet-50 (ImageNet), CLIP ViT-B/16, DINOv2, FashionCLIP (domain baseline). Verify exact Hub names and licences |
| Vector DB | Qdrant (dense + sparse + payload filters, RRF fusion); FAISS flat as exact baseline |
| Experiment tracking | MLflow (local; log params, metrics, artifacts) |
| Serving | FastAPI, Streamlit demo, Docker Compose |
| Data processing | pandas / pyarrow. Optional: Databricks Free Edition (PySpark + Delta) for the embedding job |
| CI | GitHub Actions (ruff, pytest) |

LoRA note: use the Hugging Face CLIP implementation. Its attention layers are
`q_proj`, `k_proj`, `v_proj`, `out_proj`, which `peft` handles cleanly.
OpenCLIP's fused attention is harder to adapt.

## 4. Compute plan

- **Kaggle (2x T4)**: development, subset building (competition data is mounted
  at `/kaggle/input`, so no 30 GB download), small training runs. T4 has no
  bf16: use fp16 + GradScaler. Two experiments can run in parallel, one per GPU.
  Check the current weekly GPU quota.
- **University A40 (48 GB)**: final training runs, larger batches, bf16.
  Work in tmux, save checkpoints often, copy outputs back as a Kaggle dataset or
  with rsync.
- Code lives in GitHub. Kaggle notebooks do `git clone`, `pip install -e .`,
  then call CLI scripts. Save embeddings and adapters as Kaggle datasets.

## 5. Data

**Primary dataset:** H&M Personalized Fashion Recommendations (Kaggle). It has
images, attributes (`product_type_name`, `colour_group_name`,
`graphical_appearance_name`, `department_name`, `garment_group_name`, ...),
a text `detail_desc`, and transactions for the demand-transfer check.
**Fallback:** Fashion Product Images (small) if licence or size is a problem.
**Licence:** read the competition's data terms. Do not commit images. Publish
code, download scripts and metrics only unless terms clearly allow more.

**Subset:** about 20k articles with images, capped per `product_type_name` so no
type dominates.

**Splits (group-aware):**
- Split by `product_code` into train 70 / val 10 / test 20.
- Additionally hold out about 20% of product *types* entirely (never in train):
  the "unseen-type" test.
- Near-duplicate check: cluster items whose frozen-CLIP cosine similarity
  exceeds about 0.97 and keep each cluster in a single split.
- Test queries are test-split items; the gallery is the test split. For each
  query, exclude its own `product_code` from the results.

**Graded relevance from attributes (used for training positives and for
a first evaluation):**
- 3: same product type AND colour group AND graphical appearance
- 2: same product type AND (colour group OR graphical appearance)
- 1: same product type
- 0: otherwise

This is a proxy. The gold set (section 7) is the independent check.

## 6. Fine-tuning data preparation (how embedding training data works)

An embedding model is trained on a definition of "similar". The data is not
(input, answer); it is relationships between items.

Formats:
- **Pairs** (anchor, positive). Other items in the batch act as negatives
  (InfoNCE, in-batch negatives).
- **Triplets** (anchor, positive, negative), usually with *hard* negatives:
  items that look close but are not the same kind.
- **Labelled groups.** Items sharing a label are positives. This is the
  supervised-contrastive / multi-positive form and the one used here, since
  attributes give group labels for free.
- **Image-text pairs** (CLIP-style): an image and a caption built from attributes
  and `detail_desc`.

Rules to implement:
- **Class-aware batch sampler:** each batch holds several items from the same
  group, so positives exist in every batch.
- **False-negative handling:** with multi-positive loss, items in the same
  group are not treated as negatives of each other.
- **Hard-negative mining:** after a first model, mine nearest neighbours that
  have relevance 0-1 and add them to later epochs.
- **Augmentations:** random resized crop (scale 0.6-1.0), horizontal flip.
  **No hue or colour jitter**: colour is part of the label.
- Split by `product_code` (section 5) so test items never leak into training.

## 7. Labelling plan (three tiers)

1. **Free labels from attributes** (section 5). All training groups come from
   these. No AI labelling needed for bulk training.
2. **Gold evaluation set (AI-labelled, human-audited).** TREC-style pooling:
   - Pick about 150 test queries, stratified by product type.
   - For each, pool the top-10 results from every system (ResNet, CLIP, DINOv2,
     FashionCLIP, attributes, hybrid, later the LoRA model) and dedupe.
   - Render each query + candidates as one contact sheet image with numbered
     tiles.
   - Claude Code (reads images natively) rates each candidate 0-3 for
     visual similarity of garment type, shape, pattern and colour, and writes
     `gold_v1.jsonl` with the rubric and prompt stored in the repo.
   - Anas independently rates a random 15-20% sample. Report agreement
     (Cohen's kappa) between his labels and Claude's.
   - The gold set is evaluation only; it is never used for training. This also
     keeps the project clear of any restriction on training models with
     Claude outputs. Anas should read Anthropic's current terms himself.
3. **Optional bulk enrichment with an open VLM.** Run Qwen2.5-VL-7B (check its
   licence) on the A40 to extract fine-grained attributes as JSON (sleeve
   length, neckline, pattern, fit, style) that H&M lacks. Use them as extra
   payload filters and as additional training positives in one experiment.
   Claude audits about 200 outputs. Report the audit error rate.

README must state which labels were AI-generated, by which model, and how
they were checked.

## 8. Phases

Each phase ends with exit criteria and interview questions for Anas.

### Phase 0: setup (0.5 day)
Repo, `uv`, ruff, pytest, pre-commit, CI, MLflow, `LEARNING_LOG.md`,
`DECISIONS.md`. Licence check on the dataset.
- Exit: CI green on an empty package.
- Interview: Why group-aware splits? What does MLflow track here?

### Phase 1: data subset and splits (1 day)
Build the subset on Kaggle, clean attributes, create splits and the relevance
function, run the near-duplicate check.
- Exit: split files committed (ids only), leakage test passing, relevance unit
  tests passing.
- Interview: What leaks if you split by `article_id`? Why is relevance only a proxy?

### Phase 2: baselines and evaluation harness (1-1.5 days)
Extract and cache embeddings for ResNet-50, CLIP, DINOv2, FashionCLIP. Add
attribute-only and text-only baselines. Build the harness: Recall@k, Hit@1, MRR,
mAP@10, nDCG@10 with graded relevance, bootstrap confidence intervals.
- Exit: baseline table in MLflow and README; metrics tested on hand-computed
  fixtures.
- Interview: What is an embedding? Why cosine similarity? What does nDCG add
  over precision@k?

### Phase 3: Qdrant and hybrid search (1 day)
Collection with named vectors (image dense, text dense) and a sparse vector
for attributes/BM25; payload filters; hybrid query with RRF fusion. FAISS flat
as an exact baseline to measure HNSW recall loss.
- Exit: hybrid vs single-signal comparison; HNSW recall vs exact reported.
- Interview: Exact vs approximate search? What is RRF and why use it?

### Phase 4: gold set (1-2 days)
Pooling, contact sheets, Claude judgments, human audit, kappa. Re-evaluate all
baselines on the gold set.
- Exit: `gold_v1.jsonl`, rubric, audit report, baselines re-scored.
- Interview: Why pooling? Why is gold separate from training labels?

### Phase 5: LoRA fine-tuning (3-4 days)
Training code: HF `CLIPModel` + `peft` LoRA, multi-positive contrastive loss
with class-aware sampler, MLflow logging, resumable checkpoints. Start with a
small Kaggle run to debug, then final runs on the A40.
Starting point (tune from here): LoRA r=16, alpha=32, dropout 0.05 on q/k/v/out
projections of the vision tower, lr 1e-4, cosine schedule with warmup, weight
decay 0.01, batch 128-256 (gradient checkpointing if needed), 5-10 epochs,
bf16 on A40 / fp16 on T4.
- Exit: experiment matrix (section 9) complete, comparison with frozen CLIP and
  FashionCLIP on test, unseen-type test and gold set.
- Interview: What does LoRA change and why use it? How does the loss work?
  What would overfitting look like here?

### Phase 6: scalability and serving (1-2 days)
FastAPI `/search` (image upload or article id, optional filters), Streamlit
demo, Docker Compose (Qdrant + API + demo). Benchmarks: embedding throughput
(T4 vs A40), index build time, query latency p50/p95, recall@10 vs HNSW `ef`
at 10k / 100k / 1M vectors (label the larger sizes as synthetic).
- Exit: `docker compose up` works from a clean clone; benchmark table.
- Interview: How would this scale to a full retailer catalogue?

### Phase 7: demand-transfer check (1-2 days)
Temporal split on H&M transactions: items first sold in the last N weeks are
"new". For each, retrieve K look-alikes from earlier items and estimate first
4-8 weeks of units as a similarity-weighted average of the neighbours' first
weeks (aligned by weeks since launch). Compare with baselines: global average,
category average, attribute-only neighbours. Metrics: WAPE, bias, MAE.
Limitations to state: transactions are a sample, no stock or availability data,
so only relative comparisons are meaningful.
- Exit: table with real numbers and a short limitations paragraph.
- Interview: Why does a better similarity search help forecasting a new item?

### Phase 8: packaging (1 day)
README with architecture diagram and results, model card (data, intended use,
limitations, AI-labelling disclosure), 2-3 minute demo GIF, final
`LEARNING_LOG.md` review.
- Exit: a clean clone reproduces the main table from cached embeddings.

## 9. Experiment matrix

| ID | Change |
|---|---|
| B0-B4 | Baselines: ResNet-50, CLIP, DINOv2, FashionCLIP, attributes only |
| H1 | Hybrid (image + text + attributes) via RRF |
| E1 | LoRA, vision tower, attribute-group multi-positive loss |
| E2 | E1 + hard-negative mining |
| E3 | E1 with image-text contrastive (captions from attributes) |
| E4 | E1 + VLM-enriched attributes as extra positives |
| A1-A3 | Ablations: rank 4/8/16; with vs without class-aware sampler; with vs without crop augmentation |

Report every run, including ones that did not help.

## 10. Evaluation protocol

- Systems are compared on: test split, unseen-type split, and the gold set.
- Always report mean with bootstrap 95% confidence interval.
- A result counts as an improvement only if the interval separates from the
  baseline, or the text says it does not.
- Inspect the 20 worst queries per system and write the failure categories
  (colour confusion, pattern confusion, background effects, label noise).

## 11. Repo layout

```
lookalike/
  CLAUDE.md  PLAN.md  README.md  LEARNING_LOG.md  DECISIONS.md  MODEL_CARD.md
  pyproject.toml  uv.lock  Makefile  Dockerfile  docker-compose.yml
  configs/{data,model,train,eval}/
  src/lookalike/
    data/      subset, splits, dedupe, relevance
    embed/     models, extract
    index/     qdrant_store, faiss_baseline
    train/     lora_clip, losses, sampler, hard_negatives
    eval/      metrics, harness, pooling, gold
    labeling/  contact_sheets, vlm_enrich
    coldstart/ neighbours, estimate
    serve/     api
    app/       streamlit_app
  notebooks/   thin Kaggle wrappers
  tests/
  .github/workflows/ci.yml
  data/ artifacts/    (gitignored)
```

## 12. Risks

| Risk | Mitigation |
|---|---|
| Training and evaluating on the same attribute definition inflates scores | Gold set, unseen-type split, report both |
| Split leakage via colour variants and near-duplicates | Group split by `product_code`, near-duplicate clustering, test |
| LoRA does not beat FashionCLIP | Report honestly; analyse why |
| Dataset licence limits publishing | Publish code and metrics only until checked |
| Labelling cost and effort | Claude labels only the small gold set; bulk labels are free or from an open VLM |
| Scope creep into forecasting | Phase 7 stays a small, clearly labelled check |
| Not being able to explain it | Learning log, interview questions per phase, mock interview at the end |

## 13. Timeline (part-time, with Claude Code)

Core (Phases 0-3): 4-5 days. Gold set: 1-2. Fine-tuning: 3-4. Serving and
benchmarks: 1-2. Demand check: 1-2. Packaging: 1. Total about 12-16 working
days, plus time to understand everything. If time is short, cut in this order:
VLM enrichment (E4), Phase 7, Databricks, the 1M-vector benchmark. Never cut the
gold set or the ablation table.
