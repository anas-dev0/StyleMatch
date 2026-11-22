# LookAlike

Visual look-alike search for fashion items (CLIP-family embeddings, LoRA fine-tuning, Qdrant
hybrid search, rigorous evaluation) on the H&M Personalized Fashion Recommendations images.
Work in progress: see `PLAN.md`. Data licence: non-commercial / academic only, so no images,
embeddings or weights are published here.

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
