# LookAlike: working rules for Claude Code

Project: visual look-alike search for fashion items (CLIP-family embeddings,
LoRA fine-tuning, Qdrant hybrid search, rigorous evaluation).
Read `PLAN.md` first. Work **one phase at a time** and stop at each phase's
exit criteria to report.

The owner (Anas) must be able to explain every line in an interview. These
rules exist for that reason.

1. **Teach as you build.** Before each new module, add 5-10 lines to
   `LEARNING_LOG.md`: what it does, why this approach, alternatives rejected.
   At the end of each phase, ask Anas the phase's interview questions. He
   answers in his own words in `LEARNING_LOG.md`. Never answer for him.
2. **Never invent numbers.** Every metric in the README or model card must come
   from a logged MLflow run (cite the run id). If it has not been run, write TBD.
3. **Reproducibility.** Fixed seeds, YAML configs in `configs/`, no hard-coded
   paths, pinned dependencies (`uv.lock`).
4. **Tests.** Every module in `src/` gets pytest tests. Metrics are tested on tiny
   hand-computed fixtures. CI (ruff + pytest) must stay green.
5. **Leakage guard.** Splits are by `product_code`, never by `article_id`
   (colour variants of one product share a `product_code`). A test must
   assert that no `product_code` appears in two splits. Run the near-duplicate
   check described in `PLAN.md` section 5.
6. **Data and secrets.** Never commit data, images, `kaggle.json`, tokens or API
   keys. `data/` and `artifacts/` are gitignored. Do not publish model weights
   or derived data until the dataset licence has been checked (see `PLAN.md`).
7. **Costs.** Ask before any paid API call, with an estimate first.
8. **Long jobs** are CLI scripts with resumable checkpoints and logging, e.g.
   `python -m lookalike.train.lora_clip --config configs/train/e1.yaml`.
   Notebooks are thin wrappers used only on Kaggle.
9. **Labelling honesty.** For every AI-generated label, record the labeler
   (model, date, prompt file). The gold evaluation set is **evaluation only**
   and must never be used for training.
10. **Decisions** go in `DECISIONS.md` (what, why, alternatives).
11. Small commits with clear messages. No giant "initial everything" commit.
