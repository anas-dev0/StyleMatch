"""Score every baseline embedding on the held-out splits.

    python -m lookalike.eval.run_baselines --config configs/eval/baselines.yaml

For each model and each evaluation split: queries = gallery = the items of that split, each
query's own product is removed (D11). Results go to a CSV and, if MLflow is installed, to MLflow.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config
from lookalike.data.relevance import APPEARANCE_COL, COLOUR_COL, TYPE_COL
from lookalike.eval import metrics as M
from lookalike.eval.harness import evaluate_embeddings, summarise

ATTRIBUTE_COLS = [TYPE_COL, COLOUR_COL, APPEARANCE_COL]


def split_items(subset: pd.DataFrame, splits: pd.DataFrame, split: str) -> pd.DataFrame:
    """Items of one split with the attribute columns needed for relevance."""
    ids = splits.loc[splits["split"] == split, ["article_id", "product_code"]]
    attrs = subset[["article_id", *ATTRIBUTE_COLS]]
    items = ids.merge(attrs, on="article_id", how="left", validate="one_to_one")
    return items.reset_index(drop=True)


def align_embeddings(emb: np.ndarray, emb_ids: np.ndarray, article_ids: np.ndarray) -> np.ndarray:
    """Rows of `emb` for `article_ids`, in that order. Raises if an id has no embedding."""
    if len(emb) != len(emb_ids):
        raise ValueError(f"{len(emb)} embeddings but {len(emb_ids)} ids")
    row_of = pd.Series(np.arange(len(emb_ids)), index=emb_ids)
    missing = pd.Index(article_ids).difference(row_of.index)
    if len(missing):
        raise KeyError(f"{len(missing)} article ids have no embedding, e.g. {list(missing[:3])}")
    return emb[row_of.loc[article_ids].to_numpy()]


def random_embeddings(n: int, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Pure noise: the floor every real model must beat."""
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def attribute_embeddings(items: pd.DataFrame) -> np.ndarray:
    """One-hot of type/colour/appearance. Circular with the relevance proxy: reference only."""
    onehot = pd.get_dummies(items[ATTRIBUTE_COLS].astype("string"), dtype=np.float32)
    return onehot.to_numpy()


def evaluate_split(emb: np.ndarray, items: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Per-query metrics with queries = gallery = `items` (`emb` rows aligned to `items`)."""
    return evaluate_embeddings(
        emb,
        emb,
        items,
        items,
        ks=tuple(cfg["ks"]),
        ndcg_k=cfg["ndcg_k"],
        map_k=cfg["map_k"],
        threshold=cfg["relevance_threshold"],
    )


def paired_table(
    per_query: dict[str, dict[str, np.ndarray]], reference: str, metric: str, cfg: dict
) -> pd.DataFrame:
    """Mean per-query difference (model - reference) on one metric, with a bootstrap interval."""
    rows = []
    for name, values in per_query.items():
        if name == reference:
            continue
        mean, lo, hi = M.paired_bootstrap_diff(
            values[metric], per_query[reference][metric], n_boot=cfg["n_boot"], seed=cfg["seed"]
        )
        rows.append({"model": name, "vs": reference, "metric": metric, "diff": mean,
                     "ci_low": lo, "ci_high": hi})  # fmt: skip
    return pd.DataFrame(rows)


def log_to_mlflow(
    summary: pd.DataFrame, model: str, split: str, meta: dict, cfg: dict
) -> str | None:
    """One MLflow run per (model, split). Returns the run id, or None if MLflow is missing."""
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run(run_name=f"{model}-{split}") as run:
        mlflow.log_params(
            {
                "model": model,
                "split": split,
                "seed": cfg["seed"],
                "n_boot": cfg["n_boot"],
                "relevance_threshold": cfg["relevance_threshold"],
                "commit_hash": meta.get("commit_hash"),
                "hf_id": meta.get("hf_id"),
            }
        )
        for row in summary.itertuples():
            mlflow.log_metric(row.metric, row.mean)
            mlflow.log_metric(f"{row.metric}_ci_low", row.ci_low)
            mlflow.log_metric(f"{row.metric}_ci_high", row.ci_high)
        return run.info.run_id


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="Evaluate baseline embeddings.")
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    subset = pd.read_parquet(cfg["subset_path"])
    splits = pd.read_csv(cfg["splits_path"])
    root = Path(cfg["embeddings_root"])

    frames, run_ids = [], {}
    for split in cfg["eval_splits"]:
        items = split_items(subset, splits, split)
        print(f"\n== split '{split}': {len(items)} items, {items.product_code.nunique()} products")
        per_query: dict[str, dict[str, np.ndarray]] = {}
        metas: dict[str, dict] = {}
        systems = {
            "random": random_embeddings(len(items), seed=cfg["seed"]),
            "attributes": attribute_embeddings(items),
        }
        for model in cfg["models"]:
            emb = np.load(root / model / "embeddings.npy")
            ids = np.load(root / model / "ids.npy")
            systems[model] = align_embeddings(emb, ids, items["article_id"].to_numpy())
            metas[model] = json.loads((root / model / "meta.json").read_text(encoding="utf-8"))
        for name, emb in systems.items():
            per_query[name] = evaluate_split(emb, items, cfg)
            summary = summarise(per_query[name], n_boot=cfg["n_boot"], seed=cfg["seed"])
            summary.insert(0, "split", split)
            summary.insert(0, "model", name)
            frames.append(summary)
            run_ids[(name, split)] = log_to_mlflow(summary, name, split, metas.get(name, {}), cfg)

        metric = f"ndcg@{cfg['ndcg_k']}"
        print(f"\nPaired difference on {metric} vs {cfg['reference_model']} ({split}):")
        print(paired_table(per_query, cfg["reference_model"], metric, cfg).round(4).to_string())

    results = pd.concat(frames, ignore_index=True)
    keys = zip(results["model"], results["split"], strict=True)
    results["mlflow_run_id"] = [run_ids[key] for key in keys]
    out = Path(cfg["results_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    for split in cfg["eval_splits"]:
        table = results[results.split == split].pivot(
            index="model", columns="metric", values="mean"
        )
        print(f"\nMeans on '{split}':\n{table.round(4).to_string()}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
