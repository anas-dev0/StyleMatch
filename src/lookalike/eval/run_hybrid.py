"""Hybrid vs single-signal comparison on the held-out splits (DECISIONS.md D20-D23).

    python -m lookalike.eval.run_hybrid --config configs/eval/hybrid.yaml

Per split, queries = gallery = the split's items, with the own product excluded (D11). Each
signal (dense image, dense text, BM25 text) is ranked exactly over the split. Systems with
several signals are fused with RRF. Every system is scored with the Phase 2 metrics, bootstrap
CIs and a paired bootstrap against the reference system. Signals marked `circular` reuse the
attributes the relevance proxy is built from; their systems are flagged in the output.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config
from lookalike.embed.text import build_captions
from lookalike.eval.fusion import rank_by_scores, rrf_fuse
from lookalike.eval.harness import same_product_mask, summarise
from lookalike.eval.rankings import evaluate_rankings
from lookalike.eval.run_baselines import align_embeddings, log_to_mlflow, paired_table, split_items
from lookalike.index.sparse import BM25, tokenize


def split_items_with_text(
    subset: pd.DataFrame, splits: pd.DataFrame, split: str, fields
) -> pd.DataFrame:
    """`run_baselines.split_items` plus the text columns some signals need."""
    items = split_items(subset, splits, split)
    extra = [f for f in fields if f not in items.columns]
    if extra:
        items = items.merge(subset[["article_id", *extra]], on="article_id", how="left")
    return items


def _cosine(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x @ x.T


def signal_ranking(
    spec: dict, items: pd.DataFrame, exclude: np.ndarray, candidates: int, cfg: dict
) -> np.ndarray:
    """(n_items, candidates) exact ranking of the split for one signal, best first."""
    if spec["kind"] == "dense":
        root = Path(cfg["embeddings_root"]) / spec["model"]
        emb = align_embeddings(
            np.load(root / "embeddings.npy"),
            np.load(root / "ids.npy"),
            items["article_id"].to_numpy(),
        )
        return rank_by_scores(_cosine(emb), exclude, candidates)
    if spec["kind"] == "bm25":
        docs = [tokenize(c) for c in build_captions(items, spec["fields"])]
        bm25 = BM25(cfg["bm25_k1"], cfg["bm25_b"]).fit(docs)
        return rank_by_scores(bm25.score_matrix(docs, docs), exclude, candidates, min_score=0.0)
    raise ValueError(f"unknown signal kind: {spec['kind']}")


def system_ranking(signals: list[str], rankings: dict[str, np.ndarray], width: int, rrf_k: int):
    """One signal: its own list. Several: RRF over their candidate lists. Padded to `width`."""
    if len(signals) == 1:
        out = rankings[signals[0]][:, :width]
    else:
        out = rrf_fuse([rankings[s] for s in signals], k=rrf_k, top_k=width)
    if out.shape[1] < width:  # a split smaller than `width` items
        out = np.pad(out, ((0, 0), (0, width - out.shape[1])), constant_values=-1)
    return out


def evaluate_split(items: pd.DataFrame, cfg: dict) -> dict[str, dict[str, np.ndarray]]:
    """Per-query metrics of every configured system on one split."""
    exclude = same_product_mask(items, items)
    needed = {s for signals in cfg["systems"].values() for s in signals}
    rankings = {
        s: signal_ranking(cfg["signals"][s], items, exclude, cfg["candidates"], cfg) for s in needed
    }
    width = max(*cfg["ks"], cfg["ndcg_k"], cfg["map_k"])
    out = {}
    for name, signals in cfg["systems"].items():
        ranked = system_ranking(signals, rankings, width, cfg["rrf_k"])
        out[name] = evaluate_rankings(
            ranked,
            items,
            items,
            ks=tuple(cfg["ks"]),
            ndcg_k=cfg["ndcg_k"],
            map_k=cfg["map_k"],
            threshold=cfg["relevance_threshold"],
        )
    return out


def is_circular(signals: list[str], cfg: dict) -> bool:
    return any(cfg["signals"][s].get("circular", False) for s in signals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid and single-signal search.")
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    subset = pd.read_parquet(cfg["subset_path"])
    splits = pd.read_csv(cfg["splits_path"])
    fields = {f for spec in cfg["signals"].values() for f in spec.get("fields", [])}

    frames, paired = [], []
    metric = f"ndcg@{cfg['ndcg_k']}"
    for split in cfg["eval_splits"]:
        items = split_items_with_text(subset, splits, split, sorted(fields))
        print(f"\n== split '{split}': {len(items)} items")
        per_query = evaluate_split(items, cfg)
        for name, values in per_query.items():
            signals = cfg["systems"][name]
            summary = summarise(values, n_boot=cfg["n_boot"], seed=cfg["seed"])
            summary.insert(0, "split", split)
            summary.insert(0, "system", name)
            summary["signals"] = "+".join(signals)
            summary["circular"] = is_circular(signals, cfg)
            meta = {"hf_id": ",".join(cfg["signals"][s].get("model", "bm25") for s in signals)}
            summary["mlflow_run_id"] = log_to_mlflow(
                summary.rename(columns={"system": "model"}), name, split, meta, cfg
            )
            frames.append(summary)
        table = paired_table(per_query, cfg["reference_system"], metric, cfg)
        table.insert(0, "split", split)
        paired.append(table)
        print(f"Paired {metric} difference vs {cfg['reference_system']}:")
        print(table.round(4).to_string(index=False))

    results = pd.concat(frames, ignore_index=True)
    for key, df in (("results_path", results), ("paired_path", pd.concat(paired))):
        out = Path(cfg[key])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out}")
    for split in cfg["eval_splits"]:
        view = results[
            (results.split == split) & (results.metric.isin([metric, "hit@1", "hit@10"]))
        ]
        print(
            f"\n'{split}':\n{view.pivot(index='system', columns='metric', values='mean').round(4)}"
        )


if __name__ == "__main__":
    main()
