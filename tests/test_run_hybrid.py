import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from lookalike.config import load_config
from lookalike.eval import run_hybrid
from lookalike.eval.harness import evaluate_embeddings

TYPES = ["Shirt", "Dress", "Shoe"]
COLOURS = ["Black", "White", "Red"]
PATTERNS = ["Solid", "Stripe"]


def make_catalogue(n_products: int = 30, seed: int = 0):
    """Products with 1-2 colour variants; embeddings and text loosely follow the attributes."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_products):
        t = TYPES[p % 3]
        for v in range(1 + p % 2):
            c, g = COLOURS[rng.integers(3)], PATTERNS[rng.integers(2)]
            rows.append(
                {
                    "article_id": 1000 + 10 * p + v,
                    "product_code": p,
                    "product_type_name": t,
                    "colour_group_name": c,
                    "graphical_appearance_name": g,
                    "detail_desc": f"{c.lower()} {t.lower()} {'striped' if g == 'Stripe' else ''}",
                }
            )
    subset = pd.DataFrame(rows)
    one_hot = pd.get_dummies(subset[["product_type_name", "colour_group_name"]]).to_numpy(float)
    image = one_hot + rng.normal(scale=0.3, size=one_hot.shape)
    text = one_hot + rng.normal(scale=0.5, size=one_hot.shape)
    codes = subset.product_code.unique()
    split_of = {c: ("test" if i % 3 else "train") for i, c in enumerate(codes)}
    splits = subset[["article_id", "product_code"]].assign(split=subset.product_code.map(split_of))
    return subset, splits, image, text


def write_inputs(tmp_path):
    subset, splits, image, text = make_catalogue()
    subset.to_parquet(tmp_path / "subset.parquet")
    splits.to_csv(tmp_path / "splits.csv", index=False)
    for name, emb in (("fashionclip", image), ("fashionclip_text_desc", text)):
        d = tmp_path / "emb" / name
        d.mkdir(parents=True)
        order = np.random.default_rng(1).permutation(len(subset))  # file order != subset order
        np.save(d / "embeddings.npy", emb[order].astype(np.float32))
        np.save(d / "ids.npy", subset.article_id.to_numpy()[order])
    return subset, splits, image


def small_cfg(tmp_path) -> dict:
    cfg = load_config("configs/eval/hybrid.yaml")
    cfg.update(
        subset_path=str(tmp_path / "subset.parquet"),
        splits_path=str(tmp_path / "splits.csv"),
        embeddings_root=str(tmp_path / "emb"),
        results_path=str(tmp_path / "out" / "hybrid.csv"),
        paired_path=str(tmp_path / "out" / "paired.csv"),
        eval_splits=["test"],
        candidates=20,
        n_boot=50,
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )
    return cfg


def test_image_only_system_equals_the_phase2_harness(tmp_path):
    subset, splits, image = write_inputs(tmp_path)
    cfg = small_cfg(tmp_path)
    cfg["systems"] = {"image": ["image"]}
    items = run_hybrid.split_items_with_text(subset, splits, "test", ["detail_desc"])
    got = run_hybrid.evaluate_split(items, cfg)["image"]
    rows = subset.set_index("article_id").index.get_indexer(items.article_id)
    expected = evaluate_embeddings(
        image[rows], image[rows], items, items, ks=(1, 5, 10), ndcg_k=10, map_k=10, threshold=2
    )
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name], equal_nan=True, rtol=1e-5)


def test_cli_writes_results_and_paired_tables(tmp_path, monkeypatch):
    write_inputs(tmp_path)
    cfg = small_cfg(tmp_path)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_hybrid", "--config", str(path)])
    run_hybrid.main()

    results = pd.read_csv(tmp_path / "out" / "hybrid.csv")
    assert set(results.system) == set(cfg["systems"])
    assert results.loc[results.system == "image+attr_bm25", "circular"].all()
    assert not results.loc[results.system == "image+text+bm25", "circular"].any()
    ndcg = results[results.metric == "ndcg@10"].set_index("system")["mean"]
    assert ndcg.between(0, 1).all()
    paired = pd.read_csv(tmp_path / "out" / "paired.csv")
    assert set(paired.model) == set(cfg["systems"]) - {"image"}
    assert (paired.vs == "image").all()


def test_hybrid_config_is_consistent():
    cfg = load_config("configs/eval/hybrid.yaml")
    assert cfg["reference_system"] in cfg["systems"]
    for signals in cfg["systems"].values():
        assert set(signals) <= set(cfg["signals"])
    assert (
        cfg["relevance_threshold"]
        == load_config("configs/eval/baselines.yaml")["relevance_threshold"]
    )
    assert cfg["candidates"] >= max(*cfg["ks"], cfg["ndcg_k"], cfg["map_k"])


def test_unknown_signal_kind_is_rejected():
    with pytest.raises(ValueError):
        run_hybrid.signal_ranking({"kind": "magic"}, pd.DataFrame(), np.zeros((0, 0), bool), 5, {})
