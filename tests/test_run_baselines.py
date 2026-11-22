import numpy as np
import pandas as pd
import pytest

from lookalike.config import load_config
from lookalike.eval.run_baselines import (
    align_embeddings,
    attribute_embeddings,
    evaluate_split,
    log_to_mlflow,
    random_embeddings,
    split_items,
)

CFG = {
    "ks": [1, 2],
    "ndcg_k": 3,
    "map_k": 3,
    "relevance_threshold": 2,
    "seed": 0,
    "n_boot": 50,
}

# 6 items, 6 different products. Types: shirts (1-3) and shoes (4-6).
SUBSET = pd.DataFrame(
    {
        "article_id": [11, 12, 13, 14, 15, 16],
        "product_code": [1, 2, 3, 4, 5, 6],
        "product_type_name": ["Shirt"] * 3 + ["Shoe"] * 3,
        "colour_group_name": ["Black", "Black", "White", "Black", "White", "White"],
        "graphical_appearance_name": ["Solid", "Stripe", "Solid", "Solid", "Solid", "Stripe"],
        "detail_desc": list("abcdef"),  # extra column that must be ignored
    }
)
SPLITS = pd.DataFrame(
    {
        "article_id": [11, 12, 13, 14, 15, 16],
        "product_code": [1, 2, 3, 4, 5, 6],
        "split": ["train", "test", "test", "test", "test", "val"],
    }
)


def test_split_items_selects_only_that_split_and_keeps_attributes():
    items = split_items(SUBSET, SPLITS, "test")
    assert list(items["article_id"]) == [12, 13, 14, 15]
    assert list(items["product_type_name"]) == ["Shirt", "Shirt", "Shoe", "Shoe"]
    assert "detail_desc" not in items.columns


def test_align_embeddings_follows_requested_order_not_file_order():
    emb = np.array([[1.0, 0], [2, 0], [3, 0]])
    ids = np.array([30, 10, 20])  # file order differs from the order we ask for
    out = align_embeddings(emb, ids, np.array([10, 20, 30]))
    np.testing.assert_array_equal(out[:, 0], [2.0, 3.0, 1.0])


def test_align_embeddings_raises_on_missing_id_and_on_length_mismatch():
    emb = np.zeros((2, 3))
    with pytest.raises(KeyError):
        align_embeddings(emb, np.array([1, 2]), np.array([1, 99]))
    with pytest.raises(ValueError):
        align_embeddings(emb, np.array([1, 2, 3]), np.array([1]))


def test_attribute_embeddings_are_equal_exactly_when_all_attributes_match():
    items = split_items(SUBSET, SPLITS, "test")
    e = attribute_embeddings(items)
    items2 = pd.concat([items, items.iloc[[0]]], ignore_index=True)  # duplicate attributes
    e2 = attribute_embeddings(items2)
    np.testing.assert_array_equal(e2[0], e2[-1])
    assert not np.array_equal(e[0], e[1])  # same type but different appearance


def test_attribute_baseline_beats_random_on_the_proxy_it_is_built_from():
    items = split_items(SUBSET, SPLITS, "test")
    attr = evaluate_split(attribute_embeddings(items), items, CFG)["ndcg@3"]
    rand_scores = [
        np.nanmean(evaluate_split(random_embeddings(len(items), seed=s), items, CFG)["ndcg@3"])
        for s in range(20)
    ]
    assert np.nanmean(attr) > np.mean(rand_scores)


def test_random_embeddings_are_reproducible_for_a_seed():
    np.testing.assert_array_equal(random_embeddings(5, seed=1), random_embeddings(5, seed=1))
    assert not np.array_equal(random_embeddings(5, seed=1), random_embeddings(5, seed=2))


def test_eval_config_has_the_keys_the_script_reads():
    cfg = load_config("configs/eval/baselines.yaml")
    assert cfg["reference_model"] in cfg["models"]
    assert {"seed", "n_boot", "ks", "ndcg_k", "map_k", "relevance_threshold"} <= set(cfg)
    assert set(cfg["eval_splits"]) <= {"test", "test_unseen", "val"}


def test_mlflow_logging_returns_a_run_id_and_stores_the_metric(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    summary = pd.DataFrame(
        {"metric": ["ndcg@10"], "mean": [0.5], "ci_low": [0.4], "ci_high": [0.6], "n_queries": [9]}
    )
    cfg = {
        **CFG,
        "mlflow_tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "mlflow_experiment": "t",
    }
    run_id = log_to_mlflow(summary, "clip", "test", {"commit_hash": "abc"}, cfg)
    assert run_id
    run = mlflow.get_run(run_id)
    assert run.data.metrics["ndcg_at_10"] == 0.5
    assert run.data.metrics["ndcg_at_10_ci_low"] == 0.4
    assert run.data.params["commit_hash"] == "abc"
