import numpy as np
import pytest

from lookalike.config import load_config
from lookalike.embed.extract import run_extraction, shard_bounds

IDS = np.arange(100, 105)  # 5 items; shard_size=2 -> shards of 2, 2, 1


def fake_encode(batch_ids):
    """Deterministic 'embedding': [id, 2*id]. Lets us check that order is preserved."""
    ids = np.asarray(batch_ids, dtype=np.float32)
    return np.stack([ids, 2 * ids], axis=1)


def test_shard_bounds_by_hand():
    assert shard_bounds(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert shard_bounds(4, 2) == [(0, 2), (2, 4)]
    assert shard_bounds(1, 2048) == [(0, 1)]


def test_result_matches_encoding_everything_at_once(tmp_path):
    out = run_extraction(IDS, fake_encode, tmp_path, shard_size=2, log=lambda m: None)
    np.testing.assert_array_equal(out, fake_encode(IDS))
    assert len(list(tmp_path.glob("shard_0000?.npy"))) == 3


def test_resume_after_crash_only_recomputes_missing_shards(tmp_path):
    calls = []

    def crashing(batch_ids):
        calls.append(list(batch_ids))
        if len(calls) == 3:  # the third shard "crashes"
            raise RuntimeError("session died")
        return fake_encode(batch_ids)

    with pytest.raises(RuntimeError):
        run_extraction(IDS, crashing, tmp_path, shard_size=2, log=lambda m: None)
    assert len(calls) == 3  # shards 0 and 1 saved, shard 2 failed

    resumed_calls = []

    def counting(batch_ids):
        resumed_calls.append(list(batch_ids))
        return fake_encode(batch_ids)

    out = run_extraction(IDS, counting, tmp_path, shard_size=2, log=lambda m: None)
    assert resumed_calls == [[104]]  # only the last shard was recomputed
    np.testing.assert_array_equal(out, fake_encode(IDS))


def test_leftover_temp_file_from_a_crash_is_ignored(tmp_path):
    np.save(tmp_path / "shard_00000.tmp.npy", np.zeros((2, 2), dtype=np.float32))  # junk
    out = run_extraction(IDS, fake_encode, tmp_path, shard_size=2, log=lambda m: None)
    np.testing.assert_array_equal(out, fake_encode(IDS))


def test_shard_with_wrong_row_count_is_recomputed(tmp_path):
    np.save(tmp_path / "shard_00000.npy", np.zeros((1, 2), dtype=np.float32))  # should have 2 rows
    out = run_extraction(IDS, fake_encode, tmp_path, shard_size=2, log=lambda m: None)
    np.testing.assert_array_equal(out, fake_encode(IDS))


def test_encoder_returning_wrong_number_of_rows_raises(tmp_path):
    with pytest.raises(ValueError):
        run_extraction(IDS, lambda b: fake_encode(b)[:-1], tmp_path, shard_size=2)


def test_empty_ids_raises(tmp_path):
    with pytest.raises(ValueError):
        run_extraction([], fake_encode, tmp_path)


def test_baselines_config_lists_the_four_models_with_required_keys():
    cfg = load_config("configs/embed/baselines.yaml")
    assert set(cfg["models"]) == {"resnet50", "clip", "dinov2", "fashionclip"}
    for spec in cfg["models"].values():
        assert {"hf_id", "kind"} <= set(spec)
        assert spec["kind"] in {"resnet", "clip", "dinov2"}
