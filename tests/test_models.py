from lookalike.embed.models import resolve_revision


def test_pinned_revision_is_used_without_any_lookup():
    def must_not_be_called(_):
        raise AssertionError("network lookup should not happen when a revision is pinned")

    assert resolve_revision("org/model", "abc123", get_sha=must_not_be_called) == "abc123"


def test_unpinned_revision_asks_the_hub_for_the_current_hash():
    assert resolve_revision("org/model", None, get_sha=lambda model_id: f"sha-of-{model_id}") == (
        "sha-of-org/model"
    )


def test_lookup_failure_gives_none_instead_of_crashing(capsys):
    def broken(_):
        raise ConnectionError("offline")

    assert resolve_revision("org/model", None, get_sha=broken) is None
    assert "WARNING" in capsys.readouterr().out
