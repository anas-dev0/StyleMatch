from app.__main__ import apply_env, parse_args


def test_defaults_and_options_are_parsed():
    args = parse_args([])
    assert (args.host, args.port, args.data_dir) == ("127.0.0.1", 8000, None)
    args = parse_args(["--data-dir", "D:/x", "--port", "9000", "--qdrant-url", "http://q:6333"])
    assert (args.data_dir, args.port, args.qdrant_url) == ("D:/x", 9000, "http://q:6333")


def test_options_become_the_environment_variables_the_app_reads():
    env = {"HF_HOME": "kept"}
    apply_env(parse_args(["--data-dir", "D:/x", "--hf-home", "D:/hf"]), env)
    assert env == {"HF_HOME": "D:/hf", "LOOKALIKE_DATA_DIR": "D:/x"}


def test_unset_options_do_not_overwrite_existing_environment():
    env = {"LOOKALIKE_DATA_DIR": "from-env", "HF_HOME": "from-env"}
    apply_env(parse_args([]), env)
    assert env == {"LOOKALIKE_DATA_DIR": "from-env", "HF_HOME": "from-env"}
