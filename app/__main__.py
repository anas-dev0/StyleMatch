"""One-command launcher: `python -m app` (instead of the long uvicorn command).

    python -m app --data-dir C:\\lookalike_data\\app_data --hf-home C:\\lookalike_data\\hf_cache

Every option can also come from the environment (LOOKALIKE_DATA_DIR, HF_HOME) as before.
"""

import argparse
import os


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m app", description="Start the text-search app.")
    p.add_argument("--data-dir", help="folder made by export_assets (sets LOOKALIKE_DATA_DIR)")
    p.add_argument("--hf-home", help="Hugging Face cache folder (sets HF_HOME)")
    p.add_argument("--qdrant-url", help="use a Qdrant server instead of embedded storage")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def apply_env(args: argparse.Namespace, env=None) -> None:
    """Turn the command-line options into the environment variables the app reads."""
    env = os.environ if env is None else env
    for value, name in (
        (args.data_dir, "LOOKALIKE_DATA_DIR"),
        (args.hf_home, "HF_HOME"),
        (args.qdrant_url, "LOOKALIKE_QDRANT_URL"),
    ):
        if value:
            env[name] = value


def main(argv=None) -> None:
    args = parse_args(argv)
    apply_env(args)
    import uvicorn  # imported late so --help works without it

    uvicorn.run("app.backend.main:build_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
