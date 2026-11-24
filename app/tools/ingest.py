"""Load the exported vectors and item fields into Qdrant.

    python -m app.tools.ingest --config configs/app/app.yaml            # add / refresh
    python -m app.tools.ingest --config configs/app/app.yaml --recreate # empty it first

Where the data goes is set in configs/app/app.yaml (`qdrant.url` for a Docker server, otherwise
embedded storage under data_dir). Embedded storage is locked by one process at a time: run this
before starting the API, not while it is running.
"""

import argparse

from app.backend.qdrant_store import ingest, make_client
from app.backend.search import read_export
from app.backend.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Load vectors and item fields into Qdrant.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--recreate", action="store_true", help="delete the collection first")
    args = parser.parse_args()
    settings = load_settings(args.config)
    data_dir = settings["data_dir"]
    qd = settings["qdrant"]

    embeddings, ids, items = read_export(data_dir)
    client, is_server = make_client(qd.get("url"), data_dir / "qdrant")
    total = ingest(
        client,
        qd["collection"],
        embeddings,
        ids,
        items,
        recreate=args.recreate,
        create_indexes=is_server,
    )
    where = qd.get("url") or str(data_dir / "qdrant")
    print(f"Collection '{qd['collection']}' at {where} now holds {total} items.")


if __name__ == "__main__":
    main()
