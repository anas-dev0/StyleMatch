"""Resumable text-embedding extraction with a CLIP-family text tower (DECISIONS.md D20).

Each item gets one caption built from configured columns (default: `detail_desc` only). The
caption is embedded with `get_text_features`, so the vector lives in the same space as the
image vectors of the same model. Output layout matches image extraction:
<out_root>/<name>/{embeddings.npy, ids.npy, meta.json}.

    python -m lookalike.embed.text --config configs/embed/text.yaml --model fashionclip_text_desc
    (add --limit 64 for a smoke test on the first 64 items)
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lookalike.config import load_config
from lookalike.embed.extract import run_extraction


def build_caption(row: pd.Series, fields: list[str], sep: str = ". ") -> str:
    """Join the non-missing, non-blank values of `fields` in order. Missing everything -> ''."""
    parts = []
    for field in fields:
        value = row.get(field)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return sep.join(parts)


def build_captions(df: pd.DataFrame, fields: list[str]) -> list[str]:
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise KeyError(f"caption fields not in the data: {missing}")
    return [build_caption(row, fields) for _, row in df.iterrows()]


class TextEncoder:
    """CLIP text tower. torch/transformers are imported here so the module imports without them."""

    max_length = 77  # CLIP's context length; longer captions are truncated

    def __init__(self, spec: dict, device: str | None = None):
        import torch
        import transformers
        from transformers import AutoTokenizer, CLIPModel

        from lookalike.embed.models import resolve_revision

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device == "cuda"  # fp16 on GPU; T4 has no bf16

        revision = resolve_revision(spec["hf_id"], spec.get("revision"))
        self.tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"], revision=revision)
        model = CLIPModel.from_pretrained(spec["hf_id"], revision=revision).eval().to(self.device)
        self.model = model.half() if self.half else model
        self.meta = {
            "hf_id": spec["hf_id"],
            "kind": "clip_text",
            "commit_hash": revision,
            "revision_pinned_in_config": bool(spec.get("revision")),
            "device": self.device,
            "fp16": self.half,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        """list of captions -> float32 array (len(texts), dim)."""
        torch = self.torch
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            out = self.model.get_text_features(**tokens)
        out = out if torch.is_tensor(out) else out.pooler_output
        return out.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache text embeddings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="a key under `models:` in the config")
    parser.add_argument("--limit", type=int, default=None, help="smoke test on the first N items")
    args = parser.parse_args()
    cfg = load_config(args.config)
    spec = cfg["models"][args.model]

    subset = pd.read_parquet(cfg["subset_path"])
    if args.limit:
        subset = subset.iloc[: args.limit]
    ids = subset["article_id"].to_numpy()
    caption_of = dict(zip(ids, build_captions(subset, spec["fields"]), strict=True))
    n_empty = sum(1 for c in caption_of.values() if not c)
    print(f"{len(ids)} captions from {spec['fields']}; {n_empty} are empty")

    out_root = cfg["out_root"] + ("_smoke" if args.limit else "")
    out_dir = Path(out_root) / args.model
    encoder = TextEncoder(spec)
    batch_size = cfg["batch_size"]

    def encode(batch_ids: np.ndarray) -> np.ndarray:
        texts = [caption_of[a] for a in batch_ids]
        parts = [
            encoder.encode(texts[s : s + batch_size]) for s in range(0, len(texts), batch_size)
        ]
        return np.concatenate(parts)

    emb = run_extraction(ids, encode, out_dir, cfg["shard_size"])
    np.save(out_dir / "embeddings.npy", emb)
    np.save(out_dir / "ids.npy", ids.astype(np.int64))
    meta = {
        **encoder.meta,
        "fields": spec["fields"],
        "n_items": int(len(ids)),
        "n_empty_captions": int(n_empty),
        "dim": int(emb.shape[1]),
        "date": date.today().isoformat(),
        "smoke_test": bool(args.limit),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{args.model}: {emb.shape} saved to {out_dir}")


if __name__ == "__main__":
    main()
