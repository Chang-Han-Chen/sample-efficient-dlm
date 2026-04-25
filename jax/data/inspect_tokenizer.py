"""Inspect a trained ClimbMix tokenizer for basic sanity checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def token_display(tokenizer: Tokenizer, token_id: int) -> dict[str, object]:
    token = tokenizer.id_to_token(token_id)
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    return {
        "id": token_id,
        "token": token,
        "decoded": decoded,
        "utf8_bytes": len(decoded.encode("utf-8")),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer_json", type=Path)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--token-bytes", type=Path, default=None)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    vocab_size = tokenizer.get_vocab_size()
    metadata = {}
    if args.metadata is not None and args.metadata.exists():
        metadata = json.loads(args.metadata.read_text())

    print(json.dumps({"vocab_size": vocab_size, "metadata_vocab_size": metadata.get("vocab_size")}))
    if metadata:
        special_tokens = metadata.get("special_tokens", [])
        special_rows = [
            {
                "token": token,
                "id": tokenizer.token_to_id(token),
                "decoded": tokenizer.decode([tokenizer.token_to_id(token)], skip_special_tokens=False),
            }
            for token in special_tokens
        ]
        print(json.dumps({"special_tokens": special_rows}))
        print(
            json.dumps(
                {
                    "diffusion_vocab_size": metadata.get("diffusion_vocab_size"),
                    "diffusion_mask_token_id": metadata.get("diffusion_mask_token_id"),
                    "bos_token": metadata.get("bos_token"),
                    "bos_token_id": metadata.get("bos_token_id"),
                }
            )
        )

    if args.token_bytes is not None and args.token_bytes.exists():
        token_bytes = np.load(args.token_bytes)
        print(
            json.dumps(
                {
                    "token_bytes_shape": tuple(int(x) for x in token_bytes.shape),
                    "token_bytes_min": int(token_bytes.min()),
                    "token_bytes_max": int(token_bytes.max()),
                    "token_bytes_mean": float(token_bytes.mean()),
                    "zero_byte_tokens": int(np.sum(token_bytes == 0)),
                }
            )
        )

    rng = random.Random(args.seed)
    head_ids = list(range(min(16, vocab_size)))
    random_ids = [rng.randrange(vocab_size) for _ in range(args.samples)]
    tail_ids = list(range(max(0, vocab_size - 8), vocab_size))
    rows = [token_display(tokenizer, i) for i in head_ids + random_ids + tail_ids]
    print(json.dumps({"sample_vocab": rows}, ensure_ascii=False))

    examples = [
        "Hello world! Numbers: 123. Unicode: ni hao.",
        "def f(x): return x * 2\nprint(f(21))",
        "Language diffusion models mask tokens differently from AR models.",
    ]
    roundtrips = []
    for text in examples:
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids)
        roundtrips.append(
            {
                "text": text,
                "num_tokens": len(ids),
                "first_ids": ids[:24],
                "decoded_matches": decoded == text,
                "decoded": decoded,
            }
        )
    print(json.dumps({"roundtrips": roundtrips}, ensure_ascii=False))


if __name__ == "__main__":
    main()
