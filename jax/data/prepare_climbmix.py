"""Prepare Nvidia ClimbMix shards for JAX training.

This script intentionally supports small smoke preps and full data preps with
the same metadata layout. It downloads parquet shards, trains a 32k byte-level
BPE tokenizer, tokenizes each shard into flat `.npy` token streams, and writes
metadata needed to reproduce the run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pyarrow.parquet as pq
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
BASE_VOCAB_SIZE = 32768
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = SPECIAL_TOKENS[0]
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| """
    r"""?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def shard_name(i: int) -> str:
    return f"shard_{i:05d}.parquet"


def default_data_root() -> Path:
    env_root = os.environ.get("SAMPLE_EFFICIENT_DLM_DATA")
    if env_root:
        return Path(env_root) / "climbmix"
    return Path(__file__).resolve().parents[2] / "data" / "climbmix"


def download_file(url: str, dst: Path, *, chunk_size: int = 8 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for attempt in range(1, 6):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            tmp.rename(dst)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def download_shards(raw_dir: Path, train_shards: list[int], val_shard: int) -> list[Path]:
    shard_ids = sorted(set(train_shards + [val_shard]))
    paths = []
    for shard_id in shard_ids:
        name = shard_name(shard_id)
        path = raw_dir / name
        print(f"download {name}")
        download_file(f"{BASE_URL}/{name}", path)
        paths.append(path)
    return paths


def iter_texts(parquet_paths: Iterable[Path], *, max_chars: int | None = None, max_docs: int | None = None):
    chars = 0
    docs = 0
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            for text in rg.column("text").to_pylist():
                yield text
                chars += len(text)
                docs += 1
                if max_chars is not None and chars >= max_chars:
                    return
                if max_docs is not None and docs >= max_docs:
                    return


def train_tokenizer(
    train_paths: list[Path],
    tokenizer_dir: Path,
    *,
    vocab_size: int,
    max_chars: int | None,
    max_docs: int | None,
) -> Tokenizer:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    metadata_json = tokenizer_dir / "metadata.json"
    token_bytes_path = tokenizer_dir / "token_bytes.npy"
    if tokenizer_json.exists() and metadata_json.exists() and token_bytes_path.exists():
        return Tokenizer.from_file(str(tokenizer_json))

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    print(f"train tokenizer vocab_size={vocab_size}")
    tokenizer.train_from_iterator(
        iter_texts(train_paths, max_chars=max_chars, max_docs=max_docs),
        trainer=trainer,
        length=max_docs,
    )
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(f"Tokenizer vocab is {tokenizer.get_vocab_size()}, expected {vocab_size}")
    tokenizer.save(str(tokenizer_json))

    vocab = tokenizer.get_vocab()
    id_to_token = {idx: token for token, idx in vocab.items()}
    token_bytes = np.zeros((vocab_size,), dtype=np.int32)
    special_ids = {vocab[t] for t in SPECIAL_TOKENS}
    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_bytes[token_id] = 0
        else:
            text = tokenizer.decode([token_id], skip_special_tokens=False)
            token_bytes[token_id] = len(text.encode("utf-8"))
    np.save(token_bytes_path, token_bytes)

    test = "Hello world! Numbers: 123. Unicode: 你好"
    decoded = tokenizer.decode(tokenizer.encode(test).ids)
    if decoded != test:
        raise RuntimeError(f"Tokenizer roundtrip failed: {decoded!r}")

    metadata = {
        "tokenizer": "tokenizers.ByteLevelBPE",
        "vocab_size": vocab_size,
        "diffusion_vocab_size": vocab_size + 1,
        "diffusion_mask_token_id": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "bos_token": BOS_TOKEN,
        "bos_token_id": vocab[BOS_TOKEN],
        "split_pattern_reference": SPLIT_PATTERN,
        "max_chars": max_chars,
        "max_docs": max_docs,
        "id_to_token_sample": {str(i): id_to_token[i] for i in range(min(16, vocab_size))},
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))
    return tokenizer


def tokenize_shard(path: Path, tokenizer: Tokenizer, out_path: Path, *, batch_size: int = 512) -> dict[str, int]:
    if out_path.exists():
        arr = np.load(out_path, mmap_mode="r")
        return {"tokens": int(arr.shape[0]), "docs": -1}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    token_chunks: list[np.ndarray] = []
    total = 0
    docs = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal total, docs, batch
        if not batch:
            return
        encoded = tokenizer.encode_batch(batch)
        for enc in encoded:
            ids = np.asarray([bos_id] + enc.ids, dtype=np.uint16)
            token_chunks.append(ids)
            total += int(ids.shape[0])
            docs += 1
        batch = []

    pf = pq.ParquetFile(path)
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=["text"])
        for text in rg.column("text").to_pylist():
            batch.append(text)
            if len(batch) >= batch_size:
                flush()
    flush()

    tokens = np.concatenate(token_chunks) if token_chunks else np.empty((0,), dtype=np.uint16)
    np.save(out_path, tokens)
    return {"tokens": int(tokens.shape[0]), "docs": docs}


def write_manifest(root: Path, metadata: dict) -> None:
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=default_data_root())
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--val-shard", type=int, default=VAL_SHARD)
    p.add_argument("--vocab-size", type=int, default=BASE_VOCAB_SIZE)
    p.add_argument("--max-tokenizer-chars", type=int, default=200_000_000)
    p.add_argument("--max-tokenizer-docs", type=int, default=None)
    p.add_argument("--tokenizer-batch-size", type=int, default=512)
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-tokenize", action="store_true")
    args = p.parse_args()

    root = args.output_dir
    raw_dir = root / "raw"
    tokenizer_dir = root / "tokenizer"
    token_dir = root / "tokens"
    train_ids = list(range(args.num_shards))

    if not args.skip_download:
        download_shards(raw_dir, train_ids, args.val_shard)
    if args.download_only:
        return

    train_paths = [raw_dir / shard_name(i) for i in train_ids]
    val_path = raw_dir / shard_name(args.val_shard)
    missing = [p for p in train_paths + [val_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing parquet shards: {missing}")

    tokenizer = train_tokenizer(
        train_paths,
        tokenizer_dir,
        vocab_size=args.vocab_size,
        max_chars=args.max_tokenizer_chars,
        max_docs=args.max_tokenizer_docs,
    )
    if args.skip_tokenize:
        return

    token_stats = {"train": {}, "val": {}}
    for shard_id, path in zip(train_ids, train_paths, strict=True):
        out = token_dir / "train" / f"shard_{shard_id:05d}.npy"
        print(f"tokenize train {path.name} -> {out}")
        token_stats["train"][path.name] = tokenize_shard(
            path,
            tokenizer,
            out,
            batch_size=args.tokenizer_batch_size,
        )
    val_out = token_dir / "val" / f"shard_{args.val_shard:05d}.npy"
    print(f"tokenize val {val_path.name} -> {val_out}")
    token_stats["val"][val_path.name] = tokenize_shard(
        val_path,
        tokenizer,
        val_out,
        batch_size=args.tokenizer_batch_size,
    )

    write_manifest(
        root,
        {
            "dataset": "karpathy/climbmix-400b-shuffle",
            "base_url": BASE_URL,
            "train_shards": [shard_name(i) for i in train_ids],
            "val_shard": shard_name(args.val_shard),
            "vocab_size": args.vocab_size,
            "diffusion_vocab_size": args.vocab_size + 1,
            "diffusion_mask_token_id": args.vocab_size,
            "tokenizer_dir": str(tokenizer_dir),
            "train_tokens": str(token_dir / "train"),
            "val_tokens": str(token_dir / "val"),
            "token_stats": token_stats,
        },
    )
    print(f"wrote {root / 'metadata.json'}")


if __name__ == "__main__":
    main()
