import argparse
import struct
from pathlib import Path
import multiprocessing as mp

import numpy as np
import psutil
import pyarrow.parquet as pq
import pandas as pd
from tokenizers import ByteLevelBPETokenizer, Tokenizer


# =========================
# Memory-aware config
# =========================


def memory_config():
    avail = psutil.virtual_memory().available
    budget = int(avail * 0.25)  # use 25% of free RAM

    chunk_bytes = min(budget // 4, 32 * 1024 * 1024)
    batch_limit = min(max(chunk_bytes // (512 * 1024), 8), 64)

    return max(chunk_bytes, 4 * 1024 * 1024), batch_limit


# =========================
# Streaming readers
# =========================


def stream_text(path, target_bytes):
    ext = path.suffix.lower()
    buffer, size = [], 0

    if ext == ".parquet":
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["text"])
            col = table["text"]
            for i in range(len(col)):
                t = col[i].as_py()
                if not t:
                    continue
                buffer.append(t)
                size += len(t)
                if size >= target_bytes:
                    yield buffer
                    buffer, size = [], 0

    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                buffer.append(line)
                size += len(line)
                if size >= target_bytes:
                    yield buffer
                    buffer, size = [], 0

    elif ext == ".csv":
        for df in pd.read_csv(path, chunksize=1000):
            rows = (
                df.select_dtypes(include=["object"])
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .tolist()
            )
            for r in rows:
                if not r:
                    continue
                buffer.append(r)
                size += len(r)
                if size >= target_bytes:
                    yield buffer
                    buffer, size = [], 0
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if buffer:
        yield buffer


# =========================
# Tokenizer training
# =========================


def train_tokenizer(files, vocab_size, out_path):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(f) for f in files],
        vocab_size=vocab_size,
        min_frequency=4,
        special_tokens=["<UNK>", "<|endoftext|>"],
    )
    tokenizer.save(str(out_path))


# =========================
# Worker: tokenize only
# =========================


def tokenize_worker(files, tokenizer_path, max_seq_len, queue):
    chunk_bytes, batch_limit = memory_config()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    for f in files:
        for chunk in stream_text(f, chunk_bytes):
            batch = []
            for t in chunk:
                batch.append(t)
                if len(batch) >= batch_limit:
                    encs = tokenizer.encode_batch(batch)
                    for e in encs:
                        ids = e.ids[:max_seq_len]
                        if ids:
                            queue.put(ids)
                    batch = []

            if batch:
                encs = tokenizer.encode_batch(batch)
                for e in encs:
                    ids = e.ids[:max_seq_len]
                    if ids:
                        queue.put(ids)

    queue.put(None)  # signal done


# =========================
# Writer: single owner of disk
# =========================


def writer_process(bin_path, idx_path, queue, num_workers, split_name):
    total_tokens = 0
    offsets = []
    finished = 0

    with open(bin_path, "wb", buffering=16 * 1024 * 1024) as bf:
        while finished < num_workers:
            item = queue.get()
            if item is None:
                finished += 1
                continue

            arr = np.asarray(item, dtype=np.uint32)
            bf.write(arr.tobytes())
            offsets.append((total_tokens, len(arr)))
            total_tokens += len(arr)

    with open(idx_path, "wb") as f:
        f.write(struct.pack("<Q", len(offsets)))
        for off, ln in offsets:
            f.write(struct.pack("<QQ", off, ln))

    print(f"✓ {split_name}: {total_tokens:,} tokens written")


# =========================
# Tokenize one split
# =========================


def tokenize_split(
    files, tokenizer_path, out_bin, out_idx, max_seq_len, num_workers, split_name
):
    shards: list[list[Unknown] | Unknown] = [[] for _ in range(num_workers)]
    for i, f in enumerate(files):
        shards[i % num_workers].append(f)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=256)

    writer = ctx.Process(
        target=writer_process,
        args=(out_bin, out_idx, queue, num_workers, split_name),
    )
    writer.start()

    workers = []
    for shard in shards:
        p = ctx.Process(
            target=tokenize_worker,
            args=(shard, tokenizer_path, max_seq_len, queue),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
    writer.join()


# =========================
# Main
# =========================


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--train_tokenizer", action="store_true")
    args = p.parse_args()

    root = Path.cwd()
    raw_root = root / args.model_name / "raw_data"
    out_root = root / args.model_name / "processed_data"
    ckpt = root / "checkpoints" / args.model_name
    train_out_root = out_root / "train"
    valid_out_root = out_root / "valid"

    train_out_root.mkdir(parents=True, exist_ok=True)
    valid_out_root.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    tokenizer_path = ckpt / "tokenizer.json"

    train_files = [
        f
        for f in (raw_root / "train").rglob("*")
        if f.suffix.lower() in {".txt", ".parquet", ".csv"}
    ]
    valid_files = [
        f
        for f in (raw_root / "valid").rglob("*")
        if f.suffix.lower() in {".txt", ".parquet", ".csv"}
    ]

    if args.train_tokenizer:
        print("Training tokenizer on train split...")
        train_tokenizer(train_files, args.vocab_size, tokenizer_path)
        print("✓ Tokenizer trained")
        return

    if not tokenizer_path.exists():
        raise RuntimeError("Tokenizer not found. Train it first.")

    tokenize_split(
        train_files,
        tokenizer_path,
        train_out_root / "train.bin",
        train_out_root / "train.idx",
        args.max_seq_len,
        args.num_workers,
        "train",
    )

    tokenize_split(
        valid_files,
        tokenizer_path,
        valid_out_root / "valid.bin",
        valid_out_root / "valid.idx",
        args.max_seq_len,
        args.num_workers,
        "valid",
    )


if __name__ == "__main__":
    main()
