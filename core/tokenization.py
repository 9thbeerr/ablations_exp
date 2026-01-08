import argparse
import os
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

    # Use 25% of available RAM for text buffers
    budget = int(avail * 0.25)

    # Hard caps (never exceed)
    chunk_bytes = min(budget // 4, 32 * 1024 * 1024)  # ≤32 MB
    batch_limit = min(max(budget // (8 * 1024 * 1024), 8), 64)

    return max(chunk_bytes, 4 * 1024 * 1024), batch_limit


# =========================
# Streaming readers
# =========================


def stream_text_from_parquet(path, target_bytes):
    pf = pq.ParquetFile(path)
    buffer, size = [], 0

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["text"])
        col = table["text"]

        for i in range(len(col)):
            txt = col[i].as_py()
            if not txt:
                continue

            buffer.append(txt)
            size += len(txt)

            if size >= target_bytes:
                yield buffer
                buffer, size = [], 0

    if buffer:
        yield buffer


def stream_text_from_txt(path, target_bytes):
    buffer, size = [], 0
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

    if buffer:
        yield buffer


def stream_text_from_csv(path, target_bytes):
    for df in pd.read_csv(path, chunksize=1000):
        rows = (
            df.select_dtypes(include=["object"])
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )

        buffer, size = [], 0
        for r in rows:
            if not r:
                continue

            buffer.append(r)
            size += len(r)

            if size >= target_bytes:
                yield buffer
                buffer, size = [], 0

        if buffer:
            yield buffer


def stream_text(path, target_bytes):
    ext = path.suffix.lower()
    if ext == ".parquet":
        yield from stream_text_from_parquet(path, target_bytes)
    elif ext == ".txt":
        yield from stream_text_from_txt(path, target_bytes)
    elif ext == ".csv":
        yield from stream_text_from_csv(path, target_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


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
# Tokenization
# =========================


def tokenize_worker(args):
    files, tokenizer_path, out_bin, out_idx, max_seq_len = args
    chunk_bytes, batch_limit = memory_config()

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    doc_offsets = []
    total_tokens = 0

    with open(out_bin, "wb", buffering=16 * 1024 * 1024) as bf:
        for f in files:
            for chunk in stream_text(f, chunk_bytes):
                batch = []
                for txt in chunk:
                    batch.append(txt)
                    if len(batch) >= batch_limit:
                        encs = tokenizer.encode_batch(batch)
                        for e in encs:
                            ids = e.ids[:max_seq_len]
                            if ids:
                                arr = np.asarray(ids, dtype=np.uint32)
                                bf.write(arr.tobytes())
                                doc_offsets.append((total_tokens, len(ids)))
                                total_tokens += len(ids)
                        batch = []

                if batch:
                    encs = tokenizer.encode_batch(batch)
                    for e in encs:
                        ids = e.ids[:max_seq_len]
                        if ids:
                            arr = np.asarray(ids, dtype=np.uint32)
                            bf.write(arr.tobytes())
                            doc_offsets.append((total_tokens, len(ids)))
                            total_tokens += len(ids)

    with open(out_idx, "wb") as f:
        f.write(struct.pack("<Q", len(doc_offsets)))
        for off, ln in doc_offsets:
            f.write(struct.pack("<QQ", off, ln))

    return total_tokens


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
    raw_dir = root / args.model_name / "raw_data"
    ckpt = root / "checkpoints" / args.model_name
    ckpt.mkdir(parents=True, exist_ok=True)

    tokenizer_path = ckpt / "tokenizer.json"

    files = [
        f
        for f in raw_dir.rglob("*")
        if f.suffix.lower() in {".txt", ".parquet", ".csv"}
    ]

    if args.train_tokenizer:
        print("Training tokenizer...")
        train_tokenizer(files, args.vocab_size, tokenizer_path)
        print("✓ Tokenizer trained")
        return

    if not tokenizer_path.exists():
        raise RuntimeError("Tokenizer not found. Train it first.")

    out_dir = root / args.model_name / "processed_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split files across workers
    shards = [[] for _ in range(args.num_workers)]
    for i, f in enumerate(files):
        shards[i % args.num_workers].append(f)

    tasks = []
    for i, shard in enumerate(shards):
        tasks.append(
            (
                shard,
                tokenizer_path,
                out_dir / f"train_{i}.bin",
                out_dir / f"train_{i}.idx",
                args.max_seq_len,
            )
        )

    with mp.Pool(args.num_workers) as pool:
        totals = pool.map(tokenize_worker, tasks)

    print(f"✓ Total tokens written: {sum(totals):,}")


if __name__ == "__main__":
    main()
