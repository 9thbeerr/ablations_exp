import argparse
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
from tokenizers import ByteLevelBPETokenizer, Tokenizer


# =========================
# Streaming text reader
# =========================


def stream_text(files, chunk_rows=2048):
    for path in files:
        ext = path.suffix.lower()

        if ext == ".parquet":
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["text"])
                col = table["text"]
                for i in range(0, len(col), chunk_rows):
                    batch = col.slice(i, chunk_rows).to_pylist()
                    yield [t for t in batch if t]

        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                buf = []
                for line in f:
                    line = line.strip()
                    if line:
                        buf.append(line)
                    if len(buf) >= chunk_rows:
                        yield buf
                        buf = []
                if buf:
                    yield buf

        elif ext == ".csv":
            for df in pd.read_csv(path, chunksize=chunk_rows):
                rows = (
                    df.select_dtypes(include=["object"])
                    .fillna("")
                    .astype(str)
                    .agg(" ".join, axis=1)
                    .tolist()
                )
                rows = [r for r in rows if r]
                if rows:
                    yield rows

        else:
            raise ValueError(f"Unsupported file type: {path}")


# =========================
# Tokenizer training
# =========================


def train_tokenizer(files, vocab_size, out_path):
    tokenizer = ByteLevelBPETokenizer()

    def iterator():
        for chunk in stream_text(files):
            for t in chunk:
                yield t

    tokenizer.train_from_iterator(
        iterator(),
        vocab_size=vocab_size,
        min_frequency=4,
        special_tokens=["<UNK>", "<|endoftext|>"],
    )

    tokenizer.save(str(out_path))


# =========================
# Streaming token packer
# =========================


def token_sequence_generator(
    files,
    tokenizer_path,
    seq_len,
    eos_token="<|endoftext|>",
):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id(eos_token)

    buffer = []

    for chunk in stream_text(files):
        encs = tokenizer.encode_batch(chunk)

        for e in encs:
            buffer.extend(e.ids)
            if eos_id is not None:
                buffer.append(eos_id)

            while len(buffer) >= seq_len + 1:
                seq = buffer[: seq_len + 1]
                buffer = buffer[seq_len:]
                yield seq

    # drop tail


# =========================
# Main
# =========================


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--train_tokenizer", action="store_true")
    args = p.parse_args()

    root = Path.cwd()
    raw_root = root / args.model_name / "raw_data"
    ckpt = root / "checkpoints" / args.model_name
    ckpt.mkdir(parents=True, exist_ok=True)

    tokenizer_path = ckpt / "tokenizer.json"

    train_files = [
        f
        for f in (raw_root / "train").rglob("*")
        if f.suffix.lower() in {".txt", ".parquet", ".csv"}
    ]

    if args.train_tokenizer:
        train_tokenizer(train_files, args.vocab_size, tokenizer_path)
        print("✓ Tokenizer trained")
        return

    if not tokenizer_path.exists():
        raise RuntimeError("Tokenizer not found")

    # Example usage: plug this generator into your training loop
    gen = token_sequence_generator(
        train_files,
        tokenizer_path,
        args.seq_len,
    )

    # sanity check
    for i, seq in enumerate(gen):
        if i == 5:
            break
        print(len(seq))


if __name__ == "__main__":
    main()
