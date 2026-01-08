import argparse
import os
import struct
import gc
import shutil
from pathlib import Path
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer, Tokenizer


def read_file_content_chunked(file_path, chunk_size=10000):
    """Ultra memory-efficient - smaller chunks for 2.5GB files"""
    ext = file_path.suffix.lower()

    if ext == ".parquet":
        parquet_file = pq.ParquetFile(file_path)
        for i in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(i, columns=["text"])
            texts = [str(val) for val in table["text"].to_pylist() if val]
            del table

            for j in range(0, len(texts), chunk_size):
                yield "\n".join(texts[j : j + chunk_size])
            del texts

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            buffer = []
            for line in f:
                buffer.append(line.strip())
                if len(buffer) >= chunk_size:
                    yield "\n".join(buffer)
                    buffer = []
            if buffer:
                yield "\n".join(buffer)

    elif ext == ".csv":
        df = pd.read_csv(file_path, chunksize=chunk_size)
        for chunk_df in df:
            text_cols = chunk_df.select_dtypes(include=["object"]).columns
            texts = chunk_df[text_cols].apply(
                lambda x: " ".join(x.dropna().astype(str)), axis=1
            )
            yield "\n".join(texts)

    elif ext == ".jsonl":
        import json

        buffer = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    buffer.append(str(data))
                    if len(buffer) >= chunk_size:
                        yield "\n".join(buffer)
                        buffer = []
                except:
                    continue
            if buffer:
                yield "\n".join(buffer)


def read_file_content_streaming(file_path, chunk_size):
    """Stream file content in larger chunks for tokenization"""
    ext = file_path.suffix.lower()

    if ext == ".parquet":
        parquet_file = pq.ParquetFile(file_path)
        buffer = []
        buffer_size = 0

        for i in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(i, columns=["text"])
            texts = [str(val) for val in table["text"].to_pylist() if val]

            for text in texts:
                buffer.append(text)
                buffer_size += len(text)

                if buffer_size >= chunk_size:
                    yield "\n".join(buffer)
                    buffer = []
                    buffer_size = 0

            del table, texts

        if buffer:
            yield "\n".join(buffer)

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            buffer = []
            buffer_size = 0
            for line in f:
                buffer.append(line.strip())
                buffer_size += len(line)

                if buffer_size >= chunk_size:
                    yield "\n".join(buffer)
                    buffer = []
                    buffer_size = 0

            if buffer:
                yield "\n".join(buffer)

    elif ext == ".csv":
        df = pd.read_csv(file_path, chunksize=1000)
        buffer = []
        buffer_size = 0

        for chunk_df in df:
            text_cols = chunk_df.select_dtypes(include=["object"]).columns
            texts = (
                chunk_df[text_cols]
                .apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
                .tolist()
            )

            for text in texts:
                buffer.append(text)
                buffer_size += len(text)

                if buffer_size >= chunk_size:
                    yield "\n".join(buffer)
                    buffer = []
                    buffer_size = 0

        if buffer:
            yield "\n".join(buffer)

    elif ext == ".jsonl":
        import json

        buffer = []
        buffer_size = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = str(data)
                    buffer.append(text)
                    buffer_size += len(text)

                    if buffer_size >= chunk_size:
                        yield "\n".join(buffer)
                        buffer = []
                        buffer_size = 0
                except:
                    continue

        if buffer:
            yield "\n".join(buffer)


def process_file_to_disk(args):
    """Process one 2.5GB file, write directly to disk"""
    file_path, output_path = args
    try:
        print(f"Processing: {file_path}")
        with open(output_path, "w", encoding="utf-8", buffering=8192 * 1024) as out:
            chunk_count = 0
            for chunk in read_file_content_chunked(Path(file_path)):
                out.write(chunk)
                out.write("\n")
                chunk_count += 1
                if chunk_count % 100 == 0:
                    print(f"  {file_path}: {chunk_count} chunks written")
                del chunk

        print(f"✓ Completed: {file_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error {file_path}: {e}")
        return None


def get_available_memory():
    """Get available system memory in GB"""
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)


def get_max_batch_size():
    """Calculate max batch size based on available memory"""
    available_gb = get_available_memory()
    usable_gb = available_gb * 0.7  # Use 70% of available memory

    # Estimate: 1GB can hold ~150 text chunks (5MB each)
    max_chunks = int(usable_gb * 150)
    return max(16, min(max_chunks, 2000))  # Between 16-2000


def train_tokenizer(args, training_filename_list, tokenizer_path, checkpoints):
    """Train BPE tokenizer on raw data files"""
    print(
        f"Processing {len(training_filename_list)} files with {args.num_workers} workers"
    )
    print("⚠️  Each file is ~2.5GB - processing with memory limits")

    temp_dir = checkpoints / "temp_txt"
    temp_dir.mkdir(exist_ok=True)

    # Create file pairs
    file_pairs = [
        (file_path, str(temp_dir / f"temp_{i}.txt"))
        for i, file_path in enumerate(training_filename_list)
    ]

    # Process files at a time with imap (streaming results)
    temp_files = []
    with mp.Pool(args.num_workers) as pool:
        for result in pool.imap(process_file_to_disk, file_pairs):
            if result:
                temp_files.append(result)
                print(f"Progress: {len(temp_files)}/{len(file_pairs)} files completed")

            gc.collect()

    print(f"✓ Created {len(temp_files)} temp files")

    # Train tokenizer
    print("Training tokenizer...")
    bpe_tokenizer = ByteLevelBPETokenizer()
    bpe_tokenizer.train(
        files=temp_files,
        vocab_size=args.vocab_size,
        min_frequency=4,
        special_tokens=["<UNK>", "<|endoftext|>"],
    )
    bpe_tokenizer.save(str(tokenizer_path))
    print(f"✓ Saved to {tokenizer_path}")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("✓ Cleanup complete")


def tokenize_data(args, data_dir, tokenizer_path, found_extensions):
    """Tokenize data files and save as binary format"""
    print("Starting tokenization...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_padding()

    # Check memory
    available_mem = get_available_memory()
    print(f"Available memory: {available_mem:.1f}GB")

    for mode in ["train", "valid"]:
        processed_data_dir = data_dir / "processed_data" / mode
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        tokenized_data_bin_path = processed_data_dir / f"{mode}.bin"
        tokenized_data_idx_path = processed_data_dir / f"{mode}.idx"

        # Auto-detect files
        mode_path = Path(data_dir, "raw_data", mode)
        tokenize_filename_list = []
        for ext in found_extensions:
            files = [f for f in mode_path.rglob(f"*{ext}") if f.is_file()]
            tokenize_filename_list.extend(files)

        if not tokenize_filename_list:
            print(f"⚠️  No files found for {mode} mode, skipping...")
            continue

        print(f"\nTokenizing {len(tokenize_filename_list)} files for {mode}")

        # Dynamic batch size based on memory
        max_batch_size = get_max_batch_size()
        print(f"Using batch size: {max_batch_size} chunks (targeting 70% RAM usage)")

        # Larger text chunks for efficiency
        text_chunk_size = 5 * 1024 * 1024  # 5MB per chunk

        doc_offsets = []
        total_tokens = 0
        chunk_count = 0

        with open(tokenized_data_bin_path, "wb") as out_file:
            for file_idx, raw_data_file in enumerate(tokenize_filename_list):
                print(
                    f"\nProcessing file {file_idx + 1}/{len(tokenize_filename_list)}: {raw_data_file}"
                )

                text_chunks = []

                # Load as many chunks as memory allows
                for text_chunk in read_file_content_streaming(
                    raw_data_file, text_chunk_size
                ):
                    text_chunks.append(text_chunk)

                    # Check memory usage periodically
                    if len(text_chunks) % 100 == 0:
                        current_mem = psutil.virtual_memory().percent
                        if current_mem > 85:  # Safety threshold
                            print(
                                f"  Memory at {current_mem}% - processing batch early"
                            )
                            break

                    # Process when batch is full
                    if len(text_chunks) >= max_batch_size:
                        print(f"  Processing batch of {len(text_chunks)} chunks...")

                        # Tokenize entire batch at once
                        encodings = tokenizer.encode_batch(text_chunks)

                        for encoding in encodings:
                            token_ids = encoding.ids
                            if len(token_ids) > 0:
                                token_arr = np.array(token_ids, dtype=np.uint16)
                                out_file.write(token_arr.tobytes())
                                doc_offsets.append((total_tokens, len(token_ids)))
                                total_tokens += len(token_ids)
                                chunk_count += 1
                                print(f"Processed {chunk_count}")

                        print(
                            f"  Total: {chunk_count} chunks, {total_tokens:,} tokens, "
                            f"RAM: {psutil.virtual_memory().percent:.1f}%"
                        )

                        text_chunks = []
                        del encodings
                        gc.collect()

                # Process remaining chunks from this file
                if text_chunks:
                    print(f"  Processing final batch of {len(text_chunks)} chunks...")
                    encodings = tokenizer.encode_batch(text_chunks)

                    for encoding in encodings:
                        token_ids = encoding.ids
                        if len(token_ids) > 0:
                            token_arr = np.array(token_ids, dtype=np.uint16)
                            out_file.write(token_arr.tobytes())
                            doc_offsets.append((total_tokens, len(token_ids)))
                            total_tokens += len(token_ids)
                            chunk_count += 1
                            print(f"Processed {chunk_count}")

                    del text_chunks, encodings
                    gc.collect()

                print(f"✓ File complete: {total_tokens:,} tokens")

        # Write index file
        with open(tokenized_data_idx_path, "wb") as f:
            f.write(struct.pack("<Q", len(doc_offsets)))
            for offset, length in doc_offsets:
                f.write(struct.pack("<QQ", offset, length))

        print(
            f"\n✓ {mode} tokenization complete: {total_tokens:,} tokens in {chunk_count} chunks"
        )
        print(f"Final memory usage: {psutil.virtual_memory().percent:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--vocab_size", type=int, default=1024, help="Tokenizer vocabulary size"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=256, help="Maximum sequence length"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model/dataset name"
    )
    parser.add_argument(
        "--train_tokenizer", action="store_true", help="Train new tokenizer"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of parallel workers"
    )
    args = parser.parse_args()
    print(args)

    # Setup paths
    model_name = args.model_name
    root_path = Path(os.getcwd())
    data_dir = root_path / model_name

    # Auto-detect file extensions
    supported_extensions = {".txt", ".parquet", ".csv", ".jsonl"}
    raw_data_path = Path(data_dir, "raw_data")

    found_extensions = set()
    for f in raw_data_path.rglob("*"):
        if f.is_file() and f.suffix.lower() in supported_extensions:
            found_extensions.add(f.suffix.lower())

    print(f"Auto-detected extensions: {found_extensions}")

    # Get all training files
    training_filename_list = []
    for ext in found_extensions:
        files = [
            str(f.relative_to(root_path))
            for f in raw_data_path.rglob(f"*{ext}")
            if f.is_file()
        ]
        training_filename_list.extend(files)

    # Setup checkpoint directory
    checkpoints = root_path / "checkpoints" / model_name
    checkpoints.mkdir(parents=True, exist_ok=True)
    tokenizer_path = checkpoints / f"tokenizer_{model_name}.json"

    # Train tokenizer or tokenize data
    if args.train_tokenizer:
        train_tokenizer(args, training_filename_list, tokenizer_path, checkpoints)
    else:
        if not tokenizer_path.exists():
            print(f"❌ Tokenizer not found at {tokenizer_path}")
            print("Run with --train_tokenizer first to create tokenizer")
            return

        tokenize_data(args, data_dir, tokenizer_path, found_extensions)


if __name__ == "__main__":
    main()
