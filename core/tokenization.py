import argparse
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
import numpy as np
import os
from pathlib import Path
import struct
import pandas as pd
import pyarrow.parquet as pq


def read_parquet_content(file_path):
    """
    Memory-efficient parquet reader that streams and yields text.
    Reads only the 'text' column.
    """
    parquet_file = pq.ParquetFile(file_path)

    # Use only 'text' column
    if "text" not in parquet_file.schema.names:
        raise ValueError("Column 'text' not found in parquet file")

    result_lines = []

    # Process each row group (natural chunks in parquet)
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=["text"])

        # Process rows in this group
        for i in range(table.num_rows):
            val = table["text"][i].as_py()
            if val:  # Skip None/null values
                result_lines.append(str(val))

        # Yield periodically to avoid memory buildup
        if len(result_lines) >= 10000:
            yield "\n".join(result_lines)
            result_lines = []

    # Yield remaining lines
    if result_lines:
        yield "\n".join(result_lines)


# Usage example - to get all text at once:
def read_parquet_all_text(file_path):
    """If you need all text at once (use carefully with large files)"""
    return "\n".join(read_parquet_content(file_path))


# Usage example - streaming:
# for chunk in read_parquet_content("data.parquet"):
#     process(chunk)


def read_file_content(file_path):
    """Read content from various file formats."""
    ext = file_path.suffix.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".parquet":
        return read_parquet_all_text(file_path)

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        text_cols = df.select_dtypes(include=["object"]).columns
        return "\n".join(
            df[text_cols].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        )

    elif ext == ".jsonl":
        import json

        texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                texts.append(str(data))
        return "\n".join(texts)

    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_tokenizer", action="store_true")
    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    root_path = Path(os.getcwd())
    data_dir = root_path / model_name
    train_tokenizer = args.train_tokenizer

    # Auto-detect file extensions from directory
    supported_extensions = {".txt", ".parquet", ".csv", ".jsonl"}
    raw_data_path = Path(data_dir, "raw_data")

    found_extensions = set()
    for f in raw_data_path.rglob("*"):
        if f.is_file() and f.suffix.lower() in supported_extensions:
            found_extensions.add(f.suffix.lower())

    print(f"Auto-detected extensions: {found_extensions}")

    # Get all files with detected extensions
    training_filename_list = []
    for ext in found_extensions:
        files = [
            str(f.relative_to(root_path))
            for f in raw_data_path.rglob(f"*{ext}")
            if f.is_file()
        ]
        training_filename_list.extend(files)

    checkpoints = root_path / "checkpoints" / model_name
    checkpoints.mkdir(parents=True, exist_ok=True)
    tokenizer_path = checkpoints / f"tokenizer_{model_name}.json"

    if train_tokenizer:
        print(f"Training BPETokenizer with {len(training_filename_list)} files")

        # Create temp txt files for tokenizer training
        temp_dir = checkpoints / "temp_txt"
        temp_dir.mkdir(exist_ok=True)
        temp_files = []

        for i, file_path in enumerate(training_filename_list):
            content = read_file_content(Path(file_path))
            temp_file = temp_dir / f"temp_{i}.txt"
            print(len(content))
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(content[:-1][:200])
            del content
            temp_files.append(str(temp_file))
            print("temp file locations with .txt", temp_files)

        bpe_tokenizer = ByteLevelBPETokenizer()
        bpe_tokenizer.train(
            files=temp_files,
            vocab_size=args.vocab_size,
            min_frequency=4,
            special_tokens=["<UNK>", "<|endoftext|>"],
        )
        bpe_tokenizer.save(str(tokenizer_path))

        # Cleanup temp files
        for tf in temp_files:
            os.remove(tf)
        temp_dir.rmdir()

    else:
        print("Starting tokenization...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        for mode in ["train", "valid"]:
            processed_data_dir = data_dir / "processed_data" / mode
            processed_data_dir.mkdir(parents=True, exist_ok=True)

            tokenized_data_bin_path = processed_data_dir / f"{mode}.bin"
            tokenized_data_idx_path = processed_data_dir / f"{mode}.idx"

            # Auto-detect files to tokenize
            mode_path = Path(data_dir, "raw_data", mode)
            tokenize_filename_list = []

            for ext in found_extensions:
                files = [f for f in mode_path.rglob(f"*{ext}") if f.is_file()]
                tokenize_filename_list.extend(files)

            print(f"Tokenizing {len(tokenize_filename_list)} files for {mode}")

            chunk_size = args.max_seq_len * 1024 * 256
            doc_offsets = []
            chunk_count = 0
            total_tokens = 0

            with open(tokenized_data_bin_path, "wb") as out_file:
                for raw_data_file in tokenize_filename_list:
                    content = read_file_content(raw_data_file)

                    # Process in chunks
                    for i in range(0, len(content), chunk_size):
                        text_chunk = content[i : i + chunk_size]
                        if not text_chunk:
                            break

                        token_ids = tokenizer.encode(text_chunk).ids
                        token_arr = np.array(token_ids, dtype=np.uint16)
                        out_file.write(token_arr.tobytes())

                        doc_offsets.append((total_tokens, len(token_ids)))
                        total_tokens += len(token_ids)
                        chunk_count += 1
                        print(f"Chunks: {chunk_count}, Tokens: {total_tokens}")

            with open(tokenized_data_idx_path, "wb") as f:
                f.write(struct.pack("<Q", len(doc_offsets)))
                for offset, length in doc_offsets:
                    f.write(struct.pack("<QQ", offset, length))
