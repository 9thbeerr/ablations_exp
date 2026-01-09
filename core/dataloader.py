import torch
import numpy as np
from tokenizers import Tokenizer
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from typing import Iterator, Tuple


class StreamingMegatronDataset:
    def __init__(
        self,
        files,
        tokenizer_path: str,
        context_length: int,
        batch_size: int,
        device: str,
        shuffle_buffer: int = 8192,
        eos_token: str = "<|endoftext|>",
    ):
        self.files = files
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.eos_id = self.tokenizer.token_to_id(eos_token)

        self.shuffle_buffer = shuffle_buffer

    # -------------------------
    # Text streaming
    # -------------------------

    def _stream_text(self, chunk_rows=2048):
        for path in self.files:
            ext = path.suffix.lower()

            if ext == ".parquet":
                pf = pq.ParquetFile(path)
                for rg in range(pf.num_row_groups):
                    table = pf.read_row_group(rg, columns=["text"])
                    col = table["text"]
                    for i in range(0, len(col), chunk_rows):
                        batch = col.slice(i, chunk_rows).to_pylist()
                        for t in batch:
                            if t:
                                yield t

            elif ext == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line

            elif ext == ".csv":
                for df in pd.read_csv(path, chunksize=chunk_rows):
                    rows = (
                        df.select_dtypes(include=["object"])
                        .fillna("")
                        .astype(str)
                        .agg(" ".join, axis=1)
                        .tolist()
                    )
                    for r in rows:
                        if r:
                            yield r

            else:
                raise ValueError(f"Unsupported file type: {path}")

    # -------------------------
    # Main iterator
    # -------------------------

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        token_buffer = []
        seq_buffer = []

        for text in self._stream_text():
            enc = self.tokenizer.encode(text)
            token_buffer.extend(enc.ids)
            if self.eos_id is not None:
                token_buffer.append(self.eos_id)

            while len(token_buffer) >= self.context_length + 1:
                seq = token_buffer[: self.context_length + 1]
                token_buffer = token_buffer[self.context_length :]

                seq_buffer.append(seq)

                # lightweight shuffle
                if len(seq_buffer) >= self.shuffle_buffer:
                    np.random.shuffle(seq_buffer)

                if len(seq_buffer) >= self.batch_size:
                    batch = seq_buffer[: self.batch_size]
                    seq_buffer = seq_buffer[self.batch_size :]

                    x = np.stack([s[:-1] for s in batch])
                    y = np.stack([s[1:] for s in batch])

                    yield (
                        torch.from_numpy(x).to(self.device),
                        torch.from_numpy(y).to(self.device),
                    )
