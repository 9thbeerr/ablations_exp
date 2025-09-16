import numpy as np
import torch
import struct
from typing import Tuple, List


class MegatronDataset:
    def __init__(self, bin_path: str, idx_path: str, dtype=np.uint16):
        self.tokens = np.memmap(bin_path, dtype=dtype, mode="r")
        self.doc_offsets = self._load_index(idx_path)
        self.total_docs = len(self.doc_offsets)

    def _load_index(self, idx_path: str) -> List[Tuple[int, int]]:
        with open(idx_path, "rb") as f:
            doc_count = struct.unpack("<Q", f.read(8))[0]
            return [struct.unpack("<QQ", f.read(16)) for _ in range(doc_count)]

    def get_batch(self, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        x_list = []
        y_list = []

        attempts = 0
        while len(x_list) < batch_size and attempts < batch_size * 4:
            # Random document
            doc_id = np.random.randint(0, self.total_docs)
            offset, length = self.doc_offsets[doc_id]

            if length <= context_length:
                attempts += 1
                continue  # Skip short docs

            # Random start inside doc
            max_start = length - context_length - 1
            start = np.random.randint(0, max_start)
            idx = offset + start

            x = self.tokens[idx:idx + context_length]
            y = self.tokens[idx + 1:idx + 1 + context_length]

            x_list.append(np.array(x, copy=False))
            y_list.append(np.array(y, copy=False))

        x_batch = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
        y_batch = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)

        return x_batch, y_batch