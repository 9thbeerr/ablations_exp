import numpy as np
import torch
from typing import Tuple


class MegatronDataset:
    def __init__(self, bin_path: str, context_length: int, dtype=np.uint32):
        self.context_length = context_length
        self.tokens = np.memmap(bin_path, dtype=dtype, mode="r")

        # Number of full sequences available
        self.num_samples = len(self.tokens) // context_length

    def get_batch(
        self,
        batch_size: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Random sample indices
        idxs = np.random.randint(0, self.num_samples, size=batch_size)

        x = np.empty((batch_size, self.context_length), dtype=np.int64)
        y = np.empty((batch_size, self.context_length), dtype=np.int64)

        for i, idx in enumerate(idxs):
            start = idx * self.context_length
            end = start + self.context_length

            x[i] = self.tokens[start:end]
            y[i] = self.tokens[start + 1 : end + 1]

        return (
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device),
        )