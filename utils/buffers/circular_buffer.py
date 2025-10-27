# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Optional, Sequence


class CircularBuffer:
    """Circular buffer for storing a history of batched tensor data."""

    def __init__(self, max_len: int, batch_size: int, device: str):
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")

        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)
        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._pointer: int = -1
        self._buffer: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_length(self) -> int:
        return int(self._max_len[0].item())

    @property
    def current_length(self) -> torch.Tensor:
        return torch.minimum(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> torch.Tensor:
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        buf = torch.flip(buf, dims=[0])
        return buf

    def reset(self, batch_ids: Optional[Sequence[int]] = None):
        if batch_ids is None:
            batch_ids = slice(None)
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor):
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has {data.shape[0]} environments while expecting {self.batch_size}")

        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self._device)

        self._pointer = (self._pointer + 1) % self.max_length
        self._buffer[self._pointer] = data.to(self._device)

        if 0 in self._num_pushes.tolist():
            fill_ids = [i for i, x in enumerate(self._num_pushes.tolist()) if x == 0]
            self._buffer[:, fill_ids, :] = data.to(self._device)[fill_ids]

        self._num_pushes += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        if len(key) != self.batch_size:
            raise ValueError(f"The argument 'key' has length {key.shape[0]}, while expecting {self.batch_size}")
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        valid_keys = torch.minimum(key, self._num_pushes - 1)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
        return self._buffer[index_in_buffer, self._ALL_INDICES]
