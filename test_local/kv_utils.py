"""Helpers for flattening/unflattening KV cache tensors emitted by exports."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch


def flatten_kv(past: Sequence[Sequence[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    """Convert ((k0, v0), (k1, v1), ...) into (k0, v0, k1, v1, ...)."""

    flat = []
    for layer in past:
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise ValueError("past must be iterable of (k, v) pairs")
        flat.extend(layer)
    return tuple(flat)


def unflatten_kv(flat: Sequence[torch.Tensor], num_layers: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Inverse of flatten_kv."""

    if len(flat) != num_layers * 2:
        raise ValueError(f"expected {num_layers * 2} tensors, got {len(flat)}")
    it = iter(flat)
    layers = []
    for _ in range(num_layers):
        k = next(it)
        v = next(it)
        layers.append((k, v))
    return tuple(layers)


def truncate_kv(flat: Sequence[torch.Tensor], keep_tokens: int) -> Tuple[torch.Tensor, ...]:
    """Keep only the most recent `keep_tokens` positions along the sequence dimension."""

    if keep_tokens is None:
        return tuple(flat)
    truncated = []
    for tensor in flat:
        if tensor.ndim < 3:
            truncated.append(tensor)
            continue
        seq_dim = tensor.ndim - 2  # (..., seq, head_dim)
        seq_len = tensor.shape[seq_dim]
        if keep_tokens >= seq_len:
            truncated.append(tensor)
            continue
        start = seq_len - keep_tokens
        slicer = [slice(None)] * tensor.ndim
        slicer[seq_dim] = slice(start, seq_len)
        truncated.append(tensor[tuple(slicer)])
    return tuple(truncated)


def kv_sequence_length(flat: Sequence[torch.Tensor]) -> int:
    """Return the sequence length encoded inside a flattened KV cache."""

    for tensor in flat:
        if tensor.ndim >= 3:
            return int(tensor.shape[-2])
    return 0
