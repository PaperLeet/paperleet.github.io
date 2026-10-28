# BUG: 把物理页顺序当成逻辑块表
"""Logical KV pages and single-query attention: a CPU specification of PagedAttention."""
import math
import torch
from torch import Tensor


def _check_cache(key_cache, value_cache):
    if key_cache.ndim != 4 or key_cache.numel() == 0 or not key_cache.is_floating_point():
        raise ValueError("key_cache must have nonempty floating shape (P,S,H,D)")
    if value_cache.shape != key_cache.shape or value_cache.dtype != key_cache.dtype or value_cache.device != key_cache.device:
        raise ValueError("value_cache must match key_cache")


def _check_table(key_cache, block_table, context_len):
    if block_table.ndim != 1 or block_table.dtype != torch.long or block_table.device != key_cache.device:
        raise ValueError("block_table must be a matching-device int64 vector")
    if type(context_len) is not int or context_len < 1 or context_len > block_table.numel() * key_cache.shape[1]:
        raise ValueError("context_len must fit the table and be a positive Python integer")
    count = (context_len + key_cache.shape[1] - 1) // key_cache.shape[1]
    active = block_table[:count]
    if ((active < 0) | (active >= key_cache.shape[0])).any():
        raise ValueError("active block IDs must index physical cache pages")


def gather_paged_kv(key_cache: Tensor, value_cache: Tensor, block_table: Tensor,
                    context_len: int) -> tuple[Tensor, Tensor]:
    """Return logical-order (L,H,D) key/value tensors; ignore unused pages and tail slots."""
    _check_cache(key_cache, value_cache)
    _check_table(key_cache, block_table, context_len)
    page_size = key_cache.shape[1]
    count = (context_len + page_size - 1) // page_size
    active = torch.arange(count, device=key_cache.device)
    keys = key_cache[active].reshape(-1, *key_cache.shape[2:])[:context_len]
    values = value_cache[active].reshape(-1, *value_cache.shape[2:])[:context_len]
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("active KV tokens must be finite")
    return keys, values


def paged_attention(query: Tensor, key_cache: Tensor, value_cache: Tensor,
                    block_tables: Tensor, context_lens: Tensor) -> Tensor:
    """Return (B,H,D) scaled attention over each request's logical context."""
    _check_cache(key_cache, value_cache)
    if query.ndim != 3 or query.shape[0] < 1 or query.shape[1:] != key_cache.shape[2:]:
        raise ValueError("query must have shape (B,H,D) matching the cache")
    if query.dtype != key_cache.dtype or query.device != key_cache.device or not torch.isfinite(query).all():
        raise ValueError("query must be finite and share cache dtype/device")
    if block_tables.ndim != 2 or block_tables.shape[0] != query.shape[0] or block_tables.dtype != torch.long or block_tables.device != query.device:
        raise ValueError("block_tables must have matching-device int64 shape (B,M)")
    if context_lens.shape != query.shape[:1] or context_lens.dtype != torch.long or context_lens.device != query.device:
        raise ValueError("context_lens must have matching-device int64 shape (B,)")
    outputs = []
    for index in range(query.shape[0]):
        keys, values = gather_paged_kv(key_cache, value_cache, block_tables[index], int(context_lens[index].item()))
        scores = (keys * query[index]).sum(-1).transpose(0, 1) / math.sqrt(query.shape[-1])
        weights = torch.softmax(scores, dim=-1).transpose(0, 1)
        outputs.append((weights[:, :, None] * values).sum(0))
    return torch.stack(outputs)
