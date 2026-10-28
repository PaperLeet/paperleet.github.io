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
    """Follow the block table to read one request's logical KV sequence.

    Returns:
        A tuple (keys, values), both Tensors of shape (context_len,H,D), with
        cache dtype/device and gradients connected to the original cache. Cache
        shape is (P,S,H,D); table[j] is physical page for logical page j. Ignore
        unused table entries and final-page tail slots, which may contain NaN.
        Active tokens must be finite. Preserve input storage and page order.
    """
    _check_cache(key_cache, value_cache)
    _check_table(key_cache, block_table, context_len)
    raise NotImplementedError("Implement logical-to-physical KV gathering")


def paged_attention(query: Tensor, key_cache: Tensor, value_cache: Tensor,
                    block_tables: Tensor, context_lens: Tensor) -> Tensor:
    """Compute one decode query per request over its own logical KV context.

    Returns:
        A Tensor of shape (B,H,D), with query dtype/device. Each query head uses
        dot products scaled by 1/sqrt(D), softmax over all context tokens, then a
        weighted value sum. Tables are int64 (B,M); lengths are int64 (B,) and
        positive. Preserve gradients to query, keys and values; shared physical
        pages accumulate gradients from all uses. Do not write into any input.
    """
    _check_cache(key_cache, value_cache)
    if query.ndim != 3 or query.shape[0] < 1 or query.shape[1:] != key_cache.shape[2:]:
        raise ValueError("query must have shape (B,H,D) matching the cache")
    if query.dtype != key_cache.dtype or query.device != key_cache.device or not torch.isfinite(query).all():
        raise ValueError("query must be finite and share cache dtype/device")
    if block_tables.ndim != 2 or block_tables.shape[0] != query.shape[0] or block_tables.dtype != torch.long or block_tables.device != query.device:
        raise ValueError("block_tables must have matching-device int64 shape (B,M)")
    if context_lens.shape != query.shape[:1] or context_lens.dtype != torch.long or context_lens.device != query.device:
        raise ValueError("context_lens must have matching-device int64 shape (B,)")
    raise NotImplementedError("Implement per-request paged attention")
