# PagedAttention: KV Pages and Decode Attention

## 1. Task

Logical token order can differ from physical KV-cache order. [PagedAttention](https://arxiv.org/abs/2309.06180) connects them with a block table, allowing scattered physical pages and shared prefixes.

Implement logical KV gathering and multihead decode attention with one query per request. This CPU specification checks indexing and numerical equivalence; it does not implement a CUDA kernel, scheduler or allocator.

## 2. Interfaces and shapes

```python
gather_paged_kv(key_cache, value_cache, block_table, context_len) -> (keys, values)
paged_attention(query, key_cache, value_cache, block_tables, context_lens) -> Tensor
```

- Both caches have shape `(P,S,H,D)`: physical pages, tokens per page, heads and head width.
- Gathering accepts an int64 table `(M,)` and a positive Python integer length `L`; it returns two `(L,H,D)` tensors.
- Attention accepts query `(B,H,D)`, int64 tables `(B,M)` and int64 lengths `(B,)`; it returns `(B,H,D)`.

All tensors share a device. Floating inputs share float32 or float64 dtype. This exercise uses the same head count and feature width for queries, keys and values.

## 3. Logical-to-physical mapping

Logical token $t$ reads

$$
\operatorname{page}(t)=\operatorname{table}[\lfloor t/S\rfloor],\qquad
\operatorname{slot}(t)=t\bmod S.
$$

A length $L$ uses the first $\lceil L/S\rceil$ table entries and only the valid prefix of the last page. For `S=3, table=[4,1,5], L=7`, read all three tokens of page 4, all three of page 1, then the first token of page 5.

Active page IDs must be in `[0,P)`. Unused table entries may contain arbitrary integer sentinels. Unused pages and tail slots may contain NaN and must not participate. Queries and active KV tokens must be finite.

## 4. Attention and gradients

For each request and head, compute attention across its entire valid context:

$$
a_t=\operatorname{softmax}_t(q^\top k_t/\sqrt D),\qquad
o=\sum_{t=0}^{L-1}a_t v_t.
$$

The cache already contains every visible token, including the last position. Normalize across tokens, rather than heads or separate pages. All context lengths are positive.

Preserve input contents and gradients to queries and physical KV tensors. A physical page may be used by several requests or logical positions; its gradient must sum all contributions.

## 5. Tests and experiment

Tests compare with scalar attention, remap relocated physical pages, poison unused slots, verify shared-page gradients and use finite differences. The behavioral experiment replays two requests with a shared prefix across nine lengths, comparing with dense scalar attention and counting allocated token slots.

Slot counts describe this synthetic storage layout. They are not GPU throughput or memory-performance measurements.

## 6. Run

```bash
paperleet init ai_infra/paged_attention
paperleet run ai_infra/paged_attention
```

## 7. Sources

[Paper](https://arxiv.org/abs/2309.06180), Sections 4.1–4.4; [official kernel design](https://docs.vllm.ai/en/latest/design/paged_attention/). The teaching layout is `(P,S,H,D)` on CPU. GPU vectorized layouts, copy-on-write and preemptive scheduling are outside this exercise.
