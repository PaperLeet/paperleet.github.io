# PagedAttention：分页 KV 缓存与解码注意力

## 1. 要解决的问题

KV cache 的逻辑 token 顺序不必等于物理存储顺序。[PagedAttention](https://arxiv.org/abs/2309.06180) 用块表连接二者，使同一请求读取分散的物理页，并允许多个请求共享前缀页。

实现逻辑 KV 读取，以及每个请求一个 query 的多头解码注意力。CPU 版本用于检验索引和数学等价性，不要求 CUDA kernel、调度器或缓存分配器。

## 2. 接口与形状

```python
gather_paged_kv(key_cache, value_cache, block_table, context_len) -> (keys, values)
paged_attention(query, key_cache, value_cache, block_tables, context_lens) -> Tensor
```

- 两个 cache 均为 `(P,S,H,D)`：物理页数、每页 token 数、head 数、每个 head 的维度。二者保持相同 dtype/device。
- `gather_paged_kv` 的块表是 int64 `(M,)`，`context_len=L` 是正 Python 整数，返回两个 `(L,H,D)` 张量。
- `paged_attention` 的 query 为 `(B,H,D)`，块表为 int64 `(B,M)`，长度为 int64 `(B,)`，返回 `(B,H,D)`。

所有整数元数据与 cache 在同一 device。浮点输入使用 float32 或 float64。这里 query 与 KV 使用相同 head 数和特征维度。

## 3. 逻辑到物理的映射

对逻辑 token $t$，读取

$$
\operatorname{page}(t)=\operatorname{table}[\lfloor t/S\rfloor],\qquad
\operatorname{slot}(t)=t\bmod S.
$$

长度为 $L$ 时使用前 $\lceil L/S\rceil$ 个块表项。最后一页只读取有效前缀。比如 `S=3, table=[4,1,5], L=7` 时，逻辑序列是页 4 的三个 token、页 1 的三个 token、页 5 的第一个 token。

有效块号必须在 `[0,P)`；未使用的块表项可以是任意占位整数。未使用的页和尾槽可以含 NaN，必须完全忽略。参与计算的 query 和有效 KV token 必须有限。

## 4. 注意力与梯度

对每个请求、每个 head，在其全部有效上下文上计算

$$
a_t=\operatorname{softmax}_t(q^\top k_t/\sqrt D),\qquad
o=\sum_{t=0}^{L-1}a_t v_t.
$$

cache 已包含所有可见 token，勿额外丢弃最后一个位置。softmax 沿逻辑 token 维度计算，不能独立对每页归一化。每个上下文长度必须为正。

不修改输入，保留 query 与物理 KV 张量的梯度。同一物理页可以在多个请求或逻辑位置复用，反向传播应累加所有使用产生的梯度。

## 5. 测试与实验

测试使用独立标量 attention、物理页重排不变性、未使用槽位污染、共享页梯度累加和有限差分。T3 回放两个共享前缀的请求，跨越九种长度并与标量连续注意力比较，同时统计已使用的物理 token 槽数。

实验的槽数是这两个请求的逻辑存储计算，不能用于推断 GPU 吞吐量或显存性能。

## 6. 怎么运行

```bash
paperleet init ai_infra/paged_attention
paperleet run ai_infra/paged_attention
```

## 7. 来源

[论文](https://arxiv.org/abs/2309.06180)，Sections 4.1–4.4；[官方 kernel 设计说明](https://docs.vllm.ai/en/latest/design/paged_attention/)。练习采用易读的 `(P,S,H,D)` CPU 布局，未实现 GPU 向量化内存布局、copy-on-write 或调度抢占。
