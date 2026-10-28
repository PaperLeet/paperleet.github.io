# BLIP-2 最大查询图文匹配 — 题解

先求 [B,Q,N] 的归一化相似度，再对 Q 归约。每个文本各自选出匹配最好的 query；先平均 query 会破坏这种多语义槽结构。最大值非并列时，梯度只流向获胜 query，随后经过归一化的雅可比；测试避开并列点做中心差分。

$$
S_{ij}=\frac{1}{\tau}\max_{q}\hat z_{iq}^{T}\hat t_j.
$$

参考实现按上述布局和归约逐步计算：

```python
def max_query_similarity(queries: Tensor, texts: Tensor, temperature: Tensor) -> Tensor:
    _floating(queries, texts, temperature)
    if queries.ndim!=3 or queries.numel()==0 or texts.ndim!=2 or texts.numel()==0 or queries.shape[-1]!=texts.shape[-1] or temperature.ndim!=0 or temperature.item()<=0:
        raise ValueError("expected (B,Q,D), (N,D), positive scalar temperature")
    q=F.normalize(queries,dim=-1,eps=1e-12);t=F.normalize(texts,dim=-1,eps=1e-12)
    return (q@t.T).max(dim=1).values/temperature
```

错误实现检测覆盖这些常见问题：

- Averages query scores instead of selecting the best query
- Uses raw query magnitudes

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

Q-Former 输出的归一化、逐文本 max-query 相似度与梯度路由；不包含 Transformer 或冻结大模型。

- [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)
