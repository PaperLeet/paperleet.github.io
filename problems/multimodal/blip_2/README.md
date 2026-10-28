# BLIP-2 最大查询图文匹配

## 1. 问题

实现 `max_query_similarity(queries: Tensor, texts: Tensor, temperature: Tensor) -> Tensor`。

Q-Former 输出的归一化、逐文本 max-query 相似度与梯度路由；不包含 Transformer 或冻结大模型。

## 2. 核心公式

$$
S_{ij}=\frac{1}{\tau}\max_{q}\hat z_{iq}^{T}\hat t_j.
$$

## 3. 公式到代码

先求 [B,Q,N] 的归一化相似度，再对 Q 归约。每个文本各自选出匹配最好的 query；先平均 query 会破坏这种多语义槽结构。最大值非并列时，梯度只流向获胜 query，随后经过归一化的雅可比；测试避开并列点做中心差分。

## 4. 本题约定

queries 为 [B,Q,D]，texts 为 [N,D]，temperature 为正标量；沿 D 归一化，返回 [B,N]，每个图文对独立沿 Q 取最大值。最大值并列时采用 torch.max 的首个索引梯度；不实现完整 Q-Former。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/blip_2 --impl problems/multimodal/blip_2/solution.py
paperleet mutate multimodal/blip_2
paperleet vizdata multimodal/blip_2
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 3.1: image-text contrastive learning takes the maximum similarity over query outputs

- [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)
