# InstructBLIP 指令感知查询块

## 1. 问题

实现 `instruction_query_block(queries: Tensor, instruction: Tensor, visual: Tensor, instruction_mask: Tensor) -> Tensor`。

恒等投影的两级查询自注意力和视觉交叉注意力，验证指令影响与 padding 隔离；不声称复现完整 Q-Former。

## 2. 核心公式

$$
U=Q+\operatorname{softmax}\left(Q[Q;I]^T/\sqrt D+M\right)[Q;I],\quad O=U+\operatorname{softmax}(UV^T/\sqrt D)V.
$$

## 3. 公式到代码

第一层把指令信息写入 query，第二层用这些更新后的 query 检索图像。把指令仅送给后续语言模型会绕过本题机制。padding 仅屏蔽指令键；query 自身始终可见，保证每行 softmax 有定义。两个残差都属于本题明确的缩小块，不能省略；实际模型的投影、归一化与 FFN 留给扩展。

## 4. 本题约定

queries=[Q,D]，instruction=[I,D]，visual=[N,D]；instruction_mask=[I] bool。先让 query 对 query 与有效指令做自注意力，并加残差；再用更新后的 query 对视觉做交叉注意力，再加残差。所有投影为恒等、单头、无 LayerNorm/FFN，是机制教学缩小块；全屏蔽指令仍允许 query 自注意力。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/instructblip --impl problems/multimodal/instructblip/solution.py
paperleet mutate multimodal/instructblip
paperleet vizdata multimodal/instructblip
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 3.2 and Figure 2: instruction-aware visual feature extraction with query-text self-attention

- [https://arxiv.org/abs/2305.06500](https://arxiv.org/abs/2305.06500)
