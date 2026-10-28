# Flamingo 媒体因果门控注意力

## 1. 问题

实现 `gated_media_attention(x: Tensor, keys: Tensor, values: Tensor, text_media: Tensor, visual_media: Tensor, gate: Tensor) -> Tensor`。

最近媒体掩码、无媒体行的安全 softmax 和零初始化 tanh 门控；简化为单头恒等投影的交叉注意力。

## 2. 核心公式

$$
y_i=x_i+\tanh(g)\sum_{j:m_j=t_i>0}\operatorname{softmax}_{j}(x_i^Tk_j/\sqrt D)v_j.
$$

## 3. 公式到代码

media ID 表示最近出现的媒体，因此使用相等比较而非小于等于。没有可见媒体的行不能对全负无穷做 softmax；先替换成安全行，再用 mask 把权重归零。门控乘在视觉残差上，tanh(0)=0 保证初始文本路径为恒等，而 tanh 在零处的导数为一，仍可学到视觉信息。

## 4. 本题约定

x=[T,D]，keys/values=[N,D]，text_media=[T] 与 visual_media=[N] 为 int64。媒体从 1 编号；text_media 给出文本前最近媒体，0 表示尚无媒体。仅相同媒体 ID 可见，非匹配行输出严格为 x。gate 是标量，残差使用 tanh(gate)。使用单头、恒等投影，不含 Perceiver 和 FFN。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/flamingo --impl problems/multimodal/flamingo/solution.py
paperleet mutate multimodal/flamingo
paperleet vizdata multimodal/flamingo
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 2.2 and Appendix A: single-image causal masking and tanh-gated cross-attention

- [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)
