# Chameleon 逐头 QK LayerNorm

## 1. 问题

实现 `qk_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor`。

逐头 LayerNorm 的统计轴、总体方差和仿射参数；用注意力熵实验观察输入范数增长下的稳定性。

## 2. 核心公式

$$
\mu_{th}=D^{-1}\sum_d x_{thd},\quad \sigma^2_{th}=D^{-1}\sum_d(x_{thd}-\mu_{th})^2,\quad y_{thd}=\gamma_{hd}\frac{x_{thd}-\mu_{th}}{\sqrt{\sigma^2_{th}+\epsilon}}+\beta_{hd}.
$$

## 3. 公式到代码

每个 token 的每个 head 单独统计，不能跨 token 或 head 归约，也不能用不去均值的 RMSNorm 代替。方差是总体方差，分母为 D。先除以 sqrt(var+eps)，再施加逐头仿射参数；常量 head 的归一化结果为零，因此输出应等于 bias。对范数放大的查询和键应用此核后，注意力分布不应迅速饱和。

## 4. 本题约定

x=[T,H,D]，weight/bias=[H,D]；沿最后一维 D 计算每个 token/head 的均值和总体方差，eps>0 加在开方内。权重偏置逐头独立。输出保留全部输入梯度。此核可分别用于 query 和 key；不实现 RoPE 或完整多模态生成。

CPU float32/float64；不得修改输入；仅接受有限浮点值。测试包含独立 oracle、边界、适用的梯度检查，以及被测函数参与的实验。

## 5. 交互图

确定性 CPU 实验调用被测函数；图表展示实际优化、检索或分组行为，请结合 metrics 阅读。合成实验不代表论文的完整基准结果。

## 6. 运行方式

```bash
paperleet test multimodal/chameleon --impl problems/multimodal/chameleon/solution.py
paperleet mutate multimodal/chameleon
paperleet vizdata multimodal/chameleon
```

## 7. 延伸

阅读下面来源的对应机制，再将独立核扩展到论文完整架构和数据管线。本题使用上面明确列出的可判题范围与教学简化。

Section 2.3 Stability; ChameleonLayerNorm computes statistics only over head_dim

- [https://arxiv.org/abs/2405.09818](https://arxiv.org/abs/2405.09818)
- [https://github.com/huggingface/transformers/blob/main/src/transformers/models/chameleon/modeling_chameleon.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/chameleon/modeling_chameleon.py)
