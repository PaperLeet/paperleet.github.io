# Chameleon 逐头 QK LayerNorm — 题解

每个 token 的每个 head 单独统计，不能跨 token 或 head 归约，也不能用不去均值的 RMSNorm 代替。方差是总体方差，分母为 D。先除以 sqrt(var+eps)，再施加逐头仿射参数；常量 head 的归一化结果为零，因此输出应等于 bias。对范数放大的查询和键应用此核后，注意力分布不应迅速饱和。

$$
\mu_{th}=D^{-1}\sum_d x_{thd},\quad \sigma^2_{th}=D^{-1}\sum_d(x_{thd}-\mu_{th})^2,\quad y_{thd}=\gamma_{hd}\frac{x_{thd}-\mu_{th}}{\sqrt{\sigma^2_{th}+\epsilon}}+\beta_{hd}.
$$

参考实现按上述布局和归约逐步计算：

```python
def qk_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    _floating(x,weight,bias)
    if x.ndim!=3 or x.numel()==0 or weight.shape!=x.shape[1:] or bias.shape!=weight.shape or eps<=0:
        raise ValueError("expected (T,H,D), (H,D) affine parameters, positive eps")
    centered=x-x.mean(dim=-1,keepdim=True)
    variance=centered.square().mean(dim=-1,keepdim=True)
    return centered/torch.sqrt(variance+eps)*weight+bias
```

错误实现检测覆盖这些常见问题：

- Omits mean centering
- Uses sample variance instead of population variance

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

逐头 LayerNorm 的统计轴、总体方差和仿射参数；用注意力熵实验观察输入范数增长下的稳定性。

- [https://arxiv.org/abs/2405.09818](https://arxiv.org/abs/2405.09818)
- [https://github.com/huggingface/transformers/blob/main/src/transformers/models/chameleon/modeling_chameleon.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/chameleon/modeling_chameleon.py)
