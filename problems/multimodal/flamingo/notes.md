# Flamingo 媒体因果门控注意力 — 题解

media ID 表示最近出现的媒体，因此使用相等比较而非小于等于。没有可见媒体的行不能对全负无穷做 softmax；先替换成安全行，再用 mask 把权重归零。门控乘在视觉残差上，tanh(0)=0 保证初始文本路径为恒等，而 tanh 在零处的导数为一，仍可学到视觉信息。

$$
y_i=x_i+\tanh(g)\sum_{j:m_j=t_i>0}\operatorname{softmax}_{j}(x_i^Tk_j/\sqrt D)v_j.
$$

参考实现按上述布局和归约逐步计算：

```python
def gated_media_attention(x: Tensor, keys: Tensor, values: Tensor, text_media: Tensor, visual_media: Tensor, gate: Tensor) -> Tensor:
    _floating(x, keys, values, gate)
    if x.ndim!=2 or keys.ndim!=2 or x.numel()==0 or keys.numel()==0 or values.shape!=keys.shape or x.shape[1]!=keys.shape[1] or text_media.shape!=(x.shape[0],) or visual_media.shape!=(keys.shape[0],) or gate.ndim!=0:
        raise ValueError("incompatible attention shapes")
    if text_media.dtype!=torch.long or visual_media.dtype!=torch.long or (text_media<0).any() or (visual_media<=0).any(): raise ValueError("invalid media IDs")
    mask=(text_media[:,None]==visual_media[None,:]) & (text_media[:,None]>0)
    scores=(x@keys.T)/math.sqrt(x.shape[-1])
    masked=scores.masked_fill(~mask,float('-inf'))
    safe=torch.where(mask.any(-1,keepdim=True),masked,torch.zeros_like(scores))
    weights=torch.softmax(safe,-1)*mask.to(x.dtype)
    return x+torch.tanh(gate)*(weights@values)
```

错误实现检测覆盖这些常见问题：

- Attends all earlier media
- Omits tanh bound

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

最近媒体掩码、无媒体行的安全 softmax 和零初始化 tanh 门控；简化为单头恒等投影的交叉注意力。

- [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)
