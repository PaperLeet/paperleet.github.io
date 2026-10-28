# LLaVA 视觉 Token 注入与标签对齐 — 题解

图像占位是打包指令，不是额外保留的 token。将其左右切片与视觉块拼接，标签使用相同切片边界。视觉 embedding 参与上下文计算，但没有对应的文本预测标签，因此置为 -100。concat 的反向传播自然把梯度分配回各来源，删除的占位不会影响输出。

$$
E=[E_{<i};V;E_{>i}],\qquad y=[y_{<i};\underbrace{-100,\ldots,-100}_{P};y_{>i}].
$$

参考实现按上述布局和归约逐步计算：

```python
def pack_visual_tokens(text: Tensor, visual: Tensor, labels: Tensor, image_index: int) -> tuple[Tensor, Tensor]:
    _floating(text,visual)
    if text.ndim!=2 or visual.ndim!=2 or text.numel()==0 or visual.numel()==0 or text.shape[1]!=visual.shape[1] or labels.shape!=(text.shape[0],) or labels.dtype!=torch.long or not 0<=image_index<text.shape[0]:
        raise ValueError("expected nonempty compatible features, long labels and valid image index")
    packed=torch.cat((text[:image_index],visual,text[image_index+1:]),dim=0)
    ignored=torch.full((visual.shape[0],),-100,dtype=labels.dtype,device=labels.device)
    packed_labels=torch.cat((labels[:image_index],ignored,labels[image_index+1:]),dim=0)
    return packed,packed_labels
```

错误实现检测覆盖这些常见问题：

- Keeps the image placeholder in the embedding sequence
- Assigns language targets to image features

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

单图单样本的视觉序列打包与监督标签对齐；输入特征已经投影，不实现视觉编码器、语言模型或训练数据构建。

- [https://arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485)
- [https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py)
