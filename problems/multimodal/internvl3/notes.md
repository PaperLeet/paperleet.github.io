# InternVL3 可变视觉位置增量 — 题解

递推依赖当前 token 的模态，等价于所有增量的前缀和再减一。同一图像使用同一个 delta，跨图像或文本边界都不能重置累计位置。delta 的梯度由该图像 token 在后续前缀中出现的次数决定。delta=1 应恢复普通 0 到 L-1 的位置；本题首 token 若是视觉，则位置是 delta-1，这是显式原点约定。

$$
p_i=-1+\sum_{j=0}^{i}\left[(1-v_j)+v_j\delta_{\operatorname{image}(j)}\right].
$$

参考实现按上述布局和归约逐步计算：

```python
def variable_visual_positions(visual: Tensor, image_ids: Tensor, deltas: Tensor) -> Tensor:
    _floating(deltas)
    if visual.ndim!=1 or visual.dtype!=torch.bool or image_ids.shape!=visual.shape or image_ids.dtype!=torch.long or deltas.ndim!=1 or (deltas<=0).any() or (deltas>1).any():
        raise ValueError("invalid masks, image IDs or deltas")
    if (image_ids[~visual]!=-1).any() or (image_ids[visual]<0).any() or (image_ids[visual]>=deltas.numel()).any():raise ValueError("invalid image assignment")
    increments=[]
    for is_visual,index in zip(visual.tolist(),image_ids.tolist()):
        increments.append(deltas[index] if is_visual else deltas.new_tensor(1.))
    if not increments:return deltas.new_empty((0,))
    return torch.stack(increments).cumsum(0)-1
```

错误实现检测覆盖这些常见问题：

- Uses ordinary integer positions for images
- Starts positions one unit too late

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

每图像可变增量的位置递推、跨模态累计与梯度；原点明确固定为 -1，不实现完整旋转嵌入或 InternVL 训练。

- [https://arxiv.org/abs/2504.10479](https://arxiv.org/abs/2504.10479)
