# 全局梯度裁剪：一个范数缩放所有参数

可独立完成的大模型基础练习，无需先做完其他题。配套代码和合成实验已提供。

## 任务与标准约定

gradients 为非空列表，各项是同 dtype/device 的有限 float32/64 CPU 梯度，可有不同形状，含标量和空张量；max_norm 有限且≥0，eps 有限且>0。计算所有参数全部元素的全局 L2 范数；返回（新的梯度列表，裁剪前的标量 total_norm），保持各项 shape/dtype/device。所有项乘同一个 scale=min(1,max_norm/(total_norm+eps))。全空或全零时范数为零；max_norm=0 输出全零。纯函数，不写 .grad、不更新参数、不修改输入张量或列表，不要求高阶梯度。这是范数裁剪，不是逐元素截断；epsilon 的位置为本题明确约定。

```python
clip_gradients(gradients: list[Tensor], max_norm: float, eps: float = 1e-6) -> tuple[list[Tensor], Tensor]
```

$$
N=\sqrt{\sum_i\|g_i\|_2^2},\quad s=\min(1, M/(N+\epsilon)),\quad \tilde g_i=s g_i
$$

## 示例

```python
clip_gradients([torch.tensor([3.]), torch.tensor([4.])], 2.5)
# ([tensor([1.4999997]), tensor([1.9999996])], tensor(5.))
```

## 动手前想一想

先汇总所有平方和，再开根号，只计算一个缩放系数。返回原始范数，不是裁剪后的范数；小梯度不能被放大。

## 测试与实验

T0 检查接口；T1 对齐独立基准；T2 检查边界、不变量、适用梯度和输入保护；T3 在配套实验中调用你的函数。图表来自实际运行，不声称大模型训练效果或吞吐提升。

## 参考资料

- [Primary documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html)
