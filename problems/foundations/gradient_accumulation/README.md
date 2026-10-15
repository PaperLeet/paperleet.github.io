# 梯度累积：按有效 token 数加权

可独立完成的大模型基础练习，无需先做完其他题。配套代码和合成实验已提供。

## 任务与标准约定

gradients、token_counts 是等长非空列表。每个梯度对应同一个参数，均为相同 shape/dtype/device 的有限 float32/64 CPU 张量，可为标量或空张量。计数为非负整数，总数>0；第 i 个输入梯度已是其有效 token 的平均梯度。返回 Σ(count_i×gradient_i)/Σcount_i，同 shape/dtype/device；计数为零的批次不贡献。不能等权平均 microbatch、额外除以累积步数、执行优化器更新、修改输入或保留状态。不要求高阶梯度；这里不合并不同参数的梯度。

```python
accumulate_gradients(gradients: list[Tensor], token_counts: list[int]) -> Tensor
```

$$
g=\frac{\sum_i n_i g_i}{\sum_i n_i}
$$

## 示例

```python
accumulate_gradients([torch.tensor([1., 3.]), torch.tensor([5., 7.])], [1, 3])
# tensor([4., 6.])
```

## 动手前想一想

每个输入已经除过自己的 token 数，先乘回计数再求全局平均。零 token 批次不应稀释分母。实验用不等长小批次验证同一次参数更新与完整批次相同。

## 测试与实验

T0 检查接口；T1 对齐独立基准；T2 检查边界、不变量、适用梯度和输入保护；T3 在配套实验中调用你的函数。图表来自实际运行，不声称大模型训练效果或吞吐提升。

## 参考资料

- [Primary documentation](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation)
