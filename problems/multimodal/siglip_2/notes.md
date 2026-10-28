# SigLIP 2 掩码 Patch 自蒸馏 — 题解

teacher_prob 已是目标分布，不应再次 softmax。先停止 teacher 梯度，再计算每个 patch 的类别交叉熵，最后按 mask 求和并除以选中 patch 数量。未选中位置的学生梯度必须为零；选中位置的导数是 (softmax(student/tau)-teacher)/(M*tau)。这一目标抽取了论文辅助训练的一部分。

$$
L=-\frac{\sum_{b,p}m_{bp}\sum_k\operatorname{stopgrad}(q_{bpk})\log\operatorname{softmax}(z_{bp}/\tau)_k}{\sum_{b,p}m_{bp}}.
$$

参考实现按上述布局和归约逐步计算：

```python
def masked_patch_distillation(student: Tensor, teacher_prob: Tensor, mask: Tensor, temperature: float) -> Tensor:
    _floating(student, teacher_prob)
    if student.ndim != 3 or student.numel()==0 or teacher_prob.shape!=student.shape or mask.shape!=student.shape[:2] or mask.dtype!=torch.bool or not mask.any() or temperature<=0:
        raise ValueError("expected (B,P,K), nonempty boolean mask, and positive temperature")
    if (teacher_prob<0).any() or not torch.allclose(teacher_prob.sum(-1),torch.ones_like(teacher_prob.sum(-1)),atol=1e-6,rtol=1e-6):
        raise ValueError("teacher targets must be probabilities")
    per_patch=-(teacher_prob.detach()*F.log_softmax(student/temperature,dim=-1)).sum(-1)
    return (per_patch*mask.to(student.dtype)).sum()/mask.sum()
```

错误实现检测覆盖这些常见问题：

- Divides by all patches
- Backpropagates through teacher

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

实现 SigLIP 2 辅助掩码预测中的概率蒸馏核；teacher 概率已给定，不实现 EMA 编码器、增强、投影头或完整多目标训练。

- [https://arxiv.org/abs/2502.14786](https://arxiv.org/abs/2502.14786)
