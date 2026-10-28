# BLIP 动量软目标混合 — 题解

先在包含队列负例的列维度归一化 teacher，再混合软目标和明确指定的正样本 one-hot。每行目标的和仍为一；因此学生导数为 (softmax(logits)-targets)/B。alpha=0 退化为硬标签，alpha=1 仅蒸馏。动量教师的更新属于外部训练环节，不能让本损失向教师回传。

$$
q_{ij}=\alpha\,\operatorname{stopgrad}(\operatorname{softmax}(s^m_i)_j)+(1-\alpha)\mathbf 1[j=p_i],\quad L=-\frac1B\sum_{i,j}q_{ij}\log\operatorname{softmax}(s_i)_j.
$$

参考实现按上述布局和归约逐步计算：

```python
def momentum_target_loss(logits: Tensor, teacher_logits: Tensor, positives: Tensor, alpha: float) -> Tensor:
    _floating(logits, teacher_logits)
    if logits.ndim!=2 or logits.numel()==0 or logits.shape!=teacher_logits.shape or positives.shape!=(logits.shape[0],) or positives.dtype!=torch.long or not 0<=alpha<=1:
        raise ValueError("expected (B,N) logits, (B,) long positives and alpha in [0,1]")
    if (positives<0).any() or (positives>=logits.shape[1]).any(): raise ValueError("positive index out of range")
    hard=(torch.arange(logits.shape[1],device=logits.device)[None,:]==positives[:,None]).to(logits.dtype)
    targets=alpha*torch.softmax(teacher_logits.detach(),-1)+(1-alpha)*hard
    return -(targets*F.log_softmax(logits,-1)).sum(-1).mean()
```

错误实现检测覆盖这些常见问题：

- Reverses teacher and hard-label weights
- Updates momentum teacher through targets

验证：T0 检查契约；T1 使用独立 oracle；T2 检查边界、不变量和有定义的梯度；T3 用被测函数运行确定性合成实验。完整输入、更新规则、步数和指标见 experiment.py，门限在 T3 测试中明确断言。

实现 BLIP 一个检索方向的动量软标签蒸馏；支持非方形负样本队列，不实现队列更新、动量编码器或 CapFilt 数据生成。

- [https://arxiv.org/abs/2201.12086](https://arxiv.org/abs/2201.12086)
