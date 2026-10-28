# D3PO 逐去噪步骤偏好优化

## 1. 问题

把终点偏好传到各去噪步骤，并在每一步计算参考策略校正的偏好损失。

实现给定轨迹 log-prob 的逐步 D3PO 损失；不把多步概率先相乘。

## 2. 核心公式

$$
z_{bt}=\beta[(\ell^w_{bt}-\ell^{w,ref}_{bt})-(\ell^l_{bt}-\ell^{l,ref}_{bt})],\qquad L=\frac1{BT}\sum_{bt}\operatorname{softplus}(-z_{bt}).
$$

## 3. 公式到代码

```python
d3po_loss(preferred: Tensor, rejected: Tensor, ref_preferred: Tensor, ref_rejected: Tensor, beta: float) -> Tensor
```

先分别计算胜出和落选轨迹相对参考策略的 log 比率，再在相同时间步相减。每个时间步有独立的 sigmoid 分类任务，最后才对批次与时间求均值。

## 4. 本题约定

preferred/rejected/ref_preferred/ref_rejected: equal (B,T) finite log probabilities. beta>0. Return a scalar mean over B and T after the per-step softplus.

Both current preferred and rejected log probabilities differentiate; both reference tensors are detached.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A three-step preference model learns independently signed transition margins through 60 gradient updates.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/d3po --impl problems/diffusion_rl/d3po/solution.py
python -m paperleet.cli mutate diffusion_rl/d3po
```

## 7. 延伸

与对整条轨迹求和后再分类的目标比较梯度。

[Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model](https://arxiv.org/abs/2311.13231) · Section 4.3 Eq. (14): a preference objective for every state-action pair
