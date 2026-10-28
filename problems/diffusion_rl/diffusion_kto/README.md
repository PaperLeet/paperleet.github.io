# Diffusion-KTO 二元反馈与效用基线

## 1. 问题

用独立样本估计非负 KL 基线，并对喜欢／不喜欢反馈优化 sigmoid 效用。

输入已计算的每步 log-prob；实现 KTO 效用与停止梯度约定，不训练图像骨干。

## 2. 核心公式

$$
b=\max(0,\operatorname{mean}\operatorname{sg}(\ell_{unrelated})),\quad z=w\beta(\ell_\theta-\operatorname{sg}(\ell_{ref})-b),\quad L=-\operatorname{mean}(\sigma(z)-1/2).
$$

## 3. 公式到代码

```python
kto_loss(current: Tensor, reference: Tensor, labels: Tensor, unrelated_log_ratios: Tensor, beta: float) -> Tensor
```

先用无关样本的 log 比率求均值并截断到非负值。喜欢标签保留相对参考的方向，不喜欢标签反转方向。sigmoid 效用以零为中心，外层取负号用于梯度下降。

## 4. 本题约定

current/reference/labels: equal (B,) tensors; labels are exactly +1 or -1. unrelated_log_ratios: nonempty (M,). beta>0. Return mean negative centered sigmoid utility, with baseline max(mean(unrelated),0).

Only current log probabilities receive gradients. Reference, feedback labels and the unrelated-batch baseline are detached.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A scalar policy preference offset is optimized on balanced desirable/undesirable labels for 60 steps with a separate fixed baseline batch.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/diffusion_kto --impl problems/diffusion_rl/diffusion_kto/solution.py
python -m paperleet.cli mutate diffusion_rl/diffusion_kto
```

## 7. 延伸

比较 sigmoid、log-sigmoid 和线性效用的饱和行为。

[Aligning Diffusion Models by Optimizing Human Utility](https://arxiv.org/abs/2404.04465) · Section 4.1 Eqs. (7)-(8), empirical nonnegative unrelated-pair baseline
