# DPOK 高斯 KL 上界与策略梯度

## 1. 问题

计算逐步高斯 KL，并实现奖励 score 梯度与直接 KL 梯度的组合。

实现论文 Eq. (9) 的高效梯度近似；不声称包含 Eq. (8) 状态访问分布的完整导数。

## 2. 核心公式

$$
K_{bt}=\frac{\|\mu_{bt}-\mu^{ref}_{bt}\|^2}{2s_t^2},\quad L=\frac1B\sum_b[-\alpha\operatorname{sg}(r_b)\sum_t\ell_{bt}+\beta\sum_t K_{bt}].
$$

## 3. 公式到代码

```python
gaussian_kl(current: Tensor, reference: Tensor, std: Tensor) -> Tensor
```
```python
dpok_surrogate(logp: Tensor, rewards: Tensor, kl: Tensor, alpha: float, beta: float) -> Tensor
```

相同协方差的高斯 KL 化简为均值差的加权平方和。时间维求和对应轨迹 KL 上界；batch 维求均值。奖励乘当前 log 密度的 score 项，直接 KL 项保留梯度。

## 4. 本题约定

gaussian_kl: current/reference (B,T,D), shared std (T,)>0; return (B,T). dpok_surrogate: logp/kl (B,T), terminal rewards (B,), alpha>0, beta>=0; sum time and mean batch.

Current Gaussian means and KL penalties differentiate. Reference means, scales and terminal rewards are frozen; supplied logp differentiates. State-visitation gradients omitted as in Eq. (9).

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A Gaussian scalar policy with variance 1 learns a quadratic target under beta=0.4 KL to a zero-mean reference; 70 updates are compared against the analytic regularized optimum 2/(2+beta).

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/dpok --impl problems/diffusion_rl/dpok/solution.py
python -m paperleet.cli mutate diffusion_rl/dpok
```

## 7. 延伸

推导原论文附录 A.3 中被省略的状态分布梯度项。

[DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16381) · Section 4.1 Lemma 4.2 and Eqs. (7)-(9), efficient gradient approximation
