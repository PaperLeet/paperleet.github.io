# DRaFT-LV 重加噪奖励梯度平均

## 1. 问题

复用生成样本，通过多次重加噪和单步去噪构造低方差奖励梯度。

实现 DRaFT-LV 额外样本的停止梯度与平均；初始生成过程由调用者提供。

## 2. 核心公式

$$
z_k=\alpha\operatorname{sg}(x_0)+\sigma\operatorname{sg}(\epsilon_k),\quad \hat x_k=(z_k-\sigma\epsilon_\theta(z_k))/\alpha,\quad L=-\frac{r(x_0)+\sum_{k=1}^K r(\hat x_k)}{K+1}\text{ averaged over batch}.
$$

## 3. 公式到代码

```python
draft_lv_loss(generated: Tensor, noises: Tensor, alpha: float, sigma: float, denoiser, reward) -> Tensor
```

保留原始生成样本的奖励梯度。构造每个额外噪声样本时冻结原始图，再用可训练去噪器恢复样本并计算奖励。所有奖励先按样本求均值，再按原始与额外分支平均。

## 4. 本题约定

generated (B,D), noises (K,B,D), K>=1; alpha>0 and sigma>=0 amplitudes. denoiser(x) returns (B,D); reward(x) returns (B,). Include original reward and K reconstructed rewards, averaged over K+1 and B.

Original generated samples retain their reward gradient. Re-noising uses generated.detach() and frozen noises; additional gradients reach only the denoiser. Reward parameters are supplied/frozen by caller but input gradients remain live.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A scalar generator and epsilon denoiser share one parameter; two fixed symmetric re-noising draws provide extra differentiable quadratic rewards over 55 updates.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/draft --impl problems/diffusion_rl/draft/solution.py
python -m paperleet.cli mutate diffusion_rl/draft
```

## 7. 延伸

在固定采样预算下比较梯度样本方差。

[Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400) · Section 4, DRaFT-LV and Algorithm 1 re-noising lines
