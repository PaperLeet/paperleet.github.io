# ImageReward 排序学习与 ReFL 单步反馈

## 1. 问题

实现偏好排序损失，以及冻结前缀后的单步干净样本预测。

训练小型奖励评分器并通过冻结奖励函数更新单步噪声预测；不加载 BLIP 或图像解码器。

## 2. 核心公式

$$
L_{rank}=\operatorname{mean}\,\operatorname{softplus}(r_l-r_w),\qquad \hat x_0=(\operatorname{sg}(x_t)-\sigma_t\epsilon_\theta)/\alpha_t.
$$

## 3. 公式到代码

```python
ranking_loss(preferred: Tensor, rejected: Tensor) -> Tensor
```
```python
refl_clean(latent: Tensor, epsilon: Tensor, alpha: float, sigma: float) -> Tensor
```

对每对分数先求差，再用 softplus 保持大幅度输入下的稳定性。单步预测先冻结已有轨迹状态，再按噪声幅度恢复干净样本。奖励网络参数冻结不等于将奖励值 detach。

## 4. 本题约定

ranking_loss accepts equal (B,) preferred/rejected scores. refl_clean accepts equal (B,D) latent/epsilon, alpha and sigma are amplitude coefficients (not cumulative variances), alpha>0, sigma>=0.

Ranking differentiates both scores. ReFL stops the rollout latent and differentiates epsilon; the frozen reward model must still propagate gradients to its input.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

Eight scalar preference pairs train a linear reward scorer; 30 subsequent reward-gradient steps update a one-step epsilon model. This tests feedback mechanics, not image quality.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/imagereward --impl problems/diffusion_rl/imagereward/solution.py
python -m paperleet.cli mutate diffusion_rl/imagereward
```

## 7. 延伸

尝试保持奖励模型参数冻结，同时对图像解码器的输入反传。

[ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) · Section 2.2 Eq. (1); Section 3 Algorithm 1
