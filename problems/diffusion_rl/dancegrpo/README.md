# DanceGRPO 共享初噪与组内奖励

## 1. 问题

构造同 prompt 共享的初始噪声、逐奖励通道组内优势，以及逐转移 GRPO 目标。

在小型高斯生成策略中检验多奖励与共享初始条件；不运行视频骨干或视觉奖励模型。

## 2. 核心公式

$$
A_{pg}=\sum_k\frac{R_{pgk}-\mu_{pk}}{\sqrt{\operatorname{mean}_g(R_{pgk}-\mu_{pk})^2}+\varepsilon},\quad L=-\operatorname{mean}_{p,g,t}\min(e^{l_{pgt}}A_{pg},\operatorname{clip}(e^{l_{pgt}},1-c,1+c)A_{pg}).
$$

## 3. 公式到代码

```python
shared_noise(base: Tensor, group_size: int) -> Tensor
```
```python
group_advantages(rewards: Tensor, eps: float=1e-8) -> Tensor
```
```python
dance_loss(log_ratios: Tensor, advantages: Tensor, clip: float) -> Tensor
```

把每个 prompt 的一份初噪复制到组内所有样本。每种奖励分别在组内中心化和缩放，再把通道优势相加。终点优势广播到所有转移，用 clipped 目标更新策略。

## 4. 本题约定

shared_noise accepts base (P,D) and group_size>=2, returns (P,G,D). group_advantages accepts rewards (P,G,K), G>=2; population variance plus eps outside sqrt, sum K. dance_loss log_ratios (P,G,T), advantages (P,G), clip in (0,1), abs(log_ratio)<=40.

Noise and reward-derived advantages are frozen. Only current log ratios differentiate in the policy loss; each prompt is normalized independently.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

Two prompt groups share initial states but draw independent Gaussian actions. Two differently scaled quadratic reward channels train a scalar conditional policy for 70 updates.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/dancegrpo --impl problems/diffusion_rl/dancegrpo/solution.py
python -m paperleet.cli mutate diffusion_rl/dancegrpo
```

## 7. 延伸

比较同 prompt 共享与独立初噪下的组内奖励方差。

[DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818) · Section 2.2 Eqs. (9)-(10), Initialization Noise; Section 2.3 combined rewards
