# PC-Flow 分类器的相邻时刻偏好比率

## 1. 问题

从相邻时刻的分类器 logits 构造参考模型无关的偏好训练目标。

只训练偏好分类器的相邻时刻目标；基座 flow 与引导采样器不在范围内。

## 2. 核心公式

$$
d_w=\log\sigma(a^w_{next})-\log\sigma(a^w_{now}),\quad d_l=\log\sigma(a^l_{next})-\log\sigma(a^l_{now}),\quad L=\operatorname{mean}\operatorname{softplus}(-\beta T(d_w-d_l)).
$$

## 3. 公式到代码

```python
pcflow_loss(winner_next: Tensor, winner_now: Tensor, loser_next: Tensor, loser_now: Tensor, beta: float, total_steps: int) -> Tensor
```

先把 logits 用稳定的 log-sigmoid 转为 log 概率，再用相邻时刻之差构造两条轨迹的偏好比率。差值乘 beta 与总步数 T，最后计算配对 logistic 损失。

## 4. 本题约定

`next` 表示完成一步去噪后的 $t-1$ 状态，`now` 表示去噪前的 $t$ 状态。

winner_next/winner_now/loser_next/loser_now are equal (B,) classifier logits. beta>0, total_steps is positive integer. Classifier probabilities are sigmoid(logit), not raw logits.

All four classifier evaluations differentiate because both adjacent states contribute to each log probability ratio.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A linear classifier learns opposite next-state changes for winner and loser transitions over 50 actual updates.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/pc_flow --impl problems/diffusion_rl/pc_flow/solution.py
python -m paperleet.cli mutate diffusion_rl/pc_flow
```

## 7. 延伸

用训练好的分类器梯度实现原论文的引导采样更新。

[PC-Flow: Preference Alignment in Flow Matching via Classifier](https://ojs.aaai.org/index.php/AAAI/article/view/37971) · Preference Classifier Training Objective, Eq. (10)
