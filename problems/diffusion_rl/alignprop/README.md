# AlignProp 去噪链的截断奖励反传

## 1. 问题

实现一个可控制梯度起点的 DDIM 采样链，让终点奖励穿过指定的去噪后缀反传。

调用传入的小型 epsilon 模型，练习时间方向和梯度截断；不实现 LoRA 或 checkpoint 内存优化。

## 2. 核心公式

$$
\hat x_0^{(i)}=(x_i-\sigma_i\epsilon_\theta(x_i,i))/\alpha_i,\quad x_{i+1}=\alpha_{i+1}\hat x_0^{(i)}+\sigma_{i+1}\epsilon_\theta(x_i,i).
$$

## 3. 公式到代码

```python
reward_rollout(initial: Tensor, alphas: Tensor, sigmas: Tensor, denoiser, backprop_from: int) -> Tensor
```

按执行顺序遍历调度器，先预测噪声再还原干净样本，再构造下一个噪声级。指定起点之前使用 no_grad；到起点只截断一次状态，后缀内部保留整条链式法则。

## 4. 本题约定

initial (B,D); alphas/sigmas equal vectors of length T+1 in execution order noisy-to-clean, amplitudes alpha>0, sigma>=0. denoiser(x,step) returns same shape. backprop_from in [0,T-1] specifies first differentiable transition. Terminal schedule is caller-supplied.

Initial noise and schedules are frozen. Prefix transitions execute without gradients; suffix transitions retain all recurrent state and model paths.

## 5. 交互图与实验

实验使用你提交的函数。图中各点来自实际运行结果。

A four-step DDIM chain with a scalar epsilon bias is trained for 45 updates on a quadratic terminal target; every rollout uses the submitted truncation routine.

## 6. 运行方式

```bash
python -m paperleet.cli test diffusion_rl/alignprop --impl problems/diffusion_rl/alignprop/solution.py
python -m paperleet.cli mutate diffusion_rl/alignprop
```

## 7. 延伸

对不同后缀长度实测梯度偏差与存储需求。

[Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2310.03739) · Method: reward backpropagation; official sd_pipeline.py backprop_timestep
