# DDIM 跳步更新

## 1. 论文解决什么问题

DDPM 的训练目标只要求各时间的边缘加噪分布，并没有把采样轨迹唯一固定。DDIM 利用这一自由度复用已有噪声预测器，以更少的去噪步骤生成样本。

本模块只要求实现 `ddim_step`，直接抓住论文的跳步更新公式。时间子序列和完整采样循环已经提供；反演辅助代码与可视化保留为可选探索。

## 2. 核心公式

### 返回值要求

#### `ddim_step`（需实现）

返回跳到较干净时间的样本 Tensor，与 `x_t` 同形状 `[B,...]`、浮点 dtype/device；B 为批大小，末尾为任意样本维度。不返回 x0 预测或中间系数。`eta=0` 时与 noise 无关；`ab_prev=1` 时返回干净端点。函数本身不停止梯度。

#### `ddim_timesteps`（已提供）

返回形状 `[num_steps]` 的 `torch.long` Tensor（本题默认 CPU），元素为从 `T-1` 开始严格递减的离散时间索引。每个元素对应一次模型调用；最终干净边界不额外占一个元素，也不必包含索引 0。输出不带梯度。

#### `ddim_sample`（已提供）

`return_traj=False` 时返回最终样本 Tensor `[B,...]`，与 `x_T` 同形状、dtype/device。为 True 时按顺序返回二元组 `(samples, trajectory)`，trajectory 为 `[num_steps+1,B,...]` Tensor；第 0 项是初始 x_T 的副本，之后每项是一次更新，最后一项等于 samples。两者均不带梯度；不修改 x_T。

#### `ddim_invert`（已提供）

返回反演后的噪声端 Tensor，与 `x0` 同形状 `[B,...]`、dtype/device，终点对应所选序列的最大索引 `T-1`。只返回终点，不返回轨迹；不修改 x0，输出不带梯度。离散反演是近似，不承诺与采样逐位可逆。

### 公式与接口说明

设当前累计信号为 $a=\bar\alpha_t$，前一个采样时间的累计信号为 $b=\bar\alpha_{\mathrm{prev}}$，因此 $b\ge a$。先恢复干净预测，再组合确定性方向与随机噪声：

$$\hat x_0=\frac{x_t-\sqrt{1-a}\hat\epsilon}{\sqrt a},\qquad\sigma=\eta\sqrt{\frac{1-b}{1-a}}\sqrt{1-\frac ab}.$$

$$x_{\mathrm{prev}}=\sqrt b\hat x_0+\sqrt{1-b-\sigma^2}\hat\epsilon+\sigma z.$$

所有样本张量为 `[B,D]`；a、b 可以是标量，也可广播到样本形状。eta 等于零时，每一条轨迹由初始噪声唯一确定；eta 等于一且相邻时间步时，随机方差与 DDPM 后验相同。这里必须从方向项中扣掉随机项的方差，否则总噪声会过量。更新和 eta 参数化分别见 [DDIM 式 12、16](https://arxiv.org/abs/2010.02502)。

最后一次更新设置 b=1，把样本送到真正的干净端点。单点分布是很强的检查：完美 x0 预测下，无论采样一步还是五十步，都应精确到达该点。

## 3. 从公式到代码

只实现 `ddim_step(x_t, eps_pred, ab_t, ab_prev, eta, noise)`：

```text
用当前 ab_t 与 epsilon 预测恢复 x0
从当前和目标累计系数计算 sigma
从方向项方差中扣除 sigma²
组合 x0 预测、epsilon 方向和调用者提供的噪声
```

返回与 `x_t` 同形状、同 dtype 的张量。`ab_t` 与 `ab_prev` 可广播到样本；最终干净端点使用 `ab_prev=1`，此时方向项和随机项都消失。`eta=0` 时输出与传入噪声无关。

`ddim_timesteps` 和 `ddim_sample` 已提供。采样循环会对每一对时间调用你的 `ddim_step`，并可返回包含起点与终点的 `[n+1,B,D]` 轨迹。`ddim_invert` 也已提供，用于可选的往返实验，无需自行实现。

## 4. 本题约定

| 项目 | 本题 | diffusers DDIMScheduler |
| --- | --- | --- |
| trailing 步距 | T 整除 n | 实数 T/n 后取整 |
| T 能整除 n 时 | 两者相同 | 两者相同 |
| 其他选项 | 仅固定 trailing | 另有 leading、linspace |
| 干净边界 | 最后 b=1 | set_alpha_to_one 控制 |

表格按 [官方调度器源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) 于 2026-10-13 核对。非整除情况优先遵循本题定义。

## 5. 交互可视化

<viz id="eta_steps_grid" />

点击 eta 与步数组合，播放同一初始噪声的轨迹并比较最终 MMD，观察随机性与计算预算的关系。

<viz id="inversion_demo" />

选择数据点与步数，查看反演噪声、重建位置及往返误差；网格更密时通常更接近原样本。

## 6. 怎么运行

在浏览器中打开模块 **9**，只补全 `ddim_step` 后运行测试。

核心检查覆盖确定性、单点干净端点和后验方差；集成检查使用提供的辅助函数验证时间子序列、反演与收敛。T3 通过八高斯 oracle 检查五十步生成质量，并记录 eta=1 的对照数值。

## 7. 延伸

提供的反演是一阶近似：从干净端点出发时首次使用索引零，之后在当前位置和时间求 epsilon。有限步数下，它通常不是非线性采样映射的精确代数逆。可用反演可视化比较不同步数的往返误差；这部分无需补全代码。少步误差还可参考 [Progressive Distillation](https://arxiv.org/abs/2202.00512)。
