# 前向闭式加噪

## 1. 论文解决什么问题

DDPM 将复杂数据逐步变成高斯噪声，再学习反向恢复数据。前向过程的闭式表达式让训练可以随机挑选任意时间，而不必真正执行之前的每一步。

本模块选取论文式 (4) 的闭式加噪。你只需实现 `q_sample`；线性调度、累计系数和 cosine 对照调度均已提供。Cosine 调度来自另一篇论文 Improved DDPM，仅用于拓展比较。

## 2. 核心公式

### 返回值要求

#### `linear_betas`（已提供）

返回形状 `[T]` 的 `torch.float64` Tensor（本题默认 CPU），每项是一层噪声方差 beta；索引 0..T-1 对应论文时间 1..T。T>=2 时包含 beta_start 与 beta_end 两个端点；T=1 时唯一元素为 beta_start。不是累计 alpha，也不是 Python 列表；输出无梯度。

#### `cosine_betas`（已提供）

返回形状 `[T]` 的 `torch.float64` Tensor（本题默认 CPU），每项是离散余弦调度的一步 beta，并受 max_beta 上限约束。返回 T 项而非 T+1 项，也不返回累计 alpha；输出无梯度。

#### `make_schedule`（已提供）

返回 `dict[str, Tensor]`，必须包含这六个键：`alphas`、`alpha_bars`、`alpha_bars_prev`、`sqrt_alpha_bars`、`sqrt_one_minus_alpha_bars`、`log_snr`。所有值均为 `[T]`、torch.float64、与 betas 相同 device，T=`len(betas)`；alpha_bars_prev 的首项为 1。没有 betas 键，不返回元组。保留可微计算，不修改 betas。

#### `q_sample`（需实现）

返回指定逐样本时间的带噪 Tensor，与 x0 同形状 `[B,...]`、dtype/device；B 为批大小，末尾为任意样本维度。alpha_bars 中选出的系数须转换到 x0 的 dtype/device，不能使 float32 样本变成 float64。只返回样本，保留可微路径，不修改输入。

### 公式与接口说明

单步噪声方差为 $\beta_t$，保留信号的比例为 $\alpha_t=1-\beta_t$，累计信号为 $\bar\alpha_t=\prod_{s=1}^t\alpha_s$。由于每一步噪声独立，累计方差可以合并：

$$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),\qquad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

这里 $x_0,\epsilon$ 是 `[B,...]`，时间 t 为 `[B]`，调度数组为 `[T]`。当数据和噪声均为独立单位高斯，两项方差相加恰好为一。因此数据结构消失时，点云整体尺度仍可能不变。闭式边缘分布见 [DDPM 式 4](https://arxiv.org/abs/2006.11239)。

## 3. 从公式到代码

只实现 `q_sample(x0, t, noise, alpha_bars)`：

```text
按每行的 t 从 alpha_bars 提取累计系数
转到 x0 的 dtype/device，重塑为 [B,1,...,1]
用信号标准差和噪声标准差组合 x0 与 noise
```

返回值与 `x0` 同形状、同 dtype。每个样本可使用不同时间；不要让首行的时间控制整个 batch。调用者直接提供噪声，本函数无需生成随机数。

`linear_betas`、`cosine_betas`、`make_schedule` 已完整实现。你可以查看这些辅助函数来理解调度，但无需修改它们。

## 4. 本题约定

| 项目 | 本题 | 常见实现 |
| --- | --- | --- |
| 下标 | i=0 对应论文 t=1 | 论文通常从 1 开始 |
| 调度精度 | float64 | 训练时常将提取的系数转为 float32 |
| 线性 beta | 固定首尾 1e-4、0.02 | improved-diffusion 会按步数缩放端点 |
| cosine beta | 上限 0.999 | improved-diffusion 同样限制最大 beta |

## 5. 交互可视化

<viz id="schedule_curves" />

拖动步数与偏移 s，对比两种调度的累计信号与 log-SNR。末端数值不为零意味着初始标准高斯只是近似边界。

<viz id="forward_cloud" />

拖动时间，在同一组八高斯数据及噪声上对比结构消失速度，避免把随机样本差异误认为调度效果。

## 6. 怎么运行

在浏览器中打开模块 **6**，补全 `q_sample` 后运行测试。测试会将你的实现与已提供的调度辅助函数一起执行。

测试既检查逐样本时间与形状，也用十万个样本验证方差保持；统计规律能发现遗漏平方根这类形状正确但含义错误的实现。

## 7. 延伸

[Improved DDPM 式 16](https://arxiv.org/abs/2102.09672) 的 cosine 调度是可选比较内容，已在 `cosine_betas` 中提供：

$$f(\tau)=\cos^2\left(\frac{\tau+s}{1+s}\frac\pi2\right),\qquad\beta_i=\min\left(1-\frac{f((i+1)/T)}{f(i/T)},0.999\right).$$

对数信噪比 $\log\bar\alpha-\log(1-\bar\alpha)$ 用于比较调度。完成前向加噪后，可继续同一篇 DDPM 的“后验与反向去噪一步”模块。
