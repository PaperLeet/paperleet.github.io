# 后验与反向去噪一步

## 1. 论文解决什么问题

DDPM 将生成过程拆成许多小的去噪步骤。训练时可解析的高斯后验提供学习目标，采样时网络预测的噪声决定每一步往哪里走。

这是 DDPM 论文的第二个核心模块。只需实现 `q_posterior` 与 `p_sample_step`，把解析后验和噪声预测接成一次反向更新。完整采样循环 `ddpm_sample` 与测试使用的噪声预测器已经提供。

## 2. 核心公式

### 返回值要求

#### `q_posterior`（需实现）

按顺序返回二元组 `(mean, variance)`，均为 Tensor。mean 与 `x_t` 同形状 `[B,...]`；variance 为 `[B,1,...,1]`，与 x_t 同阶，每个非 batch 轴长度为 1，可广播到样本。B 为批大小；variance 是方差，不是标准差或 log variance。两者使用 x_t 的浮点 dtype/device，保留梯度；索引 t=0 的方差为零。

#### `p_sample_step`（需实现）

返回一次反向更新后的 Tensor，与 `x_t` 同形状 `[B,...]`、浮点 dtype/device；B 为样本数，末尾为坐标维度。只返回样本，不返回后验统计。逐样本处理 t=0：此时任何方差选项均不加入噪声。函数本身不停止梯度。

#### `ddpm_sample`（已提供）

`return_traj=False` 时返回形状为传入 `shape=(B,...)` 的最终 Tensor，dtype/device 来自 `betas`。为 True 时按顺序返回 `(samples, trajectory)`，trajectory 是 `[T+1,B,...]` Tensor，T=`len(betas)`；首项是初始高斯噪声，随后记录从 T-1 到 0 的每次更新，末项等于 samples。所有返回张量均不带梯度。

#### 已提供的局部辅助函数

- `_extract`：已提供：返回按 t 提取的系数 Tensor `[B,1,...,1]`，与 x 同阶，每个非 batch 轴长度为 1，B=`len(t)`。dtype/device 转换为 x 的 dtype/device；保留 values 的可微路径，不修改任何输入。

### 公式与接口说明

已知干净数据后，$q(x_{t-1}\mid x_t,x_0)$ 是高斯。它的均值由当前样本与干净数据的加权和组成：

$$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\qquad\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.$$

网络看不到真实 x0，因此先由噪声预测恢复它：

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}},\qquad x_{t-1}=\tilde\mu_t(x_t,\hat x_0)+\sigma_tz.$$

$x_t,\hat\epsilon,z$ 都是 `[B,...]`，t 是 `[B]`，每一行可以选择不同时间。方差是 `[B,1,...]`，由广播扩展到样本。后验及噪声参数化见 [DDPM 式 6、7、11 与算法 2](https://arxiv.org/abs/2006.11239)。

本题将网络换成八高斯混合分布的精确 score，得到 $\epsilon^*=-\sqrt{1-\bar\alpha_t}\nabla\log p_t$。因此无需训练，也能检验采样结果是否恢复原分布。输出仍有有限步离散化和有限样本误差，不能期待经验 MMD 恰好为零。

## 3. 从公式到代码

实现两个核心函数：

```text
q_posterior: 按 [B] 时间提取系数，组合 x0 与 x_t，返回均值和方差
p_sample_step: 从 epsilon 预测恢复 x0，调用 q_posterior
按 var_type 选择 posterior 或 beta 方差
把 t=0 的噪声贡献置零，得到更新后的样本
```

`q_posterior` 返回均值 `[B,...]` 和可广播的方差 `[B,1,...]`。`p_sample_step` 返回与 `x_t` 同形状、同 dtype 的张量。未知 `var_type` 抛出 `ValueError`。

`_extract` 和 `ddpm_sample` 已提供。后者管理初始高斯噪声、降序时间循环、调用者的 generator 与轨迹记录；每一步都会调用你的 `p_sample_step`。`return_traj=True` 时返回 `(samples, trajectory)`，轨迹 `[T+1,B,...]` 包含初始噪声。

## 4. 本题约定

| 项目 | 本题 | improved-diffusion |
| --- | --- | --- |
| x0 范围 | 二维数据不裁剪 | 图像常裁剪到 [-1,1] |
| 方差选项 | posterior 或 beta | 还支持网络学习方差 |
| 最后一步 | 两种选项均不加噪 | 同样屏蔽最终噪声 |
| 首个 log 方差 | 不计算 log 方差 | 用下一项方差填充后取 log |

## 5. 交互可视化

<viz id="reverse_particles" />

播放反向轨迹，切换 posterior 与 beta 方差，观察粒子从噪声聚集到八个模式；灰色中心表示真实数据位置。

<viz id="variance_compare" />

拖动时间比较 beta 与后验方差，最后一步两者差异最大，但都必须停止注入噪声。运行测试后，结果面板可叠加采样点、真实样本以及 MMD 阈值。

## 6. 怎么运行

在浏览器中打开模块 **8**，补全 `q_posterior` 与 `p_sample_step` 后运行测试。

T2 以高斯精度相加独立推导后验。T3 将你的单步更新接入提供的采样循环，用 1000 步、2000 个样本检查两种方差。MMD 阈值来自固定种子参考值乘 1.5，并检查模式占比和离群点。

## 7. 延伸

下一题 DDIM 保留同一个噪声预测器，却能沿时间子序列跳步。想研究学习方差，可继续阅读 [Improved DDPM](https://arxiv.org/abs/2102.09672)。
