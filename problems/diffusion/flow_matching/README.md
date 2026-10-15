# 条件路径与速度回归

## 1. 论文解决什么问题

Flow Matching 学习一个随时间变化的速度场，用常微分方程把噪声搬运到数据。条件训练目标允许直接回归可计算的速度，不必在每次训练时模拟整条生成轨迹。

本模块精选线性条件路径及其 CFM 目标。只需实现 `interpolate`、`cfm_target`、`cfm_loss`；时间采样、Euler 求解器、测试模型和训练循环已经提供。

## 2. 核心公式

### 返回值要求

#### `sample_t`（已提供）

返回时间 Tensor `[B]`，B=`batch`，每个元素严格位于 `(0,1)`，一个样本对应一个时间。使用 PyTorch 默认浮点 dtype（本题 float32）及默认 CPU；没有 dtype/device 输入参数。输出不带梯度；不是标量或 `[B,1]`。

#### `interpolate`（需实现）

返回插值样本 Tensor `[B,...]`，形状与 x0/x1 相同，保留本题匹配浮点输入的 dtype/device 和梯度。B 为批大小，末尾为任意样本维度；t 的 `[B]` 只逐样本广播，不改变输出形状。t=0 返回噪声端，t=1 返回数据端。

#### `cfm_target`（需实现）

返回条件速度 Tensor，与 x0/x1 同形状 `[B,...]`、dtype/device；每个元素对应一个样本坐标，不求和、不取均值。它是速度目标，不是插值样本。函数本身不停止输入的梯度。

#### `cfm_loss`（需实现）

返回形状 `()` 的浮点 Tensor 标量，使用样本/模型输出的 dtype/device。对 batch 及全部非 batch 坐标的速度平方误差取均值，不返回逐样本误差。保留模型训练梯度；本接口没有额外 detach 输入或目标。

#### `euler_sample`（已提供）

`return_traj=False` 时返回 t=1 的样本 Tensor，与 x0 同形状 `[B,...]`、dtype/device。为 True 时按顺序返回 `(samples, trajectory)`，trajectory 为 `[num_steps+1,B,...]` Tensor，首项是 x0 副本，后续项按积分顺序记录更新，末项等于 samples。输出均不带梯度，不修改 x0；模型只在 t<1 的左端点求值。

### 公式与接口说明

本题固定 x0 是噪声，x1 是数据。二者均为 `[B,D]`，时间 t 为 `[B]`，每个样本拥有自己的插值比例：

$$x_t=(1-t)x_0+tx_1,\qquad u=x_1-x_0,\qquad\mathcal L=\operatorname{mean}_{b,d}\left(v_\theta(x_t,t)-u\right)^2.$$

条件速度 u 是一对端点之间的位移，但网络只看见当前位置和时间。同一位置可能由不同端点对产生，因此它学到的是条件速度的平均值。即使网络达到最优，训练损失一般仍不为零；不可约方差不等于实现错误。梯度等价性及线性条件路径见 [Flow Matching 式 5、9、22、23](https://arxiv.org/html/2210.02747v2)。

单点数据是特殊情况：$v^*(x,t)=(\mu-x)/(1-t)$，代入插值即可得到恒定目标速度，因此 oracle loss 为零，Euler 一步也能到达数据点。一般混合分布的边缘轨迹会弯曲，多步求解才有意义。

## 3. 从公式到代码

实现三个围绕同一训练目标的函数：

```text
interpolate: 将 t [B] 广播到 [B,...]，构造噪声到数据的线性路径
cfm_target: 返回这条路径对时间的导数
cfm_loss: 在 interpolate 的结果上调用 model(x,t)
将模型速度与 cfm_target 的元素平方误差取平均
```

`interpolate` 和 `cfm_target` 保持端点张量的形状与 dtype。`cfm_loss` 返回一个标量，平均必须覆盖 batch 和所有坐标，并保留模型参数的梯度。

`sample_t` 与 `euler_sample` 已提供。测试 harness 管理模型、优化器和训练循环，训练时会调用你的 `cfm_loss`。默认时间均匀采样；辅助函数里的 logit-normal 选项供后续比较。

## 4. 本题约定

| 项目 | 本模块约定 |
| --- | --- |
| 端点 | `x0` 是噪声，`x1` 是数据 |
| 时间 | `t` 为 `[B]`，从 0 走向 1 |
| 条件路径 | 采用论文条件 OT 路径在终端噪声为零时的形式 |
| 配对 | 独立噪声与数据配对 |
| 损失归约 | 对 batch 及所有坐标取平均 |
| 测试模型 | `model(x,t)` 返回与 x 同形状的速度 |

由条件目标回归学到的是边缘速度场，无需为每个训练样本求解整条 ODE。

## 5. 交互可视化

<viz id="flow_particles" />

播放粒子流动并拖动时间，查看当前位置的速度场箭头。切换 Euler 步数，比较直线 oracle 与学习到的弯曲轨迹。

<viz id="convention_bug" />

切换反向采样错误，观察同一模型如何生成错误结果；同时查看 loss 曲线，理解 loss 下降不足以证明采样正确。

## 6. 怎么运行

在浏览器中打开模块 **10**，补全 `interpolate`、`cfm_target` 与 `cfm_loss` 后运行测试。

T3 使用提供的训练循环在 CPU 双线程上训练 TimeMLP：Adam、学习率 1e-3、batch 256、2000 步。训练会调用你的损失函数，再用提供的 Euler 求解器生成 2000 个点，检查 MMD、八模式覆盖与离群率。阈值和实测耗时保存在 `problem.yaml`。

## 7. 延伸

提供的 Euler 求解器沿时间递增积分，在左端点求值以避开单点 oracle 的 t=1 奇点：

$$x_{k+1}=x_k+\frac1n v_\theta(x_k,k/n),\quad k=0,\ldots,n-1.$$

可进一步阅读 [Rectified Flow](https://arxiv.org/abs/2209.03003) 的重新耦合，以及 [SD3 式 13](https://arxiv.org/html/2403.03206v1) 的反向时间约定与非均匀时间采样。这些比较属于拓展，不增加本模块的必做函数。
