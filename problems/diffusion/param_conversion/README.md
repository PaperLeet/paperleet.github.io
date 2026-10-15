# v 参数化与逆变换

## 1. 论文解决什么问题

扩散网络可以预测干净数据、加入的噪声，也可以预测速度。Salimans 与 Ho 在少步蒸馏中研究这些参数化：同一个训练样本，换一种预测目标会改变端点稳定性与误差权重。

本模块精选论文附录 D 中支持少步采样的参数化技术。只需实现 `make_v`、`v_to_x0` 和 `v_to_eps`；其余转换辅助函数已经提供。完成本模块表示掌握这组旋转关系，师生模型、蒸馏目标及逐轮减步训练属于论文的其他内容。

## 2. 核心公式

### 返回值要求

#### `eps_to_x0`（已提供）

返回单个由噪声预测恢复的干净数据 Tensor，与 `x_t` 同形状 `[B,...]`；B 为批大小，末尾为样本坐标。alpha/sigma 已按输入约定可广播，输出不增加系数轴。保留本题匹配浮点输入的 dtype/device 和梯度；不返回多种参数化的元组，也不修改输入。

#### `x0_to_eps`（已提供）

返回单个由干净数据预测恢复的噪声 Tensor，与 `x_t` 同形状 `[B,...]`；B 为批大小，末尾为样本坐标。alpha/sigma 已按输入约定可广播，输出不增加系数轴。保留本题匹配浮点输入的 dtype/device 和梯度；不返回多种参数化的元组，也不修改输入。

#### `make_v`（需实现）

返回单个角速度 v Tensor，与 `x0` 同形状 `[B,...]`；B 为批大小，末尾为样本坐标。alpha/sigma 已按输入约定可广播，输出不增加系数轴。保留本题匹配浮点输入的 dtype/device 和梯度；不返回多种参数化的元组，也不修改输入。

#### `v_to_x0`（需实现）

返回单个由 v 恢复的干净数据 Tensor，与 `x_t` 同形状 `[B,...]`；B 为批大小，末尾为样本坐标。alpha/sigma 已按输入约定可广播，输出不增加系数轴。保留本题匹配浮点输入的 dtype/device 和梯度；不返回多种参数化的元组，也不修改输入。

#### `v_to_eps`（需实现）

返回单个由 v 恢复的噪声 Tensor，与 `x_t` 同形状 `[B,...]`；B 为批大小，末尾为样本坐标。alpha/sigma 已按输入约定可广播，输出不增加系数轴。保留本题匹配浮点输入的 dtype/device 和梯度；不返回多种参数化的元组，也不修改输入。

#### `convert`（已提供）

返回 dst 所指定参数化的单个 Tensor，形状与 pred/x_t 相同 `[B,...]`，保留本题匹配浮点输入的 dtype/device 和梯度。src=dst 时直接返回 pred（同一对象），否则返回转换后的预测。不是包含 x0/eps/v 的字典；不修改输入。

### 公式与接口说明

设 $x_0,\epsilon,x_t\in\mathbb R^{B\times D}$，$\alpha=\sqrt{\bar\alpha_t}$、$\sigma=\sqrt{1-\bar\alpha_t}$ 可广播到输入形状，且 $\alpha^2+\sigma^2=1$。本题的 alpha 是标准差系数，和前向加噪模块中单步 alpha 的含义不同。

$$x_t=\alpha x_0+\sigma\epsilon,\qquad v=\alpha\epsilon-\sigma x_0.$$

$$x_0=\alpha x_t-\sigma v,\qquad\epsilon=\sigma x_t+\alpha v.$$

令 $\alpha=\cos\phi,\sigma=\sin\phi$，就能直接求导得到 $v=dx_t/d\phi$。逆变换是二维正交旋转，因而不需要除法。相反，由噪声恢复数据要除以 alpha，由数据恢复噪声要除以 sigma，在各自接近零的端点会放大预测误差。对同一个错误预测，三个表示的误差满足

$$\|\hat v-v\|^2=\frac{\|\hat x_0-x_0\|^2}{\sigma^2}=\frac{\|\hat\epsilon-\epsilon\|^2}{\alpha^2}.$$

这解释了预测目标为何也隐含损失权重。以上旋转关系见 [Progressive Distillation 附录 D，式 26、30](https://arxiv.org/html/2202.00512v2#A4)。

## 3. 从公式到代码

实现三个核心函数：

| 函数 | 输入与输出 | 要点 |
| --- | --- | --- |
| `make_v` | `x0, eps, alpha, sigma` → v | 计算旋转的切向量 |
| `v_to_x0` | `x_t, v, alpha, sigma` → x0 | 直接使用逆旋转 |
| `v_to_eps` | `x_t, v, alpha, sigma` → epsilon | 直接使用逆旋转 |

张量形状为 `[B,...]`，系数可广播到该形状。保持输入 shape 和 dtype，不沿 batch 聚合；两项逆变换均无需除法，因而在纯信号与纯噪声端点保持稳定。

`eps_to_x0`、`x0_to_eps` 与 `convert` 已提供。分母下限 `1e-8`、字符串检查和同参数化直接返回都由这些辅助函数处理。`convert` 会调用你实现的三个核心函数。

## 4. 本题约定

| 项目 | 本题 | 常见调度器接口 |
| --- | --- | --- |
| 输入系数 | 已开方的 alpha、sigma | 常保存 alpha_cumprod，再在内部开方 |
| 输出 | 直接返回 Tensor | diffusers 调度器通常返回 step 输出对象 |
| 端点除法 | clamp 到 1e-8 | 由具体调度与稳定性策略决定 |
| 数据维度 | 任意可广播形状 | 图像通常使用 B×C×H×W |

## 5. 交互可视化

<viz id="param_circle" />

拖动旋转角 phi，观察位置向量与切向量：切向量的符号由旋转方向决定。这里画的是系数平面；高维独立噪声与数据近似正交时，也可以直观理解为样本平面。

<viz id="error_amplification" />

拖动时间，查看 $1/\alpha$ 和 $1/\sigma$ 的放大系数，理解某个参数化在端点更难训练的原因。

## 6. 怎么运行

在浏览器中打开模块 **7**，补全三个 v 参数化函数后运行测试。

测试会通过提供的转换入口检查六种往返关系，并独立检查有限差分、端点和误差权重恒等式。无需实现蒸馏训练流程。

## 7. 延伸

论文的完整方法还包括师生蒸馏与逐轮缩减采样步数，可继续阅读 [Progressive Distillation](https://arxiv.org/abs/2202.00512)。Flow Matching 使用对线性时间求导的另一种速度，与这里对旋转角求导的 v 不应混用。
