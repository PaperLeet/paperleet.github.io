# 解耦 AdamW 更新

对于自适应优化器，把 L2 正则项加进梯度与直接缩小参数，是两种不同操作。[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) 的算法 2 将权重衰减与梯度更新分离。作者实验代码见 [AdamW-and-SGDW](https://github.com/loshchil/AdamW-and-SGDW)。

## 核心任务

### 返回值要求

#### `adamw_step` — 需要实现

按固定顺序返回三个浮点 Tensor 的 Python 元组 `(new_parameter, new_first_moment, new_second_moment)`，每项均保持输入的完整形状/dtype/device。返回的两项 moment 是供下一次调用使用的更新后未做偏差校正的累计量，不是校正后的 moment。此接口为函数式更新：返回新值，不修改 parameter、gradient 或输入 moments。保留 Tensor autograd 路径，由调用方决定是否关闭梯度；不能像原地优化器 step 一样返回 None。

### 公式与约定

实现一个纯函数 `adamw_step`：接收参数、梯度、旧一阶与二阶动量，以及**从 1 开始**的步数，返回新参数和两个新动量。状态初始化与二维优化循环由实验提供。

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$
$$
\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t),\qquad
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

**本题约定：** `lr = eta`、`weight_decay = lambda` 使用常见的 PyTorch AdamW 参数化方式。论文分别表示基础步长、衰减系数与调度乘数，本题把学习率调度折叠进 `lr`。衰减乘以学习率，epsilon 放在平方根**外面**。

| 项目 | 约定 |
|---|---|
| 张量参数 | 相同形状、精度与设备，二阶动量非负 |
| 状态 | 两个动量从零开始，不修改传入张量 |
| 步数 | 从 1 开始的正整数 |
| 返回 | `(新参数, 新一阶动量, 新二阶动量)` |
| 自动求导 | 纯函数保留张量梯度路径，由调用方决定是否使用 `no_grad` |
| 范围 | 标准 AdamW，不含 AMSGrad、梯度裁剪、maximize 或稀疏梯度 |

权重衰减不进入梯度和动量。梯度与历史都为零时，更新应当恰好是 `parameter * (1 - lr * weight_decay)`。

## 小例子

参数为 2、梯度为 3、`lr=.1`、`weight_decay=.2`、动量初始为零。第一步校正后的动量为 3 与 9。忽略很小的 epsilon，新参数为 `2*.98 - .1 = 1.86`。虽然首步校正值与 beta 无关，返回的原始动量仍依赖 beta。

## Toy 实验

提供的 200 步实验最小化一个两个方向曲率不同的二次函数。它持续调用用户提交的 `adamw_step`，记录参数轨迹与目标值。目标为 `[.25,-.5]`，曲率为 `[1,16]`，第二个方向的梯度更陡。

**来源：** PaperLeet 根据更新规则设计的教学改编，演示合成数据上的优化行为，不是原论文图像分类或泛化实验。全部在 CPU 上运行，无需下载。确定性最终 loss 阈值为 `.01`；单独标记的参考预览不是用户结果。

## 验证

T0 验证纯函数接口与输入保留。T1 手算首步，并用极小梯度检查 epsilon 位置。T2 检查衰减的 Jacobian、动量与衰减独立、多步与 `torch.optim.AdamW` 一致性及坐标排列性质。T3 验证二维优化并生成轨迹。错误变体覆盖耦合衰减、缺失偏差校正、epsilon 位置错误和衰减漏乘学习率。
