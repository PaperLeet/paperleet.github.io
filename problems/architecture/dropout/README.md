# Dropout：保留掩码与期望保持

## 1. 论文解决什么问题

Dropout 让训练时可用的神经元随机变化。原论文第 10 节也描述了训练时放大保留激活、推理时不缩放的等价约定。本题采用这个 inverted 约定，并把随机掩码作为显式输入，让缩放、端点和梯度都能直接检验。

[原始来源 / Original source](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) · Section 4; Section 10, page 1951: inverted Bernoulli scaling and its mean/variance.

## 2. 核心公式与返回契约

$$q=1-p,\quad M_i\sim\operatorname{Bernoulli}(q),\quad Y_i=X_iM_i/q,\quad\mathbb{E}[Y_i]=X_i,\quad\operatorname{Var}(Y_i)=X_i^2p/q.$$

必做函数：`inverted_dropout`.

## 3. 从公式到实现

对训练中保留的位置乘 `1/(1-p)`，被丢弃的位置输出零。推理直接返回输入；p=0 与 p=1 分开处理，避免除零并保留零梯度的计算图。

## 4. 本题约定

x 为任意形状的有限 float32/float64 张量，可含标量、空维度与非连续布局。**p 是丢弃概率**，为 `[0,1]` 内有限实数；论文部分公式用 p 表示保留概率，两者不能混用。

当 training=True 且 `0<p<1` 时，mask 必须是与 x 同 shape/device 的 bool **保留**掩码；不允许广播，不在函数内采样。True 表示保留并缩放，False 表示丢弃。p=0 或推理时返回与输入相等的值，忽略 mask；即使 p=1，推理仍是恒等。训练 p=1 时返回与 x 计算图连接的全零，梯度也为零，不需要 mask。保持 dtype/device/shape 与输入梯度，不修改 x 或 mask；不承诺新存储。非法概率、dtype 或随机训练掩码抛出 ValueError。

## 5. 交互实验

提供 64×4 的高斯特征、一个固定线性目标与可训练线性权重。你的 Dropout 以 p=0.25 生成训练输入，进行 100 次 Adam 更新（lr=0.06，seed=193，float64）；每步图上展示无 Dropout 推理 MSE。另外实际采样 8192 组掩码，测量均值、方差与解析式的偏差。验收 MSE 小于初始的 20%，最大均值偏差<0.06、方差偏差<0.08。这个教学实验展示函数行为，不声称 Dropout 总会改善泛化。

## 6. 运行你的实现

在网页编辑器补全 starter 后运行全部测试；通过后阅读 Notes。也可在仓库根目录运行：

```bash
paperleet test architecture/dropout --impl problems/architecture/dropout/solution.py
```

## 7. 延伸探索

对一个固定 mask 推导输入 Jacobian，再把 mask 积分掉推导均值与方差。比较原论文的测试时缩放约定，但不要混用两种参数化。
