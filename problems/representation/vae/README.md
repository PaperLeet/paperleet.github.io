# VAE：重参数化与 Bernoulli ELBO

## 1. 论文解决什么问题

Auto-Encoding Variational Bayes 联合学习概率解码器和近似后验，通过重参数化让采样结果对后验参数可微。本题选择[第 3 节式 (9)–(10)、附录 B 与 C.1](https://arxiv.org/html/1312.6114v11#S3) 的对角高斯后验与 Bernoulli 观测模型。

## 2. 核心公式与返回值要求

令 `logvar = log(sigma²)`，标准正态噪声由调用者提供：

$$z=\mu+\exp(\tfrac12\operatorname{logvar})\odot\epsilon,$$

$$K_b=\tfrac12\sum_j[\mu_{bj}^2+\exp(\operatorname{logvar}_{bj})-1-\operatorname{logvar}_{bj}],$$

$$L=\frac1B\sum_b\left[\sum_f(\operatorname{softplus}(\ell_{bf})-x_{bf}\ell_{bf})+K_b\right].$$

- `reparameterize(mu, logvar, noise)` 返回 `[B,Z]`。三个输入形状相同，`Z>0`，函数内部不能重新抽样。
- `gaussian_kl(mu, logvar)` 返回对 `N(0,I)` 的**非负 KL（损失中取正号）**，形状 `[B]`，对潜变量维度求和。
- `negative_elbo(logits, target, mu, logvar)` 返回以 nats／样本为单位的标量。logits 和 target 为非空 `[B,F]`，后验的 batch 大小一致，target 属于 `[0,1]`。

保留一致的 float32／float64、device、输入值及可微路径。采样与 KL 允许 `B=0`，损失要求 `B>0`。形状不兼容抛出 `ValueError`。

## 3. 从公式到代码

采样时先把 log variance 转成**标准差**；解析 KL 对 `Z` 求和。Bernoulli NLL 在 logits 空间稳定计算，对 `F` 求和，每个样本加一次 KL，最后平均 batch。返回值是要最小化的负下界，不是要最大化的 ELBO。

## 4. 本题约定

使用一个后验样本，KL 系数固定为原始目标的 1，不使用 beta、退火或概率裁剪。特征求和与 batch 平均不能混淆，否则会改变重构与 KL 的相对权重。前向数值正确但 detach 了 KL 的实现仍然错误。噪声作为显式输入也保留其可微路径，尽管实验抽到的噪声是常量。

## 5. 交互图与实验

线性编码器输出二维后验的均值和 log variance，线性 Bernoulli 解码器重构两种互补的六位二值模式。固定种子 107，batch 为 32，学习率 0.04 的 Adam 实际训练 120 步。你的三个函数分别参与采样、训练目标和 KL 诊断。

曲线显示采样负 ELBO、重构 NLL 与 KL。最后 20 步平均损失需小于初始损失的 80%，并保持分解一致。这是合成教学实验，不是 MNIST 复现、精确边缘似然测量，也不证明潜变量表示一定有用。

## 6. 编写与运行

补全三个函数后点击**运行测试**，或运行：

```bash
paperleet test representation/vae --impl problems/representation/vae/solution.py
```

测试包括独立标量似然与 KL、解析梯度、极端 logits、归约恒等式、纯采样、形状错误，以及真实编码器／解码器训练。

## 7. 延伸

推导为什么负 ELBO 是负对数似然的上界。比较后验采样与直接使用后验均值，说明优化的估计量如何变化。Bernoulli 观测适合本题二值数据；连续图像需要另行明确观测似然。
