# GAN：稳定对抗损失与非饱和生成目标

## 1. 论文解决什么问题

GAN 让判别器区分真实数据与生成数据，同时让生成器学会欺骗判别器。原论文第 3 节在 minimax 目标之外给出了早期训练梯度更强的非饱和目标。本题使用这个明确的生成器变体，把正负号、独立归约和梯度归属落实到 logits。

[原始来源 / Original source](https://arxiv.org/html/1406.2661v1#S3) · Section 3, Equation (1), Algorithm 1 and the non-saturating generator paragraph.

## 2. 核心公式与返回契约

$$L_D=\operatorname{mean}(\operatorname{softplus}(-r))+\operatorname{mean}(\operatorname{softplus}(f)),\qquad L_G=\operatorname{mean}(\operatorname{softplus}(-f)),\quad D=\sigma(\mathrm{logit}).$$

必做函数：`discriminator_loss`, `generator_loss`.

## 3. 从公式到实现

真实样本的目标为 1，假样本的目标为 0；判别器的两个 batch 各自取均值后相加。生成器让假样本 logits 增大。用 softplus 直接表达稳定二元交叉熵，不先计算概率。

## 4. 本题约定

两个函数都接收有限 float32/float64 logits，返回同 dtype/device 的标量，保留所有 logits 的梯度。输入可以是任意形状或标量，但每个输入必须至少含一个元素。D 的 real/fake 形状与样本数可以不同，dtype/device 必须一致。**不额外除以 2**，不做标签平滑。空输入或非法 dtype 抛出 ValueError，不修改输入。

这里 `L_G=-mean(log D(fake))` 是非饱和目标；原始 minimax 的 `mean(log(1-D(fake)))` 也是合法算法，但不符合本题指定目标。损失函数内部不 detach logits。D 步应由调用方在送入判别器前 detach 生成样本；G 步可冻结判别器参数，但必须保留从判别器输入到生成器的梯度。

## 5. 交互实验

真实数据为均值 2、方差 1 的高斯，生成器为 `G(z)=z+shift`，判别器为仿射 logit。提供 240 对 D/G 更新，每个 batch 64，两个 Adam 的 lr=0.03，seed=179，float64。你的两个损失分别驱动真实的反向传播；D 步切断 G，G 步冻结 D。图显示生成位置和两种损失；验收最终位置误差<0.5。这不复现图像质量、FID 或一般 GAN 收敛。

## 6. 运行你的实现

在网页编辑器补全 starter 后运行全部测试；通过后阅读 Notes。也可在仓库根目录运行：

```bash
paperleet test representation/gan --impl problems/representation/gan/solution.py
```

## 7. 延伸探索

在 fake logit=-1000 处比较 minimax 与非饱和目标的梯度。尝试改变 D/G 学习率比，记录振荡而不是假设每一步损失都下降。
