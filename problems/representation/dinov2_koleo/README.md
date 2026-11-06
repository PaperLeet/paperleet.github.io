# DINOv2：KoLeo 特征均匀性正则


## 1. 问题


实现 `nearest_neighbors` 与 `koleo_loss`，学习 DINOv2 采用的特征分散正则。KoLeo 增大归一化表示的最近邻距离。DINOv2 还包含 DINO/iBOT 类训练、数据整理与训练规模改进，本题只提取其中一个组件；KoLeo 早于 DINOv2，并非其首次提出。


## 2. 核心公式


先令 $z_i=x_i/\max(\|x_i\|_2,\epsilon)$，再取 $n_i=\arg\max_{j\ne i}z_i^Tz_j$。按官方实现，距离内逐坐标偏移固定为 $\epsilon_p=10^{-8}$；函数参数 `eps` 只控制归一化下界与外层对数：
$$d_i=\sqrt{\sum_a(z_{ia}-z_{n_i a}+\epsilon_p)^2},\quad L=-\frac1B\sum_i\log(d_i+\epsilon).$$
距离中的逐坐标 epsilon 固定为 `1e-8`，与单独构造的 PyTorch `PairwiseDistance` 对象一致；改变参数 `eps` 不会改变此常数，也不能把它挪到普通范数之外。两个端点都接受梯度，邻居索引选择本身是离散操作。


## 3. 公式到代码


归一化每行，在脱离梯度的余弦矩阵上屏蔽对角线并寻找最近邻。从仍有梯度的归一化特征取出邻居，计算加固定逐坐标 `1e-8` 的距离与平均负对数。把取出的邻居 detach 会丢失端点梯度；负号保证优化目标鼓励相近点分开。


## 4. 本题约定


有限 float32/float64 `[B,D]`，B≥2、D≥1；默认 `eps=1e-8`，必须有限且为正。最近邻函数接收已归一化的行并返回 int64 ID。严格排除自身，同分取最小索引。官方以 -1 屏蔽对角线，在精确反向向量同分时可能选中自身；本题明确使用负无穷作为边界约定。保持输入、dtype/device 与非连续梯度路径。零行归一化为零。邻居切换点不可导，梯度测试避开并列最近邻。


非零向量短于 `eps` 时须除以调用者给定的范数下界。独立标量与有限差分检查覆盖这个线性归一化区域。固定距离内 epsilon 的约定在非默认 `eps` 下也与官方代码一致。

## 5. 交互图与 CPU 实验


六个三维向量从相近位置出发，以 `.06` 学习率进行 100 次 SGD，共 18 个可训练值。图中损失来自实际调用待实现函数，平均最近邻欧氏距离由独立标量计算。T3 要求最终距离大于 1 且超过初值五倍，最终损失小于零。这只演示 KoLeo，不训练 DINOv2 图像模型，也不证明语义质量或任意网络的防坍塌能力。


## 6. 运行方式


网站点击 **Run tests** 运行完整 CPU/WASM，或用下面命令。检查独立标量距离、两处 epsilon、非自身与同分规则、双端点有限差分、非连续梯度、缩放和置换不变性、重合点与真实优化。


```bash
paperleet test representation/dinov2_koleo --impl problems/representation/dinov2_koleo/solution.py
paperleet mutate representation/dinov2_koleo
paperleet vizdata representation/dinov2_koleo
```



## 7. 延伸与来源


继续学习 DINO 与 iBOT 的教师—学生目标，再比较 VICReg、Barlow Twins 的协方差约束与 KoLeo 的最近邻几何。官方代码将熵正则归于 Sablayrolles 等人的 “Spreading vectors for similarity search”（2018）。数据整理、大规模 ViT 训练与蒸馏属于 DINOv2 的其它内容。


[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) · [Official implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py)
