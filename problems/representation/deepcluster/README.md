# DeepCluster 聚类伪标签学习


## 1. 问题


实现 `normalize_features`、`assign_clusters` 和 `pseudo_label_loss`。DeepCluster 交替进行特征聚类与簇标签预测，让编码器从自己的表示生成监督信号。本题从已完成 PCA/白化的特征开始，提取 L2 归一化、最近簇分配和伪标签训练三个核心步骤。


## 2. 核心公式


对给定特征 $x_i$，令 $z_i=x_i/\max(\|x_i\|_2,\epsilon)$。欧氏 K-means 给出
$$y_i=\arg\min_k\|z_i-c_k\|_2^2,\quad L=\frac1B\sum_i[\operatorname{logsumexp}_k(s_{ik})-s_{i,y_i}].$$
中心是簇内算术平均，长度不必为 1；不能擅自把欧氏分配换成余弦最大值。同距取最小中心索引。均匀 logits 的损失为 `log(K)`。


## 3. 公式到代码


按行计算带下界的范数，广播计算特征到各中心的平方距离，再沿中心维取 `argmin`。离散簇标签必须与梯度图分离；训练路径的归一化和全部分类 logits 保留梯度。用稳定的 log-sum-exp 计算样本平均交叉熵。


## 4. 本题约定


浮点矩阵为非空、有限的 float32/float64，同宽、同 dtype/device。标签为范围内 int64。默认 `eps=1e-12`，零向量归一化后仍为零；支持非连续输入，不修改输入。分配函数接收调用方已归一化的特征，不归一化中心。损失只对给定样本求均值；原系统通过采样平衡簇，本函数不会自动插入类别权重。starter 已提供输入验证。


## 5. 交互图与 CPU 实验


九个确定性二维点形成三个簇，2×2 编码器和 2×3 分类头共十个可训练参数。以学习率 `.25` 做 90 次 SGD，在第 0、30、60、90 步更新伪标签和簇均值；空簇保留旧中心。实验实际调用三个待实现函数。T3 要求三个簇均被占用、编码器移动量大于 `.01`、伪标签准确率为 1、最终损失低于 `.1`。这些指标只描述合成训练，不包含 PCA 拟合或图像编码器复现。


## 6. 运行方式


网站点击 **Run tests** 运行完整 CPU/WASM 测试，也可使用下面的命令。测试涵盖独立标量几何、归一化及非连续存储梯度、稳定分类梯度、簇重标号不变性和完整交替训练。


```bash
paperleet test representation/deepcluster --impl problems/representation/deepcluster/solution.py
paperleet mutate representation/deepcluster
paperleet vizdata representation/deepcluster
```



## 7. 延伸与来源


原系统在整批图像特征上重新拟合 PCA/白化与 K-means，再用簇平衡采样训练。完成核心后可研究这些外层策略，并比较 SwAV 的平衡软分配和跨视图预测。论文与官方实现见下方。合成数据、边界约定、测试与微型优化器是本题教学设计。


[Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520) · [Official implementation](https://github.com/facebookresearch/deepcluster/blob/main/clustering.py)
