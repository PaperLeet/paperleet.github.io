# RotNet 四方向旋转预测


## 1. 问题


实现 `rotation_batch(images)` 与 `rotation_loss(logits, labels)`。RotNet 预测图像被旋转了哪一个角度，监督标签完全由施加的变换产生。本题同时练习精确图像几何和真实旋转分类训练。


## 2. 核心公式


角度为逆时针 $90r$ 度，$r\in\{0,1,2,3\}$：
$$\widetilde x_{rB+b}=R_{90r}(x_b),\quad y_{rB+b}=r,$$
$$L=\frac1M\sum_i[\operatorname{logsumexp}_{k=0}^3(s_{ik})-s_{i,y_i}].$$
输出依次为四个旋转角度的大块。`[[1,2],[3,4]]` 逆时针旋转 90° 后是 `[[2,4],[1,3]]`；均匀 logits 得到 `log(4)`。


## 3. 公式到代码


在两个空间维度上组合转置与翻转，依次拼接 0°、90°、180°、270° 视图，每种标签重复 B 次。旋转是像素置换，不能断开图像梯度。损失提取每行给定标签的分数，从稳定 log-sum-exp 中减去，再对样本求均值。


## 4. 本题约定


图像为非空、有限 float32/float64 `[B,C,H,H]`；使用正方形使四个视图尺寸一致。输出为 `[4B,C,H,H]` 与同 device 的 int64 `[4B]` 标签。按角度分块是本题批次约定，官方加载器先逐图像生成视图。无插值、裁剪、颜色变化或原地修改；支持非连续输入。损失接收任意非空 `[M,4]` logits 与范围 `[0,3]` 的 int64 `[M]` 标签，不要求 M 是 4 的倍数。1×1 图像也生成四种标签，尽管视图无法区分。


## 5. 交互图与 CPU 实验


八张确定性非对称 4×4 图案产生 32 个视图。零初始化的 16×4 线性分类头，以 `.6` 学习率做 80 次全批 SGD，实际使用两个待实现函数。T3 要求旋转准确率 1，损失低于 `.06` 且低于初始值的 5%。对称图像可能无法判断方向，因此教学图案特意选择非对称形状。旋转训练准确率不能直接证明下游表示质量。


## 6. 运行方式


网站点击 **Run tests** 运行 CPU/WASM，或执行下列命令。测试精确像素和标签、四次旋转恒等、像素集合守恒、非连续图像梯度、独立交叉熵、极端分数与真实训练。


```bash
paperleet test representation/rotnet --impl problems/representation/rotnet/solution.py
paperleet mutate representation/rotnet
paperleet vizdata representation/rotnet
```



## 7. 延伸与来源


论文通过分类、检测和分割研究旋转任务的迁移效果，那需要训练图像编码器和独立下游数据。更早的预任务包括空间上下文预测与拼图。本题与已有 VQ “Rotation Trick” 无关，后者处理梯度估计，不是图像旋转自监督。


[Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728) · [Official implementation](https://github.com/gidariss/FeatureLearningRotNet/blob/master/dataloader.py)
