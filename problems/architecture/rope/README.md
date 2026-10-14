# RoPE 旋转位置编码

## 1. 问题

[RoFormer](https://arxiv.org/abs/2104.09864) §3.2、§3.4.2 将位置信息写入 query/key 的旋转。作者的[官方仓库](https://github.com/ZhuiyiTechnology/roformer)也给出了相邻坐标配对的示意实现。投影后的 q、k 分别调用同一个旋转函数；本题只实现 `apply_rope`，不搭建完整注意力层。

先完成[正弦位置编码](../positional_encoding/README.md)，再比较“给输入加位置向量”和“旋转投影后的特征”如何改变内积。

## 2. 核心公式

对第 i 对相邻特征 $(a,b)=(x_{2i},x_{2i+1})$，令 $\theta_i=\mathrm{base}^{-2i/D}$，$\phi=p\theta_i$：

$$a'=a\cos\phi-b\sin\phi,\qquad b'=a\sin\phi+b\cos\phi.$$

这是正角度的逆时针旋转。成对范数保持不变，且

$$\langle R(m)q,R(n)k\rangle=\langle q,R(n-m)k\rangle.$$

q 和 k 的位置同时平移，内积不变；只移动其中一个通常会改变内积。范数性质本身不能证明符号和频率正确。

## 3. 公式到代码

输入 `x` 为 `[B,H,L,D]`，`positions` 为 `[L]`。生成 D/2 个频率，让相位广播到 `[1,1,L,D/2]`。取偶数、奇数坐标，分别计算新坐标，再交错回原来的最后一维。返回形状、精度和设备都与 x 相同，不原地修改输入。

D=2、位置为 $\pi/2$ 时，`[2,3]` 应变为 `[-3,2]`；位置为 0 时不变。

## 4. 本题约定

- x 为有限的 float32/float64；D 必须为正偶数，B 或 L 可以为 0；支持非连续存储。
- 相邻坐标配对 `(0,1),(2,3),...`。前半/后半配对在相应置换后的模型布局中也可成立，但不符合这里的输入契约。
- `positions` 是长度 L 的一维 tensor，可为有符号整数或实数，也可不连续。以 x 的精度计算相位；浮点位置的可微性是显式教学扩展。
- x 和浮点 positions 的梯度都要保留。base 必须是有限且大于 1 的 Python 数值，不能是 bool。布局、长度或 base 不合法时抛 `ValueError`。
- 旋转覆盖完整 D；不包含半维 RoPE、上下文缩放、mask、value 旋转或模型训练。

## 5. 交互实验

<viz id="toy_experiment" />

相对位置图对比原始、共同平移后和独立标量公式的内积。曲线可能振荡，不应解读为每组向量的相似度都随距离单调下降。

缓存图使用九个 key：先处理前六个，再追加三个。新块必须保留绝对位置 6、7、8；错误地从零重启会改变 logits。所有曲线都调用当前提交函数；实验室的参考预览单独标记。该实验为 PaperLeet 的合成教学改编，不是论文基准或长上下文能力评测。

## 6. 怎么运行

```bash
paperleet test architecture/rope --impl problems/architecture/rope/solution.py
paperleet mutate architecture/rope
paperleet ci --problem architecture/rope
```

T0 检查接口与输入保护，T1 检查手算旋转，T2 检查独立数值、旋转性质、位置广播、解析梯度和数值 Jacobian；T3 运行真实提交的缓存与相对位置实验。

## 7. 延伸

把旋转分别接到 q、k 投影后，并显式管理不同长度的缓存位置。观察为何共同平移不变不能单独检验旋转方向。完成本题表示掌握 RoPE 算子；不等于复现 RoFormer 训练或证明位置外推可靠。
