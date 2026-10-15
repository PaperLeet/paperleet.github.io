# SiTU-GLU 平滑有界激活

实现 Kimi K3 FFN 中的激活核心。[报告 v2 §2.3.2、式 (12)](https://arxiv.org/html/2607.24653v2#S2.SS3.SSS2) 对两条分支分别作平滑限制：

$$
c_\beta(z)=\beta\tanh(z/\beta),\qquad
\operatorname{SiTU}(g,u)=c_{\beta_1}(g)\,\sigma(g)\,c_{\beta_2}(u).
$$

默认 `beta1=4`、`beta2=25`。注意 sigmoid 使用原始 `gate`，不是 softcap 后的数值。

## 实现接口

### 返回值要求

#### `softcap` — 需要实现

返回一个与 x 的形状、dtype、device 完全相同的浮点 Tensor，也支持标量和空 Tensor。各值被平滑限制在 [-beta,beta] 范围内，数值计算可能达到端点。保留 x 的梯度，不修改 x，也不附带返回阈值参数。

#### `situ_glu` — 需要实现

返回一个浮点 Tensor，与 gate、up 的共同形状/dtype/device 相同，保留任意前导维度及空轴。它是逐元素门控激活，绝对值不超过 beta1*beta2，不归约特征维度。保留两分支的梯度，不修改输入。

### 公式与约定

| 函数 | 输入 | 输出 |
|---|---|---|
| `softcap(x, beta)` | 任意形状浮点张量与正的有限标量 | 同形状的平滑限制结果 |
| `situ_glu(gate, up, beta1=4, beta2=25)` | 两个形状相同的已投影张量 | 逐元素激活结果 |

保留 dtype、device 及两条分支的梯度，不修改输入；支持非连续张量和空的前导维度。测试使用 CPU float32/float64。参数有效性检查已放在 starter 中。无需实现输入投影、输出投影或 MoE 路由。

在原点附近，`softcap(x, beta)` 接近 `x`；两条分支同时接近零时，结果接近 SwiGLU。有限的 cap 限制每个输出坐标的绝对值，默认上界为 `4 × 25 = 100`。数学性质见报告 Appendix B。硬截断会改变梯度，因此不能代替 tanh。

## Toy：重新计算 Figure 4 的标量曲线

令 `gate = up = x`，使用论文的 `[-10, 100]` 区间，对比 GLU、SwiGLU 与你的 SiTU-GLU；另绘制门控分支、up 分支和原点附近的曲线。

**来源分类：paper-derived。** 本站按公式独立计算 111 个全区间点和 81 个近原点点，添加误差指标；没有复制原图或声称复现模型训练。预览使用参考实现，运行全部测试后的实验结果调用你提交的两个函数。

T0 检查接口和已提供的参数检查；T1 使用独立标量计算；T2 检查全局上界、解析梯度、局部及极限行为、逐元素独立性；T3 产生上述图表和数值误差。
