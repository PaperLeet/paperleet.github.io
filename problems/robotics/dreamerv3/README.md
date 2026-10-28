# DreamerV3 symlog与two-hot价值编码

## 1. 问题

DreamerV3 symlog与two-hot价值编码：实现下列接口并通过数值、边界和实验检查。

实现跨数量级标量变换与离散支持编码；不训练RSSM、想象策略或Minecraft。two_hot输入已经位于symlog空间，中心由调用者指定。

## 2. 核心公式

$$
\operatorname{symlog}(x)=\operatorname{sign}(x)\ln(1+|x|),\quad \operatorname{symexp}(y)=\operatorname{sign}(y)(e^{|y|}-1),\quad p_i=\frac{b_{i+1}-y}{b_{i+1}-b_i},\quad p_{i+1}=1-p_i.
$$

## 3. 公式到代码

- `symlog(values: Tensor) -> Tensor`
- `symexp(values: Tensor) -> Tensor`
- `two_hot(values: Tensor, bins: Tensor) -> Tensor`

先symlog压缩原始标量，再在相邻两个中心间线性分配质量。编码期望等于裁剪后的变换值；解码时先在变换空间求期望，再symexp。将symexp分别作用于中心后平均不是同一操作。

## 4. 本题约定

symlog/symexp接受非空浮点张量；two_hot values任意非空形状，bins为严格递增一维，长度>=2，同dtype/device。标签固定detach，范围外值夹到端点；输出shape=values.shape+(N,)。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/dreamerv3 --impl problems/robotics/dreamerv3/starter.py
paperleet mutate robotics/dreamerv3
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) — Symlog predictions and two-hot distribution; Appendix Distributions
- [Verified source](https://arxiv.org/html/2301.04104v2)
- [Verified source](https://github.com/danijar/dreamerv3/blob/main/embodied/jax/outs.py)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
