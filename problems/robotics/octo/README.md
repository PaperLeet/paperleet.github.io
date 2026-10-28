# Octo 双重padding掩码归约

## 1. 问题

Octo 双重padding掩码归约：实现下列接口并通过数值、边界和实验检查。

提取官方动作头的masked_mean及时间/动作padding交集；计算未乘额外action_dim系数的逐坐标平方误差，不实现扩散网络。

## 2. 核心公式

$$
M_{bwhd}=T_{bw}\land A_{bwhd},\qquad L=\frac{\operatorname{mean}(M\odot(\hat a-a)^2)}{\max(\operatorname{mean}(M),10^{-5})}.
$$

## 3. 公式到代码

- `masked_action_loss(prediction: Tensor, target: Tensor, timestep_valid: Tensor, action_valid: Tensor) -> Tensor`

观察窗口中的padding与动作时域/维度padding含义不同，必须取交集。有效元素占比进入分母，因此追加padding不稀释损失；极端稀疏掩码遵循源码分母下界。

## 4. 本题约定

预测与目标为(B,W,H,D)，timestep_valid为(B,W)布尔，action_valid为(B,W,H,D)布尔。目标detach；全padding返回可反传的0；保留源码1e-5分母下界。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

`experiment.py` 在固定输入上调用提交的全部接口，记录学习或采样曲线与末端指标。实验逻辑和阈值见 T3，输出同样用于可视化。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/octo --impl problems/robotics/octo/starter.py
paperleet mutate robotics/octo
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) — Official action_heads.py masked_mean and timestep/action pad-mask intersection
- [Verified source](https://github.com/octo-models/octo/blob/main/octo/model/components/action_heads.py)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
