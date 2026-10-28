# RDT-1B 统一动作槽与有效维度

## 1. 问题

把不同机器人的局部动作放进统一语义槽，并屏蔽不存在的执行维度。

提取RDT统一动作空间与采样输出mask；使用可配置小槽表，不复制完整128维schema或12亿参数Transformer。

## 2. 核心公式

$$
U_{bht}=\sum_j A_{bhj}\,\mathbf1[t=s_j],\quad M_t=\mathbf1[t\in\{s_j\}],\quad A^{exec}=\hat U\odot M.
$$

## 3. 公式到代码

- `pack_actions(local: Tensor, slots: list, width: int) -> tuple`
- `mask_actions(actions: Tensor, valid: Tensor) -> Tensor`

统一表示中的空槽不能挤压掉：同一槽始终对应相同物理量。掩码既区分缺失维度，也在执行前清除去噪器的无效输出；源码训练MSE不等于有效维度mask损失，因此本题不替换其训练目标。

## 4. 本题约定

local为(B,H,D)，slots为长度D的Python整数list，每个槽唯一且位于[0,width)；返回(B,H,width)浮点动作及(width,)布尔mask。mask_actions接收同形动作及(width,)mask。未使用槽必须为0，动作梯度保留。

校验辅助函数已提供。浮点输入为有限 CPU float32/float64，不修改输入；非法形状、类型、范围或掩码必须抛出 `ValueError`。

## 5. 交互图与确定性实验

将三维动作映射到六维统一空间，训练一个六维常数预测器；未使用槽初始值为9，通过执行mask保证这些维度始终为零。

可视化由 `viz/gen.py` 生成；T3 将你的函数传入同一实验并记录真实指标。这是小规模确定性教学实验，不是机器人基准成绩。

## 6. 运行方式

```bash
paperleet test robotics/rdt_1b --impl problems/robotics/rdt_1b/starter.py
paperleet mutate robotics/rdt_1b
```

T0 检查接口；T1 使用独立数值推导；T2 检查边界和适用的梯度；T3 验证实验行为。

## 7. 来源与延伸

- [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/abs/2410.07864) — Physically interpretable unified action space; official RDTRunner.conditional_sample final action mask
- [Verified source](https://rdt-robotics.github.io/rdt-robotics/)
- [Verified source](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/models/rdt_runner.py)

核验日期：2026-10-28。扩展到完整论文系统还需要本题范围中明确省略的模型、数据和环境。
