# RDT-1B: 题解与易错点

统一表示中的空槽不能挤压掉：同一槽始终对应相同物理量。掩码既区分缺失维度，也在执行前清除去噪器的无效输出；源码训练MSE不等于有效维度mask损失，因此本题不替换其训练目标。

使用one-hot槽矩阵乘法可保留局部动作梯度。重复槽会让物理含义混合，应直接拒绝。有效值本身为0仍然是存在的动作维度。

## 测试捕获的错误

- The packing matrix uses local indices instead of semantic slots.
- Execution fails to mask absent action coordinates.

## 实验范围

提取RDT统一动作空间与采样输出mask；使用可配置小槽表，不复制完整128维schema或12亿参数Transformer。

## 可重复实验记录

3个局部坐标映射至6维统一动作；初值9，100步梯度下降(lr=0.5)。要求有效动作误差<初始1e-10且未用坐标严格为零。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
