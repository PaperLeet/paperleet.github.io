# RT-2: 题解与易错点

先建立逐行合法集合，再将非法logit替换为负无穷并做稳定softmax。先softmax再置零会破坏归一化。对机器人行的非法logits，输出及梯度都为零。

先建立逐行合法集合，再将非法logit替换为负无穷并做稳定softmax。先softmax再置零会破坏归一化。对机器人行的非法logits，输出及梯度都为零。

先按公式确定归约轴、有效掩码和梯度路径，再将手算小例与实现逐项比较。测试中的独立预期值不调用参考答案。

## 测试捕获的错误

- Robot rows retain forbidden language tokens.
- The action-only vocabulary is incorrectly applied to vision-language rows.

## 实验范围

实现机器人任务与视觉语言任务共享logits时的输出约束；动作token集合由调用者提供，不运行PaLI-X、PaLM-E或语义推理。

## 可重复实验记录

2行任务、4个token、2个合法动作token；零logits开始，80步负对数概率优化(lr=1)。要求loss<初始4%，机器人非法质量为零且VLM非法动作集合仍有正质量。

判题固定seed=42。计算采用CPU float64、显式输入，涉及优化时使用普通梯度更新；不下载数据集或模型权重。原生PyTorch与真实Pyodide/NumPy-WASM执行同一套T0–T3测试。阈值针对本教学实验。
