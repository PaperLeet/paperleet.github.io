# BERT：80/10/10 替换与掩码语言建模

## 1. 论文解决什么问题

自回归模型只根据一侧上下文预测下一词。BERT 的掩码语言建模允许使用左右上下文来恢复原词。本题落实原论文第 3.1 节的目标构造，不复现完整 BERT。调用方已选好需要监督的位置，并提供随机数；你只实现确定性的替换和损失。

[原始来源 / Original source](https://arxiv.org/html/1810.04805v2#S3.SS1) · Section 3.1, Task #1: Masked LM; Appendix C.2: masking procedures.

## 2. 核心公式与返回契约

$$C_{b,t}=\begin{cases}\mathrm{MASK}&S_{b,t}\land u_{b,t}<0.8\\R_{b,t}&S_{b,t}\land0.8\le u_{b,t}<0.9\\X_{b,t}&\mathrm{otherwise}\end{cases}\qquad L=-\frac{1}{|S|}\sum_{(b,t)\in S}\log\operatorname{softmax}(Z_{b,t})_{X_{b,t}}.$$

必做函数：`make_mlm_inputs`, `masked_lm_loss`.

## 3. 从公式到实现

用两组互斥布尔条件选择 MASK 和随机词，其余保留原词；标签始终来自未修改的 tokens。log-softmax 后按标签收集损失，只用有效标签数归一化。

## 4. 本题约定

`make_mlm_inputs` 的四个数组均为 `[B,L]`：tokens/random_ids 为非负 int64，selected 为 bool，draws 为 float32/float64 且每个值有限并在 `[0,1)`。返回两个独立的 int64 数组 `(corrupted,labels)`，不修改输入。选中且 `u<0.8` 时用 mask_token_id，`0.8<=u<0.9` 时用 random_ids，否则保留原词。未选中位置不替换，标签为负整数 ignore_index（默认 -100）；选中位置的标签永远是原词，**不做 next-token shift**。mask_token_id 是非负整数，可在预测词表之外。调用方负责 15% 选择率、特殊词排除和随机词分布。

`masked_lm_loss` 接收 float32/float64 logits `[B,L,V]` 与 int64 labels `[B,L]`，其中 `V>0`。标签只能是 ignore_index 或 `[0,V)` 中的词号。返回同 dtype 的标量，只按有效位置求均值。全忽略、空 batch 或空序列返回与 logits 计算图连接的零，梯度为零。形状、dtype、随机数范围或词号非法时抛出 ValueError。输入在同一设备；logits 为有限数。

## 5. 交互实验

提供 64 个三词序列，覆盖 16 种左右词组合。中心词为 `(2*left+right)%4`，只有中心可被选中，选择率 15%，替换使用你的函数。一个 16×4 的上下文表只读取损坏输入的左右词，使用你的 MLM 损失做 100 次 Adam 更新（lr=0.08，seed=163，float64）。图显示所有中心词的评估 NLL。验收要求最终 NLL<0.1 且小于初始的 10%，上下文准确率>=95%。这是分类表的教学实验，不包含 Transformer 或自然语言泛化结论。

## 6. 运行你的实现

在网页编辑器补全 starter 后运行全部测试；通过后阅读 Notes。也可在仓库根目录运行：

```bash
paperleet test foundations/bert_mlm --impl problems/foundations/bert_mlm/solution.py
```

## 7. 延伸探索

推导有效位置的 softmax 梯度，并比较全忽略 batch 与只含一个标签的 batch。阅读原论文附录 C.2 的替换策略对照；如需复现 BERT，还需实现编码器、NSP 和真实语料处理。
