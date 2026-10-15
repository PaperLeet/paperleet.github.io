# TIGER：生成 ID 与合法物品召回

## 1. 问题

模块 64 为物品生成唯一语义 ID。本题给定冻结的 `next_logits(prefix)` 回调，实现自回归 beam search，再把完整 ID 映射回合法目录物品。输入的 ID 可直接来自上一题；本题判题独立，不依赖你先完成上一题。

## 2. 核心公式

### 返回值要求

#### `beam_search` (需实现)

返回 Python `list[tuple[tuple[int, ...], float]]`；每项为 `(tokens, cumulative_log_score)`，tokens 含恰好 `length` 个 Python int，分数为 Python float，不是 Tensor。最多保留 `beam_width` 项，先按分数降序，精确并列时按 token tuple 字典序排列。`length=0` 返回 `[((), 0.0)]`。这是无梯度的冻结解码，不修改 callback 的 logits。

#### `lookup_candidates` (需实现)

返回 Python `list[tuple[int, float]]`，每项为 `(catalog_row_index, original_log_score)`；删除不在目录中的完整 ID 和重复物品后，保留 beam 顺序。整数索引对应目录 `[N,L]` 的行，不是 token。最多返回 k 项；k=0、输入/目录为空或无有效 ID 时返回 `[]`。保留 Python float 原始分数，不重新归一化、不反传、不修改输入。

### 公式与约定

$$s(c_{1:L})=\sum_{d=1}^{L}\log\operatorname{softmax}(f(c_{<d}))_{c_d}.$$

每一步扩展所有保留前缀的全部 V 个 token，按累计分数从所有候选中保留前 b 个。完成 L 步后，过滤不属于目录的完整 ID，去掉重复物品，再取 k 个。分数不重新归一化。

## 3. 公式到代码

实现 `beam_search(next_logits, length, beam_width)` 和 `lookup_candidates(beams, catalog_ids, k)`。起点为 `[((),0.0)]`。回调接收 token tuple，返回 1D logits。每一步在**整个词表**做 log-softmax。

手算：第一步 A=.6、B=.4，第二步 A→0=.51、B→0=.9，则 A0=.306、B0=.36。宽度 2 应保留 B0 在前；只扩展第一个前缀或只记最后一个 token 的概率都会出错。

## 4. 本题约定

- length 是非负整数，beam_width 是正整数。回调总是返回有限、冻结的 CPU float64 `[V]`，V 固定且至少 1。length=0 不调用回调，直接返回起点。禁止修改 logits。
- 返回 `list[(tuple[int,...], float)]`，按累计 log-score 降序，同分按完整 token tuple 字典序升序。候选少于 beam_width 时全部保留。固定长度、无 EOS；没有长度惩罚或采样。
- lookup 输入已排序，可能重复。`catalog_ids[N,L]` 是唯一的 CPU int64 行，k 是非负整数。丢弃未知 ID 和重复物品后才截取 k。返回 `(目录行号, 原分数)`，保留原顺序；k=0、空目录或无合法项均返回空列表。禁止修改输入。
- 输入满足约定，无需参数校验。数字 token 在本题按位置解释，区别于论文的分层 token 词表。固定回调代替完整的用户历史 encoder-decoder；不训练 T5，也不测试推荐相关性。
- 原论文 §4.5 说明可增大 beam size 再过滤无效 ID；前缀匹配是其后续工作。本题**没有** trie 约束或按合法目录重新归一化。有限宽度不保证找到全局最优，增大宽度对任意模型也不保证单调改善。

## 5. 交互图

Lab 使用与上一题一致的 8 个物品 ID、5 个 token、4 个位置。对 625 个可能 ID 的独立标量枚举给出**同一个固定模型**的精确排序，比较 beam 宽度 1/2/4/8/16/625 的结果。宽度 625 覆盖整棵树。top-3 overlap 只衡量模型排序近似，不是用户相关性 Recall/NDCG；图中的无效 ID 来自你的生成函数，过滤由你的 lookup 完成。

## 6. 怎么运行

```bash
paperleet test retrieval/generative_retrieval --impl problems/retrieval/generative_retrieval/solution.py
```

网页 Browser only 使用 CPU/WASM。T0 检查恒等/空输入，T1 手算，T2 检查独立标量解、前缀常数平移、累计分数、全局截断、并列规则和过滤顺序，T3 对实际提交执行 beam 宽度实验。

## 7. 延伸

阅读 [TIGER §3.2 与 §4.5](https://arxiv.org/html/2305.05065)。可以另做 trie 约束、温度采样或 learned logits，但应明确它们会改变解码规则。真实推荐评估需要用户历史、训练/验证划分以及相关性标签，不能用这个 toy 的精确模型排序替代。
