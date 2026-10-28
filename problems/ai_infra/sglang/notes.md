# SGLang 前缀 KV 缓存与叶节点 LRU — 题解

## 推导与实现

先逐token匹配，不允许把后缀相同当成前缀命中。复制cache后更新时间戳；空间不足时找无孩子且不受保护的最旧节点。一次删一个叶子，祖先成为叶子后才允许删除。

$$
h(x)=\max\{j:x_{:j}\in C\},\qquad e=\arg\min_{p\in\operatorname{leaves}(C)\setminus P}(\operatorname{lastUse}(p),p).
$$

## 边界与梯度归属

cache 为 {token元组:上次使用整数时刻}，不含空根且必须前缀闭合。prompt 为非负int的非空tuple，capacity 计非根token节点，clock 不小于缓存已有时刻。pinned 给出被运行请求使用的缓存节点，其全部祖先都受保护。本轮prompt路径也受保护；同LRU时按token元组排序。返回 (新cache, 复用token数)，失败也不修改输入。

## 独立检验

T0检查接口和无效边界。T1使用手算或独立标量公式。T2检查不变量、状态守恒、有意义的梯度或状态更新原子性。T3把待测实现传入确定性实验，不会导入两个相同参考实现互相比对。

## 行为实验

顺序回放六个有共享系统提示词的请求，使用容量 7 与 12 的缓存，对比零缓存的prefill token数，并检查节点容量。

将 radix tree 展开为每token一个节点的语义模型；模拟前缀共享、LRU和活动请求pin，不存储真实KV tensor。

### 参考测量

确定性CPU样本；Python 3.14.2, PyTorch 2.14.1.

| Metric | Reference value |
| --- | ---: |
| cached_prefill_tokens | 10 |
| uncached_prefill_tokens | 21 |
| reused_tokens | 11 |
| maximum_nodes | 10 |

## 常见错误

- m1_evict_internal: 允许驱逐仍有子节点的内部节点
- m2_ignore_pins: 忽略运行请求的受保护路径

[SGLang](https://arxiv.org/abs/2312.07104) · Section 3: RadixAttention, longest-prefix reuse, leaf-first LRU and reference counters.
