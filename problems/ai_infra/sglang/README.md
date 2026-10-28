# SGLang 前缀 KV 缓存与叶节点 LRU

## 1. 问题

寻找最长缓存前缀，并在容量限制下只驱逐未被占用的叶节点。

将 radix tree 展开为每token一个节点的语义模型；模拟前缀共享、LRU和活动请求pin，不存储真实KV tensor。

原文: [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) · Section 3: RadixAttention, longest-prefix reuse, leaf-first LRU and reference counters.

## 2. 核心公式

$$
h(x)=\max\{j:x_{:j}\in C\},\qquad e=\arg\min_{p\in\operatorname{leaves}(C)\setminus P}(\operatorname{lastUse}(p),p).
$$

## 3. 公式到代码

- `radix_cache_step(cache: dict, prompt: tuple[int, ...], capacity: int, clock: int, pinned: tuple = ()) -> tuple[dict, int]` — Copied cache after prefix reuse/insertion/eviction, and matched prefix length.

先逐token匹配，不允许把后缀相同当成前缀命中。复制cache后更新时间戳；空间不足时找无孩子且不受保护的最旧节点。一次删一个叶子，祖先成为叶子后才允许删除。

## 4. 本题约定

cache 为 {token元组:上次使用整数时刻}，不含空根且必须前缀闭合。prompt 为非负int的非空tuple，capacity 计非根token节点，clock 不小于缓存已有时刻。pinned 给出被运行请求使用的缓存节点，其全部祖先都受保护。本轮prompt路径也受保护；同LRU时按token元组排序。返回 (新cache, 复用token数)，失败也不修改输入。

## 5. 交互图

顺序回放六个有共享系统提示词的请求，使用容量 7 与 12 的缓存，对比零缓存的prefill token数，并检查节点容量。

T3 使用实际待测函数生成实验，网页参考图使用参考实现。数值来自教学样本，不表示GPU吞吐或完整模型质量。

## 6. 运行方式

```bash
paperleet test ai_infra/sglang --impl problems/ai_infra/sglang/starter.py
paperleet test ai_infra/sglang --impl problems/ai_infra/sglang/solution.py
paperleet mutate ai_infra/sglang
```

## 7. 延伸

结合所引章节，解释本题保留了哪些维度、状态或调度约束，再扩展实验样本并保持已有契约。更大模型、真实设备与通信代价需要单独测量。
