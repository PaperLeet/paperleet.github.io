# 枢纽蛋白识别：度中心性与介数中心性

建立在“生物互作网络：节点度与度分布”（`network_biology/degree_distribution`）的图约定之上；建议先完成它，但本题不复用你的旧代码。这里用两种不同方式给蛋白质互作（PPI）网络的节点排序，选出**候选枢纽**。

## 任务与标准约定

与 `network_biology/degree_distribution` 相同的简单无向图：`str` 节点名唯一，`(u, v)` 对内顺序任意，重复或反向重复的边只算一条，自环被忽略，孤立节点保留在结果中，节点名重复或边端点不在 `nodes` 中抛出 `ValueError`。

```python
degree_centrality(nodes, edges) -> dict[str, float]
betweenness_centrality(nodes, edges) -> dict[str, float]
rank_hubs(scores: dict[str, float], k: int) -> list[str]
```

**度中心性**（N 为节点数）：

$$
C_D(v)=\frac{k_v}{N-1}.
$$

`N = 1` 时每个节点为 `0.0`，没有节点时返回 `{}`。

**介数中心性**统计一个节点出现在其他节点之间最短路径上的频率。设 $\sigma_{st}$ 为 $s$-$t$ 最短路径条数，$\sigma_{st}(v)$ 为其中以 $v$ 为内部节点的条数：

$$
C_B(v)=\frac{2}{(N-1)(N-2)}\sum_{\{s,t\}\not\ni v}\frac{\sigma_{st}(v)}{\sigma_{st}}.
$$

求和遍历不含 $v$ 且相互连通的无序点对 $\{s,t\}$；不连通的点对没有贡献，路径端点自身不计分。`N < 3` 时所有分数为 `0.0`。

**排名。** `rank_hubs(scores, k)` 返回得分最高的 `k` 个节点名，由高到低。比较前先把分数四舍五入到 12 位小数，因此浮点噪声不会打破并列。并列时按节点名的升序字符串排序。`k = 0` 返回 `[]`，`k` 大于节点数则返回全部节点，`k < 0` 抛出 `ValueError`。

## 示例

合成的双模块网络（不是实验数据）：两个 4 节点团 `A1..A4` 与 `B1..B4`，只通过连接节点 `X` 相连，边为 `A1-X` 和 `X-B1`。

```python
rank_hubs(degree_centrality(nodes, edges), 3)       # ['A1', 'B1', 'A2']
rank_hubs(betweenness_centrality(nodes, edges), 3)  # ['X', 'A1', 'B1']
```

`X` 的度*最低*（8 个可能伙伴中只有 2 个），介数却最高：两个模块之间的全部 16 条最短路径都经过它，而 `A1`、`B1` 各出现在 15 个点对上。手算可得 `C_B(X) = 16/28`。

## 动手前想一想

先算出一次邻居集合。介数可以从每个节点出发做广度优先搜索并同时计数最短路径（$\sigma$），当 `dist(s,v) + dist(v,t) = dist(s,t)` 时，为点对 `(s, t)` 给节点 `v` 加上 `sigma(s,v) * sigma(v,t) / sigma(s,t)`。Brandes 算法更快，但本题用 O(N^3) 的做法已足够。

## 中心性能说明什么、不能说明什么

高分表示该节点在**这张图里**连接很多（度）或有很多最短路径经过（介数）。这只是产生假设的标签，并不证明该蛋白是必需基因、与某种疾病存在因果关系或适合作为药物靶点：这些结论需要实验。分数还取决于哪些互作被记录以及网络的完整程度。

## 测试与实验

T0 检查返回类型。T1 检查手算的双模块数值，并用穷举简单路径的基准比较随机小图。T2 覆盖排名不同、多条最短路径、空图/单节点/双节点/不连通图、重复边与自环、输入校验、并列与边界规则以及与输入顺序无关。T3 在双模块网络上运行你的函数，并展示添加一条捷径边 `A4-B4` 会降低 `X` 的介数。

## 参考资料

- Barabasi AL, Oltvai ZN. [Network biology: understanding the cell's functional organization](https://doi.org/10.1038/nrg1272). *Nature Reviews Genetics* 5(2), 101-113 (2004)。其中的 Box 1 介绍基本网络度量。
