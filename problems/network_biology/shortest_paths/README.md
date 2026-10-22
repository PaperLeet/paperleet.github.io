# 最短路径与网络邻近性

沿用“生物互作网络：节点度与度分布”（`network_biology/degree_distribution`）的图约定。你要计算两个蛋白之间相隔多少步互作，以及哪条路径实现了这个距离。

## 任务与标准约定

**无权、无向**的简单图：`str` 节点名唯一，`(u, v)` 对内顺序任意，重复或反向重复的边只算一条，自环被忽略。每条边长度为 1；置信度等权重不属于本题（加权版本需要 Dijkstra 等不同算法）。节点名重复、边端点未知、`source` 未知或 `target` 未知都抛出 `ValueError`。

```python
distances_from(nodes, edges, source) -> dict[str, int]
shortest_path_length(nodes, edges, source, target) -> int | None
shortest_path(nodes, edges, source, target) -> list[str] | None
```

- `distances_from` 把从 `source` 可达的每个节点映射到以边数计的距离；`source` 映射为 `0`；不可达节点不出现。
- `shortest_path_length` 返回距离；`source == target` 时返回 `0`（孤立节点也是如此）；目标不可达时返回 `None`。
- `shortest_path` 在 `source == target` 时返回 `[source]`，不可达时返回 `None`，否则返回一条最短路径（节点名列表）。若有多条最短路径，返回**字典序最小的名称列表**：逐个元素按字符串比较候选路径。

距离 $d(s,t)$ 为任一 $s$-$t$ 路径上最少的边数；不存在路径时无定义（`None`）。

## 示例

合成的玩具网络（不是实验数据）：节点 `P1`…`P7`，边 `P1-P2, P1-P3, P2-P4, P3-P4, P4-P5, P6-P7`。

```python
distances_from(nodes, edges, "P1")          # {'P1': 0, 'P2': 1, 'P3': 1, 'P4': 2, 'P5': 3}
shortest_path_length(nodes, edges, "P1", "P5")  # 3
shortest_path(nodes, edges, "P1", "P5")         # ['P1', 'P2', 'P4', 'P5']（另一条并列路径是 P1, P3, P4, P5）
shortest_path_length(nodes, edges, "P1", "P6")  # None，位于另一个分量
```

## 动手前想一想

广度优先搜索（BFS）按距离顺序访问节点。记录每个节点的距离和首次发现它的节点（父节点）。如果总是按排序后的顺序扩展邻居，从目标沿父节点链回到起点得到的正好是字典序最小的最短路径，无需额外的并列处理；你只需让访问顺序是确定的。

## 路径长度能说明什么、不能说明什么

两个基因或蛋白之间图距离很短，只是它们*可能*在功能上相关的**假设**；网络位置与邻近性的思路也用于网络医学。它并不是已被证明的功能或因果关系：距离取决于记录了哪些互作，枢纽蛋白会在无关过程之间制造捷径，而很长或缺失的路径可能只是数据不完整。本题只实现普通最短路径距离，不实现任何已发表的邻近性或分离度统计量，并且使用合成图。

## 测试与实验

T0 检查返回类型。T1 检查文档中的玩具网络，并用 Floyd-Warshall 距离基准和穷举字典序最小路径的搜索比较伪随机图。T2 覆盖 `source == target`、不可达目标、输入重排后的并列处理、路径有效性、对称性与三角不等式、噪声边与输入校验。T3 在合成的 10 节点网络上运行你的函数，展示添加一条边 `P01-P06` 如何缩短距离并改变从 `P01` 到 `P07` 的路径。

## 参考资料

- Menche J, Sharma A, Kitsak M, Ghiassian SD, Vidal M, Loscalzo J, Barabasi AL. [Uncovering disease-disease relationships through the incomplete interactome](https://doi.org/10.1126/science.1257601). *Science* 347(6224), 1257601 (2015)。关于疾病模块网络位置的背景；本题不实现其分离度量。
