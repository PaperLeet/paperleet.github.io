# 连通分量与最大连通分量

沿用“生物互作网络：节点度与度分布”（`network_biology/degree_distribution`）的图约定。**连通分量**是一组极大的蛋白质集合，其中任意两个蛋白都能沿互作边相互到达。

## 任务与标准约定

简单无向图：`str` 节点名唯一，`(u, v)` 对内顺序任意，重复或反向重复的边只算一条，自环被忽略，节点名重复或端点未知抛出 `ValueError`。

```python
connected_components(nodes, edges) -> list[list[str]]
largest_component(nodes, edges) -> list[str]
largest_component_fraction(nodes, edges) -> float
```

- 每个分量是按字符串升序排列的节点名列表。
- `connected_components` 按大小从大到小排列分量；大小相同的分量按其最小节点名排序。
- 没有伙伴的节点自成一个大小为一的分量。没有节点的图没有分量：`[]`。
- `largest_component` 是第一个分量（没有节点时为 `[]`），因此大小并列由最小节点名决定。
- `largest_component_fraction` 是最大分量的大小除以节点数，没有节点时为 `0.0`。

## 示例

合成的玩具网络（不是实验数据）：节点 `P1`…`P9`，边 `P1-P2, P2-P3, P3-P1, P3-P4, P5-P6, P6-P7`。

```python
connected_components(nodes, edges)
# [['P1', 'P2', 'P3', 'P4'], ['P5', 'P6', 'P7'], ['P8'], ['P9']]
largest_component(nodes, edges)           # ['P1', 'P2', 'P3', 'P4']
largest_component_fraction(nodes, edges)  # 0.4444...（9 个节点中的 4 个）
```

## 动手前想一想

按排序后的顺序访问节点。遇到尚未访问的节点就开始一次深度或广度优先遍历，收集它能到达的全部节点：这就是一个分量。孤立节点自然表现为什么都到不了的遍历。用已访问集合保证每个节点只处理一次。

## 不连通能说明什么、不能说明什么

不同分量中的两个蛋白，在*这张图里*没有已记录的互作路径。这并不证明它们在生物学上相互独立：互作可能存在但未被测到，实验可能漏掉瞬时或条件特异的接触，网络可能混合了多个数据来源，蛋白还可能通过图中没有的机制（代谢物、信号传导、调控）发生耦合。同样，孤立节点只是没有*已记录*的伙伴，可能反映研究不足而不是没有功能。这里所有的图都是合成的。

## 测试与实验

T0 检查返回类型。T1 检查文档中的玩具网络，并用并查集基准比较伪随机稀疏图。T2 覆盖孤立节点、空图、并列规则、无向边与噪声、输入校验、分量是否划分整张图且没有边跨越分量，以及单条桥边的影响。T3 在由六个 5 节点模块串成的合成链上运行你的函数，逐条移除连接边，绘制分量个数和最大分量占比。

## 参考资料

- Barabasi AL, Oltvai ZN. [Network biology: understanding the cell's functional organization](https://doi.org/10.1038/nrg1272). *Nature Reviews Genetics* 5(2), 101-113 (2004)。描述细胞网络结构的一般背景；连通分量练习本身是教学改编。
