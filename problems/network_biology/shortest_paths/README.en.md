# Shortest Paths and Network Proximity

Uses the graph conventions of *Biological Interaction Networks: Node Degree and Degree Distribution* (exercise 93). You compute how many interaction steps separate two proteins and which path realizes it.

## Task and conventions

**Unweighted, undirected** simple graph: unique `str` node names, `(u, v)` pairs in any order, repeated and reversed pairs count once, self-loops are ignored. Every edge has length 1; confidence scores or other weights are not part of this exercise (a weighted version would need a different algorithm such as Dijkstra's). A duplicate node name, an unknown edge endpoint, an unknown `source` or an unknown `target` raises `ValueError`.

```python
distances_from(nodes, edges, source) -> dict[str, int]
shortest_path_length(nodes, edges, source, target) -> int | None
shortest_path(nodes, edges, source, target) -> list[str] | None
```

- `distances_from` maps every node reachable from `source` to its distance in edges; `source` maps to `0`; unreachable nodes are omitted.
- `shortest_path_length` returns the distance, `0` when `source == target` (even for an isolated node), and `None` when the target is unreachable.
- `shortest_path` returns `[source]` when `source == target`, `None` when unreachable, and otherwise one shortest path as a list of node names. If several shortest paths exist, return the **lexicographically smallest list of names**: compare candidate paths element by element as strings.

The distance is $d(s,t)=\min$ number of edges on any $s$-$t$ path, or undefined (`None`) if no path exists.

## Example

Synthetic toy network (not experimental data): edges `P1-P2, P1-P3, P2-P4, P3-P4, P4-P5, P6-P7` over nodes `P1`...`P7`.

```python
distances_from(nodes, edges, "P1")          # {'P1': 0, 'P2': 1, 'P3': 1, 'P4': 2, 'P5': 3}
shortest_path_length(nodes, edges, "P1", "P5")  # 3
shortest_path(nodes, edges, "P1", "P5")         # ['P1', 'P2', 'P4', 'P5']  (the other tie is P1, P3, P4, P5)
shortest_path_length(nodes, edges, "P1", "P6")  # None, a different component
```

## Before implementing

Breadth-first search (BFS) visits nodes in order of distance. Record each node's distance and the node that first discovered it (its parent). If you always expand neighbors in sorted order, the parent chain from the target back to the source spells out exactly the lexicographically smallest shortest path, so no extra tie-breaking is needed; you only have to make the visiting order deterministic.

## What path length does and does not say

Short graph distance between two genes or proteins is a **hypothesis** that they may be functionally related, and the idea of network location and proximity is used in network medicine. It is not a proven functional or causal relationship: distance depends on which interactions were recorded, hub proteins create short paths between unrelated processes, and a long or missing path may just reflect incomplete data. The exercise implements plain shortest-path distance only, not any specific published proximity or separation statistic, and uses synthetic graphs.

## Tests and experiment

T0 checks the return types. T1 checks the documented toy network and compares pseudo-random graphs with a Floyd-Warshall distance oracle and an exhaustive lexicographic-minimum path search. T2 covers `source == target`, unreachable targets, tie handling under reordered input, path validity, symmetry and the triangle inequality, noise edges and validation. T3 runs your functions on a synthetic 10-node network and shows how one added edge `P01-P06` shortens the distances and changes the path from `P01` to `P07`.

## References

- Menche J, Sharma A, Kitsak M, Ghiassian SD, Vidal M, Loscalzo J, Barabasi AL. [Uncovering disease-disease relationships through the incomplete interactome](https://doi.org/10.1126/science.1257601). *Science* 347(6224), 1257601 (2015). Background on the network location of disease modules; this exercise does not implement its separation measure.
- Literature dataset record: `docs/research/network_biology_literature.json`, id 42.
