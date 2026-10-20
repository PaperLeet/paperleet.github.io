# Hub Identification: Degree and Betweenness Centrality

Builds on the graph conventions of *Biological Interaction Networks: Node Degree and Degree Distribution* (exercise 93); finishing it first is recommended but your code for it is not reused. Here you rank the nodes of a protein-protein interaction (PPI) network as **candidate hubs** in two different ways.

## Task and conventions

Same simple undirected graph as exercise 93: unique `str` node names, `(u, v)` pairs in any order, repeated and reversed pairs count once, self-loops are ignored, isolated nodes stay in the result, and a duplicate node name or an unknown edge endpoint raises `ValueError`.

```python
degree_centrality(nodes, edges) -> dict[str, float]
betweenness_centrality(nodes, edges) -> dict[str, float]
rank_hubs(scores: dict[str, float], k: int) -> list[str]
```

**Degree centrality** (N is the number of nodes):

$$
C_D(v)=\frac{k_v}{N-1}.
$$

It is `0.0` for every node when `N = 1`, and `{}` when there are no nodes.

**Betweenness centrality** counts how often a node lies on shortest paths between other nodes. With $\sigma_{st}$ the number of shortest $s$-$t$ paths and $\sigma_{st}(v)$ the number of them passing through $v$ (as an interior node):

$$
C_B(v)=\frac{2}{(N-1)(N-2)}\sum_{\{s,t\}\not\ni v}\frac{\sigma_{st}(v)}{\sigma_{st}}.
$$

The sum runs over unordered pairs $\{s,t\}$ of nodes different from $v$ that are connected; unconnected pairs contribute nothing and path endpoints are never credited. All scores are `0.0` when `N < 3`.

**Ranking.** `rank_hubs(scores, k)` returns the `k` highest-scoring node names, highest first. Scores are rounded to 12 decimal places before comparison, so floating-point noise cannot break a tie. Ties are broken by node name in ascending string order. `k = 0` gives `[]`, `k` larger than the number of nodes gives every node, and `k < 0` raises `ValueError`.

## Example

Synthetic two-module network (not experimental data): two 4-node cliques `A1..A4` and `B1..B4` joined only through a connector `X` via the edges `A1-X` and `X-B1`.

```python
rank_hubs(degree_centrality(nodes, edges), 3)       # ['A1', 'B1', 'A2']
rank_hubs(betweenness_centrality(nodes, edges), 3)  # ['X', 'A1', 'B1']
```

`X` has the *lowest* degree (2 of 8 possible partners) but the highest betweenness: all 16 shortest paths between the two modules pass through it, and `A1`, `B1` each lie on 15 pairs. By hand, `C_B(X) = 16/28`.

## Before implementing

Compute the neighbor sets once. For betweenness you can run a breadth-first search from every node while counting shortest paths ($\sigma$), then credit a node `v` for a pair `(s, t)` when `dist(s,v) + dist(v,t) = dist(s,t)`, adding `sigma(s,v) * sigma(v,t) / sigma(s,t)`. Brandes' algorithm is faster, but this O(N^3) approach is enough here.

## What centrality does and does not say

A high score marks a node that is well connected (degree) or that many shortest paths run through (betweenness) **in this graph**. It is a hypothesis-generating label, not evidence that the protein is essential, causal for a disease, or a good drug target: such claims need experiments. Scores also depend on which interactions were recorded and on how complete the network is.

## Tests and experiment

T0 checks the return types. T1 checks the hand-counted two-module values and compares random small graphs with an exhaustive simple-path oracle. T2 covers rankings that differ, multiple shortest paths, empty, one-node, two-node and disconnected graphs, duplicates and self-loops, validation, tie and boundary rules, and input-order independence. T3 runs your functions on the two-module network and shows that adding one shortcut edge `A4-B4` lowers the betweenness of `X`.

## References

- Barabasi AL, Oltvai ZN. [Network biology: understanding the cell's functional organization](https://doi.org/10.1038/nrg1272). *Nature Reviews Genetics* 5(2), 101-113 (2004). Its Box 1 introduces basic network measures.
- Literature dataset record: `docs/research/network_biology_literature.json`, id 1.
