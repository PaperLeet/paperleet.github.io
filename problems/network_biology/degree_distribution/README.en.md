# Biological Interaction Networks: Node Degree and Degree Distribution

A standalone network-biology exercise; no earlier exercise is required. You only need Python dictionaries, sets and loops. This is the first problem of the Network Biology collection.

## Task and conventions

Model a protein-protein interaction (PPI) network as a **simple undirected graph**. `nodes` is a list of unique `str` names and `edges` is a list of `(u, v)` pairs; the order inside a pair is irrelevant.

- A repeated pair, or the same pair written in the opposite order, is one edge.
- A self-loop `(u, u)` is ignored. This is a modeling choice: real proteins can interact with themselves, but a simple graph has no loops.
- A node with no partner stays in the result with degree `0`.
- A repeated name in `nodes`, or an edge endpoint that is not in `nodes`, raises `ValueError`.

```python
node_degrees(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]
degree_distribution(nodes: list[str], edges: list[tuple[str, str]]) -> dict[int, float]
```

The **degree** of a node is its number of distinct neighbors, a property of one protein. The **degree distribution** is a property of the whole network:

$$
k_v=|N(v)|,\qquad P(k)=\frac{n_k}{N},\qquad n_k=|\{v: k_v=k\}|.
$$

`degree_distribution` returns `P(k)` for every integer `k = 0, 1, ..., max degree` in ascending key order. Degrees that no node has map to `0.0`. A graph with no nodes returns `{}`.

## Example

Synthetic toy network (not experimental data): nodes `P1`...`P8`, edges `P1-P2, P1-P3, P1-P4, P1-P5, P2-P3, P5-P6, P6-P7`.

```python
node_degrees(nodes, edges)
# {'P1': 4, 'P2': 2, 'P3': 2, 'P4': 1, 'P5': 2, 'P6': 2, 'P7': 1, 'P8': 0}
degree_distribution(nodes, edges)
# {0: 0.125, 1: 0.25, 2: 0.5, 3: 0.0, 4: 0.125}
```

## Before implementing

Build a neighbor *set* per node so duplicates collapse automatically, skip `u == v`, and validate every endpoint before you use it. Then count how many nodes have each degree and divide by the number of nodes.

## What the graph does and does not say

An edge is a recorded interaction from an assay or database. Records depend on the experimental method and are incomplete, so a node's degree is the number of *recorded* partners, not a measure of biological importance. The toy graphs are synthetic. Descriptions of long-tailed or "scale-free" degree distributions in the literature are hypotheses to test with data and statistics; this exercise does not fit a power law.

## Tests and experiment

T0 checks the return types. T1 compares the documented toy network and pseudo-random graphs with an adjacency-matrix oracle. T2 covers empty and one-node graphs, isolated nodes, duplicate edges, self-loops, invalid input, normalization, and order independence. T3 runs your functions on the toy network, shows how one added edge `P4-P7` moves two nodes from degree 1 to degree 2, and plots a deterministic 60-node synthetic growth network. Charts show measured outputs of your code only.

## References

- Albert R. [Scale-free networks in cell biology](https://doi.org/10.1242/jcs.02714). *Journal of Cell Science* 118(21), 4947-4957 (2005). Reviews graph concepts, including degree and the degree distribution, for describing cellular networks.
- Literature dataset record: `docs/research/network_biology_literature.json`, id 3.
