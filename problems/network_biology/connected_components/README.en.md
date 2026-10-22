# Connected Components and the Largest Connected Component

Uses the graph conventions of *Biological Interaction Networks: Node Degree and Degree Distribution* (`network_biology/degree_distribution`). A **connected component** is a maximal set of proteins in which every protein can be reached from every other by following interaction edges.

## Task and conventions

Simple undirected graph: unique `str` node names, `(u, v)` pairs in any order, repeated and reversed pairs count once, self-loops are ignored, and a duplicate node name or an unknown endpoint raises `ValueError`.

```python
connected_components(nodes, edges) -> list[list[str]]
largest_component(nodes, edges) -> list[str]
largest_component_fraction(nodes, edges) -> float
```

- Each component is a list of node names sorted in ascending string order.
- `connected_components` orders components by size, largest first; components of equal size are ordered by their smallest node name.
- A node with no partners is a component of size one. A graph with no nodes has no components: `[]`.
- `largest_component` is the first component (`[]` if there are no nodes), so equal-size ties are decided by the smallest node name.
- `largest_component_fraction` is the size of the largest component divided by the number of nodes, `0.0` for no nodes.

## Example

Synthetic toy network (not experimental data): nodes `P1`...`P9`, edges `P1-P2, P2-P3, P3-P1, P3-P4, P5-P6, P6-P7`.

```python
connected_components(nodes, edges)
# [['P1', 'P2', 'P3', 'P4'], ['P5', 'P6', 'P7'], ['P8'], ['P9']]
largest_component(nodes, edges)           # ['P1', 'P2', 'P3', 'P4']
largest_component_fraction(nodes, edges)  # 0.4444... (4 of 9 nodes)
```

## Before implementing

Visit the nodes in sorted order. When you reach an unvisited node, start a depth-first or breadth-first traversal and collect everything it reaches: that set is one component. Isolated nodes fall out naturally as traversals that reach nothing else. Keep a visited set so each node is processed once.

## What disconnection does and does not say

Two proteins in different components have no recorded path of interactions between them *in this graph*. That is not evidence of biological independence: an interaction may exist but be unmeasured, the assay may miss transient or condition-specific contacts, the network may mix data sources, and proteins can be coupled by mechanisms the graph does not contain (metabolites, signaling, regulation). Likewise, an isolated node is a protein with no *recorded* partner, which may reflect lack of study rather than lack of function. All graphs here are synthetic.

## Tests and experiment

T0 checks the return types. T1 checks the documented toy network and compares pseudo-random sparse graphs with a disjoint-set oracle. T2 covers isolated nodes, empty graphs, the tie rule, undirected edges and noise, validation, that components partition the graph and no edge crosses them, and the effect of a single bridge edge. T3 runs your functions on a synthetic chain of six 5-node modules and removes the connecting edges one by one, plotting the number of components and the share of the largest one.

## References

- Barabasi AL, Oltvai ZN. [Network biology: understanding the cell's functional organization](https://doi.org/10.1038/nrg1272). *Nature Reviews Genetics* 5(2), 101-113 (2004). General context for describing the structure of cellular networks; the connected-component exercise itself is a teaching adaptation.
