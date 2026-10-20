# Community Detection and Candidate Biological Modules

Uses the graph conventions of *Biological Interaction Networks: Node Degree and Degree Distribution* (exercise 93). You score partitions of a network with **modularity** and implement a deterministic greedy algorithm that proposes **communities**, which can be read as candidate functional modules.

## Task and conventions

Simple undirected graph: unique `str` node names, `(u, v)` pairs in any order, repeated and reversed pairs count once, self-loops are ignored, and a duplicate node name or an unknown edge endpoint raises `ValueError`. Let `m` be the number of distinct non-loop edges and $D_c$ the sum of node degrees in community `c`.

```python
modularity(nodes, edges, communities, resolution=1.0) -> float
greedy_modularity_communities(nodes, edges, resolution=1.0) -> list[list[str]]
```

$$
Q_\gamma=\sum_{c}\left[\frac{L_c}{m}-\gamma\left(\frac{D_c}{2m}\right)^2\right],
$$

where $L_c$ is the number of edges with both ends in community $c$ and $\gamma$ is `resolution`. `modularity` returns `0.0` when `m = 0`. `communities` must be a partition: nonempty lists, disjoint, covering every node exactly once, otherwise `ValueError`.

**Greedy algorithm (deterministic, no randomness).** Start with every node in its own community. Merging two communities `a` and `b` that are joined by $e_{ab}\ge 1$ edges changes $Q_\gamma$ by

$$
\Delta Q=\frac{e_{ab}}{m}-\gamma\,\frac{D_aD_b}{2m^2}.
$$

Repeatedly perform the merge with the largest $\Delta Q$, comparing gains **rounded to 12 decimals**. Stop when the best gain is not strictly positive. If several pairs have the same rounded gain, merge the pair with the smallest `(smallest member of a, smallest member of b)` name pair, taking `a`'s smallest member to be the smaller one. Communities with no edge between them are never merged, so isolated nodes stay alone, and a graph with no edges returns singletons.

Return each community as a sorted list of names, ordered by size (largest first) and then by smallest member. No nodes gives `[]`.

## Example

Synthetic network (not experimental data): triangles `A1,A2,A3` and `B1,B2,B3` joined by the edge `A3-B1`.

```python
greedy_modularity_communities(nodes, edges)
# [['A1', 'A2', 'A3'], ['B1', 'B2', 'B3']]
modularity(nodes, edges, [["A1","A2","A3"], ["B1","B2","B3"]])   # 5/14 = 0.357142...
modularity(nodes, edges, [nodes])                                # 0.0
```

On two 4-node cliques joined by one edge, `resolution=0.1` gives one community, `1.0` gives the two cliques and `4.0` gives eight singletons. The partition you get depends on the parameter.

## Before implementing

Keep, for each community, its member list, its degree sum and a map from neighbor community to the number of edges between them. Scan every connected pair once per round, pick the best by `(−rounded gain, a, b)`, merge `b` into `a` and update the neighbor maps. Identify a community by its smallest member so the tie rule is easy to apply.

## What communities do and do not say

A detected community is a group of nodes that is more densely connected internally than expected under the modularity null model; it is a **candidate** structural module. Different algorithms, parameters (here `resolution`), tie rules and even small changes to the graph can give different partitions, and modularity maximization has a known resolution limit for small modules. Detected modules are not experimentally validated pathways, complexes or functional units: relating them to biology requires annotation enrichment, replication and experiments. The cited paper studies modular organization of metabolic networks; the algorithm here is a generic teaching method, not that paper's procedure. All graphs are synthetic.

## Tests and experiment

T0 checks the return types. T1 checks modularity against hand-computed values and a matrix-form oracle, and checks that the planted modules are recovered, using an exhaustive search over all partitions of the 8-node graph for the optimum. T2 covers partition and graph validation, edgeless, isolated and noisy graphs, the tie rule and the zero-gain stop, structural invariants (partition, connected communities, modularity beats singletons), resolution dependence, and determinism under reordering. T3 runs your functions on a synthetic three-module network and plots the number of communities and modularity across resolution values.

## References

- Ravasz E, Somera AL, Mongru DA, Oltvai ZN, Barabasi AL. [Hierarchical organization of modularity in metabolic networks](https://doi.org/10.1126/science.1073374). *Science* 297(5586), 1551-1555 (2002).
- Literature dataset record: `docs/research/network_biology_literature.json`, id 22.
