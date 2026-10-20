# Network Motifs: Detecting Feed-forward Loops in a Directed Regulatory Network

Uses the conventions of *Biological Interaction Networks: Node Degree and Degree Distribution* (exercise 93) for node lists and invalid input, but the graph is now **directed**. You search a gene regulatory network for one small recurring pattern, the feed-forward loop.

## Task and conventions

An edge `(u, v)` means "`u` regulates `v`" and is **not** the same as `(v, u)`. Nodes are unique `str` names. Repeated edges count once, self-loops (autoregulation) are ignored, and a duplicate node name or an unknown edge endpoint raises `ValueError`.

```python
find_feed_forward_loops(nodes, edges, induced=False) -> list[tuple[str, str, str]]
```

A **feed-forward loop (FFL)** is an ordered triple $(x,y,z)$ of three *distinct* nodes with

$$
x\to y,\qquad y\to z,\qquad x\to z.
$$

Here `x` is the top regulator, `y` the intermediate and `z` the target: `x` acts on `z` both directly and through `y`.

- `induced=False`: other edges among the three nodes are allowed.
- `induced=True`: the three edges must be the **only** edges among the three nodes, i.e. none of $y\to x$, $z\to x$, $z\to y$ exists.

Return every such triple, sorted ascending as tuples of strings. Roles are assigned from edge directions, never from names. With `induced=False`, one set of three nodes can give two triples when it also contains a reverse edge (for example `z -> y` in addition to the three FFL edges).

## Example

```python
nodes = ["TF1", "TF2", "G1"]
edges = [("TF1", "TF2"), ("TF2", "G1"), ("TF1", "G1")]
find_feed_forward_loops(nodes, edges)                      # [('TF1', 'TF2', 'G1')]
find_feed_forward_loops(nodes, edges + [("G1", "TF2")])    # [('TF1', 'G1', 'TF2'), ('TF1', 'TF2', 'G1')]
find_feed_forward_loops(nodes, edges + [("G1", "TF2")], induced=True)  # []
```

A chain `x->y->z`, a fan-out `x->y, x->z`, a fan-in `x->z, y->z` and a directed cycle `x->y->z->x` are **not** feed-forward loops. A cycle and a feed-forward loop have the same *undirected* skeleton, a triangle: of the eight ways to orient a triangle, six contain one loop and two (the cycles) contain none. Direction is the whole point.

## Before implementing

Store, for each node, the set of nodes it regulates. For each `x` and each target `y` of `x`, the candidates for `z` are the nodes regulated by both `y` and `x`. Loop in sorted order so the output is sorted. For `induced=True`, also reject triples with any of the three reverse edges.

## What a motif does and does not say

In the cited work, *network motifs* are interconnection patterns that occur in a real network much more often than in randomized networks with the same degrees. This exercise only **finds** a pattern; whether it is over-represented is a separate statistical question (the supplied experiment compares one count with degree-preserving randomizations, without claiming significance). Even an over-represented FFL is a structural observation: it does not show that the regulation is activating or repressing, that the loop is functional, or that it was selected for. Those claims need sign information, dynamics and experiments. The networks here are synthetic.

## Tests and experiment

T0 checks the return type. T1 checks documented roles and compares random directed graphs with an exhaustive ordered-triple oracle in both modes. T2 covers negative examples (chain, fan-out, fan-in, cycle, mutual edges), all eight orientations of a triangle, induced versus non-induced matching, overlapping loops, repeated edges and self-loops, validation and input-order independence. T3 runs your function on a synthetic 12-node regulatory network with three loops and compares its loop count with 200 seeded degree-preserving randomizations.

## References

- Milo R, Shen-Orr S, Itzkovitz S, Kashtan N, Chklovskii D, Alon U. [Network motifs: simple building blocks of complex networks](https://doi.org/10.1126/science.1078368). *Science* 298(5594), 824-827 (2002).
- Literature dataset record: `docs/research/network_biology_literature.json`, id 21.
