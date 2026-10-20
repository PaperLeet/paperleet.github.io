# Implementation notes

Keep one set of regulated targets per node and drop self-loops while reading the edges (they would otherwise let a node appear twice in a triple). Then for each `x` in sorted order, each `y` in `targets[x]` in sorted order, and each `z` in `targets[y] & targets[x]` in sorted order, record `(x, y, z)`. Distinctness is automatic once self-loops are gone: `y != x` because `y` is a target of `x`, and `z` differs from both because it is a target of both `x` and `y`.

For `induced=True`, skip a triple if any of `x in targets[y]`, `x in targets[z]` or `y in targets[z]` holds.

Common mistakes:

- Store edges in both directions, which turns every directed cycle into a "loop".
- Check only the chain `x->y->z` and forget the shortcut `x->z`.
- Keep self-loops, producing triples such as `(x, x, z)`.
- Check only one of the three reverse edges in induced mode.
- Return triples in set-iteration order.

## Complexity

The search costs the sum over edges `(x, y)` of the size of `targets[y]`, at most O(M · d_max) for M edges and maximum out-degree d_max. Checking all ordered triples of nodes (the test oracle) is O(N^3).

## Scientific caution

Finding an FFL is subgraph matching. "Motif" in the literature adds a statistical statement: the pattern is more frequent than in suitable randomized networks. Even then, regulatory function is not established by topology: it depends on edge signs, logic at the target, and kinetics, and must be tested experimentally.
