# Implementation notes

Build the neighbor sets once and reuse them. A readable betweenness implementation runs a breadth-first search (BFS) from every node and records both the distance and the number of shortest paths $\sigma$ to every other node; BFS gives $\sigma$ by adding the parent's count whenever a neighbor lies exactly one level further. Then loop over unordered pairs $\{s,t\}$ and credit each node `v` with $\sigma_{sv}\sigma_{vt}/\sigma_{st}$ when `dist(s,v) + dist(v,t) = dist(s,t)`.

Common mistakes:

- Divide degree by `N` instead of `N - 1`.
- Count ordered pairs, which doubles the score without the factor 2 in the normalizer.
- Credit the endpoints of a path.
- Add `1` when `v` is on *a* shortest path instead of the fraction of shortest paths through `v` (wrong whenever routes tie, e.g. a 4-cycle).
- Skip the tie rule, or compare raw floats so that `0.1 + 0.2` and `0.3` look different.
- Treat unconnected pairs as having `sigma = 0` and divide by zero.

## Complexity

The pair loop above is O(N^3) after N breadth-first searches of O(N + M) each. Brandes' algorithm accumulates dependencies backwards over each BFS and runs in O(NM) time for unweighted graphs; you may use it as long as the contract above is met.

## Why the two rankings differ

Degree only looks at a node's immediate neighborhood, so it ranks `A1` and `B1` first. Betweenness looks at global routes, so the connector `X` wins even with a low degree. Adding the shortcut `A4-B4` gives the modules a second route, which lowers `X`'s betweenness: a one-edge change in the graph moves a global score.

## Scientific caution

Which score is "right" depends on the question. Neither measures essentiality; whether a hub is essential, a bottleneck is a good intervention target, or a connector matters physiologically must be tested experimentally. Centrality in a toy graph says nothing about a real interactome.
