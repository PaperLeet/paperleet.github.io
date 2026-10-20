# Implementation notes

Run a breadth-first search from `source`, expanding each node's neighbors in sorted order and recording `distance` and `parent` the first time a node is seen. Distances come straight from the search; a path is rebuilt by following parents from the target back to the source and reversing the list.

Why this gives the lexicographically smallest path: BFS processes a layer in the lexicographic order of its nodes' best paths (by induction), and each node's parent is the first node in that order that is adjacent to it, so appending the node to the parent's best path yields its best path.

Common mistakes:

- Return `0` or an exception for an unreachable target instead of `None`.
- Report the number of nodes on a path instead of the number of edges.
- Iterate over a `set` without sorting, which makes ties depend on hash order.
- Forget to reverse the parent chain, or return an empty path for `source == target`.
- Treat edges as directed, or skip validation of the target.

## Complexity

One BFS is O(N + M) (plus the sorting of neighbor lists). All-pairs distances need N searches; the exhaustive path oracle in the tests is only usable on tiny graphs.

## Scientific caution

A short path is a statement about the recorded graph. It does not show that two proteins act together: promiscuous hubs create short paths between unrelated processes, and missing interactions create long or absent ones. Treat proximity as a hypothesis to be tested, not a conclusion.
