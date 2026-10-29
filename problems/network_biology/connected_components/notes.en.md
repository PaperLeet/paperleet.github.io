# Implementation notes

Build neighbor sets, loop over nodes in sorted order, and grow each component with a stack (depth-first) or queue (breadth-first). Mark a node as visited when you *push* it, so it is never pushed twice. Sort each component and then sort the list of components by `(-size, first name)`.

Common mistakes:

- Forget isolated nodes because they appear in no edge.
- Record each edge in one direction only, so the result depends on which endpoint was listed first.
- Sort ties by the wrong node (largest name, or discovery order).
- Divide the largest size by the number of components instead of the number of nodes.
- Recurse on a long chain and exceed Python's recursion limit; prefer an explicit stack.

## Complexity

Building the graph and one traversal of every node are O(N + M) in total, plus O(N log N) for sorting.

## Scientific caution

A component is a statement about the recorded graph. The share of the largest component depends on how complete and how biased the recorded data are. Disconnected components should be read as "no recorded route", never as proof that two proteins do not influence each other.
