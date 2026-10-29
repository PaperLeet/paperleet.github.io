# Implementation notes

`modularity` can be computed community by community: count the edges inside, add up the degrees, and apply the formula. Only distinct non-loop edges count, so build neighbor sets first. Check that the communities form a partition before computing anything.

For the greedy algorithm maintain `members`, `degree` and `links` (community → neighbor community → number of edges). In each round compute the gain for every linked pair `(a, b)` with `a < b`, select the minimum of `(−round(gain, 12), a, b)` and stop if the best gain is not above zero. Merging `b` into `a` adds degrees and member lists, and for each neighbor `c` of `b` moves its edge count onto `a`. The merged community keeps the id `a`, which remains its smallest member because `a < b`.

Common mistakes:

- Forget to square the degree term, or count each internal edge twice.
- Ignore `resolution` in either function.
- Merge on a gain of exactly zero (the 4-cycle ends as two pairs, not one group).
- Merge communities that have no edge between them, or drop isolated nodes.
- Leave ties to dictionary order instead of the stated name rule.

## Complexity

Each round scans all linked community pairs, O(M) for M edges, and there are at most N − 1 rounds, so the reference runs in O(N·M). Faster algorithms exist (for example Louvain, or Clauset-Newman-Moore with a heap) but are not required.

## Why results vary

Greedy modularity is a heuristic. Different tie rules, merge orders and the resolution parameter can change the partition, and modularity itself cannot resolve very small modules in a large network. The planted-module graphs in the tests have an obvious optimum, which is why exact expectations are safe there; do not expect a greedy result on a general graph to be optimal.

## Scientific caution

A community is a structural statement about a graph. Calling it a "module" is a hypothesis that needs biological support such as enrichment of a shared annotation, replication in independent data, and experiments. Do not report detected communities as validated pathways.
