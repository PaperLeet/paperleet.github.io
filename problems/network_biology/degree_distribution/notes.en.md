# Implementation notes

Use one `set` of neighbors per node. Insert both directions of every non-loop edge; the set removes repeated and reversed duplicates. Validate endpoints while reading the edges, and validate the node list before building anything else.

Common mistakes:

- Count a repeated or reversed edge twice (a list of neighbors instead of a set).
- Count a self-loop as a normal edge.
- Report only degrees that occur, so a missing degree (here `k=3`) silently disappears from the histogram.
- Divide by the number of edges instead of the number of nodes.
- Drop isolated nodes because they never appear in an edge.

## Complexity and design

Building the neighbor sets takes O(N + M) time and space for N nodes and M listed edges. The histogram takes O(N + k_max). `degree_distribution` is a function of `node_degrees`, which is the point of the exercise: the distribution forgets *which* protein has which degree.

The dense histogram (keys `0..max degree`) is a convention chosen so that two networks can be compared position by position and plotted without gaps. The handshake identity gives a quick self-check: the degrees sum to twice the number of distinct non-loop edges, so the mean degree is 2M/N.

## Scientific caution

Degree counts recorded interactions only. A high-degree node can be popular in the literature, easy to detect by the assay, or genuinely promiscuous; the graph alone cannot tell these apart. Do not read a degree as importance or essentiality.
