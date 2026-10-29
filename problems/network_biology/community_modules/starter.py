def modularity(
    nodes: list[str], edges: list[tuple[str, str]], communities: list[list[str]], resolution: float = 1.0
) -> float:
    """Newman modularity Q of a partition of a simple undirected graph.

    Q = sum over communities c of ( L_c / m - resolution * (D_c / (2 m))**2 ),
    where m is the number of distinct non-loop edges, L_c the number of edges
    with both ends in c and D_c the sum of the degrees of the nodes in c.
    Return 0.0 when the graph has no edges. `communities` must be a partition
    of `nodes`: nonempty lists, disjoint, covering every node exactly once;
    otherwise raise ValueError. Also raise ValueError for duplicate nodes or
    unknown edge endpoints. Ignore self-loops and repeated edges.
    """
    raise NotImplementedError


def greedy_modularity_communities(
    nodes: list[str], edges: list[tuple[str, str]], resolution: float = 1.0
) -> list[list[str]]:
    """Greedy agglomerative modularity maximization (deterministic).

    Start with every node in its own community. Repeatedly merge the pair of
    communities joined by at least one edge whose merge gain
        dQ = e_ab / m - resolution * D_a * D_b / (2 m**2)
    is largest (e_ab: edges between a and b), as long as the best gain, rounded
    to 12 decimals, is strictly positive. Ties in the rounded gain go to the
    pair with the smallest (smallest-member-of-a, smallest-member-of-b) name pair,
    with a's smallest member < b's. Communities without edges between them are
    never merged, so isolated nodes stay alone and a graph with no edges returns
    singletons.

    Return communities as sorted lists of names, ordered by size (largest
    first), ties by smallest member. No nodes gives [].
    """
    raise NotImplementedError
