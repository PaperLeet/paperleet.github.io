def node_degrees(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Degree of every node in a simple undirected graph.

    Raise ValueError for duplicate node names or for an edge endpoint that is
    not in `nodes`. Ignore self-loops; count each unordered pair only once.
    Isolated nodes must appear with degree 0.
    """
    raise NotImplementedError


def degree_distribution(nodes: list[str], edges: list[tuple[str, str]]) -> dict[int, float]:
    """Fraction P(k) of nodes with degree k, for every k = 0..max degree.

    Keys are consecutive integers in ascending order (absent degrees map to
    0.0). Return {} for a graph with no nodes.
    """
    raise NotImplementedError
