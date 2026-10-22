def degree_centrality(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, float]:
    """Normalized degree centrality: (number of distinct neighbors) / (N - 1).

    Simple undirected graph: ignore self-loops, count each unordered pair once.
    Return 0.0 for every node when N == 1 and {} when there are no nodes.
    Raise ValueError for duplicate node names or unknown edge endpoints.
    """
    raise NotImplementedError


def betweenness_centrality(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, float]:
    """Normalized shortest-path betweenness centrality of every node.

    For each unordered pair {s, t} of *other* nodes that are connected, a node v
    receives sigma_st(v) / sigma_st, where sigma_st counts shortest s-t paths and
    sigma_st(v) counts those that pass through v. Divide the sum by
    (N - 1)(N - 2) / 2. Unconnected pairs contribute nothing; endpoints of a
    path are never credited. Return 0.0 for every node when N < 3.
    """
    raise NotImplementedError


def rank_hubs(scores: dict[str, float], k: int) -> list[str]:
    """The k highest-scoring node names, highest first.

    Compare scores after rounding to 12 decimal places; break ties by node
    name in ascending string order. k > len(scores) returns all nodes, k == 0
    returns [], and k < 0 raises ValueError.
    """
    raise NotImplementedError
