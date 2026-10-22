def distances_from(nodes: list[str], edges: list[tuple[str, str]], source: str) -> dict[str, int]:
    """Number of edges on a shortest path from `source` to every reachable node.

    Unweighted, undirected graph: every edge has length 1 and edge weights, if
    you had them, are ignored. `source` maps to 0. Unreachable nodes are
    omitted. Ignore self-loops and repeated edges. Raise ValueError for duplicate
    node names, an edge endpoint that is not in `nodes`, or a `source` that is
    not in `nodes`.
    """
    raise NotImplementedError


def shortest_path_length(nodes: list[str], edges: list[tuple[str, str]], source: str, target: str) -> int | None:
    """Edge count of a shortest source-target path, or None if target is unreachable.

    source == target gives 0. Raise ValueError if source or target is not in
    `nodes` (plus the validation errors of distances_from).
    """
    raise NotImplementedError


def shortest_path(nodes: list[str], edges: list[tuple[str, str]], source: str, target: str) -> list[str] | None:
    """One shortest path as a list of node names from source to target, or None.

    source == target gives [source]. When several shortest paths exist, return
    the lexicographically smallest list of node names (compare the paths
    element by element as strings).
    """
    raise NotImplementedError
