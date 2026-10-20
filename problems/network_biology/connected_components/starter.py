def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components of a simple undirected graph.

    Each component is a list of node names sorted ascending. Return the
    components ordered by size (largest first), ties by the smallest node
    name. Isolated nodes are components of size one; no nodes gives [].
    Ignore self-loops and repeated edges. Raise ValueError for duplicate node
    names or an edge endpoint that is not in `nodes`.
    """
    raise NotImplementedError


def largest_component(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """The sorted node names of the largest connected component ([] if no nodes).

    If several components have the same size, the one whose smallest node name
    comes first in ascending string order wins.
    """
    raise NotImplementedError


def largest_component_fraction(nodes: list[str], edges: list[tuple[str, str]]) -> float:
    """Size of the largest component divided by the number of nodes (0.0 if none)."""
    raise NotImplementedError
