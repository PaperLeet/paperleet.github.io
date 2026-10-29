def find_feed_forward_loops(
    nodes: list[str], edges: list[tuple[str, str]], induced: bool = False
) -> list[tuple[str, str, str]]:
    """All feed-forward loops (x, y, z) in a directed regulatory network.

    An edge (u, v) means "u regulates v". A feed-forward loop is an ordered
    triple of three DISTINCT nodes with edges x->y, y->z and x->z, where x is
    the top regulator, y the intermediate and z the target.

    If induced is False, other edges among the three nodes are allowed. If
    induced is True, the three edges must be the ONLY edges among the three
    nodes (no y->x, z->x or z->y).

    Ignore self-loops and repeated edges (they never make a triple). Return the
    triples sorted ascending as tuples of strings. Raise ValueError for
    duplicate node names or an edge endpoint that is not in `nodes`.
    """
    raise NotImplementedError
