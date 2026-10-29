# BUG: 自环被当作普通边计入度数。
def node_degrees(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    names = list(nodes)
    if len(set(names)) != len(names):
        raise ValueError("duplicate node name")
    known = set(names)
    neighbors = {name: set() for name in names}
    for u, v in edges:
        if u not in known or v not in known:
            raise ValueError(f"edge {(u, v)!r} uses an unknown node")
        if True:
            neighbors[u].add(v)
            neighbors[v].add(u)
    return {name: len(neighbors[name]) for name in names}


def degree_distribution(nodes: list[str], edges: list[tuple[str, str]]) -> dict[int, float]:
    degrees = node_degrees(nodes, edges)
    if not degrees:
        return {}
    counts = [0] * (max(degrees.values()) + 1)
    for degree in degrees.values():
        counts[degree] += 1
    total = len(degrees)
    return {k: count / total for k, count in enumerate(counts)}
