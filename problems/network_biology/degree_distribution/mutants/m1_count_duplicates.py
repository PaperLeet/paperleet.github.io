# BUG: 重复边与反向重复边被重复计数（用列表而非集合）。
def node_degrees(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    names = list(nodes)
    if len(set(names)) != len(names):
        raise ValueError("duplicate node name")
    known = set(names)
    neighbors = {name: [] for name in names}
    for u, v in edges:
        if u not in known or v not in known:
            raise ValueError(f"edge {(u, v)!r} uses an unknown node")
        if u != v:
            neighbors[u].append(v)
            neighbors[v].append(u)
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
