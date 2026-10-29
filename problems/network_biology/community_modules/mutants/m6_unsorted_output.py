# BUG: 输出社区没有按大小和最小成员排序。
def _graph(nodes, edges):
    names = list(nodes)
    if len(set(names)) != len(names):
        raise ValueError("duplicate node name")
    adjacency = {name: set() for name in names}
    for u, v in edges:
        if u not in adjacency or v not in adjacency:
            raise ValueError(f"edge {(u, v)!r} uses an unknown node")
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    return adjacency, sum(len(n) for n in adjacency.values()) // 2


def modularity(
    nodes: list[str], edges: list[tuple[str, str]], communities: list[list[str]], resolution: float = 1.0
) -> float:
    adjacency, m = _graph(nodes, edges)
    flat = [name for community in communities for name in community]
    if any(len(community) == 0 for community in communities) or sorted(flat) != sorted(adjacency):
        raise ValueError("communities must partition the nodes exactly once")
    if m == 0:
        return 0.0
    total = 0.0
    for community in communities:
        members = set(community)
        internal = sum(1 for u in members for v in adjacency[u] if v in members) // 2
        degree_sum = sum(len(adjacency[u]) for u in members)
        total += internal / m - resolution * (degree_sum / (2 * m)) ** 2
    return total


def greedy_modularity_communities(
    nodes: list[str], edges: list[tuple[str, str]], resolution: float = 1.0
) -> list[list[str]]:
    adjacency, m = _graph(nodes, edges)
    members = {name: [name] for name in adjacency}      # community id = its smallest member
    degree = {name: len(adjacency[name]) for name in adjacency}
    links = {name: {} for name in adjacency}            # links[a][b] = number of edges between a and b
    for u in adjacency:
        for v in adjacency[u]:
            links[u][v] = 1
    while m > 0:
        best = None
        for a in links:
            for b, between in links[a].items():
                if a < b:
                    gain = between / m - resolution * degree[a] * degree[b] / (2 * m * m)
                    key = (-round(gain, 12), a, b)
                    if best is None or key < best[0]:
                        best = (key, a, b, gain)
        if best is None or round(best[3], 12) <= 0:
            break
        _, a, b, _ = best
        members[a] += members.pop(b)
        degree[a] += degree.pop(b)
        for c, count in links.pop(b).items():
            del links[c][b]
            if c != a:
                links[a][c] = links[a].get(c, 0) + count
                links[c][a] = links[a][c]
    return [sorted(group) for group in members.values()]
