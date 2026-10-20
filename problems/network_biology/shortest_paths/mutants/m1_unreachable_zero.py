# BUG: 不可达目标返回 0 而不是 None。
from collections import deque


def _adjacency(nodes, edges):
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
    return adjacency


def _search(nodes, edges, source):
    """Breadth-first search with sorted neighbors: distances and first-discovery parents."""
    adjacency = _adjacency(nodes, edges)
    if source not in adjacency:
        raise ValueError(f"unknown source {source!r}")
    distance, parent = {source: 0}, {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                parent[neighbor] = node
                queue.append(neighbor)
    return adjacency, distance, parent


def distances_from(nodes: list[str], edges: list[tuple[str, str]], source: str) -> dict[str, int]:
    return _search(nodes, edges, source)[1]


def shortest_path_length(nodes: list[str], edges: list[tuple[str, str]], source: str, target: str) -> int | None:
    adjacency, distance, _ = _search(nodes, edges, source)
    if target not in adjacency:
        raise ValueError(f"unknown target {target!r}")
    return distance.get(target, 0)


def shortest_path(nodes: list[str], edges: list[tuple[str, str]], source: str, target: str) -> list[str] | None:
    adjacency, distance, parent = _search(nodes, edges, source)
    if target not in adjacency:
        raise ValueError(f"unknown target {target!r}")
    if target not in distance:
        return None
    path = [target]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path
