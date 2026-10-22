# BUG: 把路径两端点也计入介数。
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


def _bfs_counts(adjacency, source):
    distance, paths = {source: 0}, {source: 1}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                paths[neighbor] = 0
                queue.append(neighbor)
            if distance[neighbor] == distance[node] + 1:
                paths[neighbor] += paths[node]
    return distance, paths


def degree_centrality(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, float]:
    adjacency = _adjacency(nodes, edges)
    n = len(adjacency)
    if n == 1:
        return {name: 0.0 for name in adjacency}
    return {name: len(neighbors) / (n - 1) for name, neighbors in adjacency.items()}


def betweenness_centrality(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, float]:
    adjacency = _adjacency(nodes, edges)
    names = sorted(adjacency)
    n = len(names)
    score = {name: 0.0 for name in adjacency}
    if n < 3:
        return score
    distance, paths = {}, {}
    for name in names:
        distance[name], paths[name] = _bfs_counts(adjacency, name)
    for i, s in enumerate(names):
        for t in names[i + 1:]:
            if t not in distance[s]:
                continue
            for v in names:
                if v not in distance[s] or t not in distance[v]:
                    continue
                if distance[s][v] + distance[v][t] == distance[s][t]:
                    score[v] += paths[s][v] * paths[v][t] / paths[s][t]
    scale = 2 / ((n - 1) * (n - 2))
    return {name: value * scale for name, value in score.items()}


def rank_hubs(scores: dict[str, float], k: int) -> list[str]:
    if k < 0:
        raise ValueError("k must be non-negative")
    ordered = sorted(scores, key=lambda name: (-round(scores[name], 12), name))
    return ordered[:k]
