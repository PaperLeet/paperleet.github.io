import pytest

TOY_NODES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
TOY_EDGES = [("P1", "P2"), ("P1", "P3"), ("P2", "P4"), ("P3", "P4"), ("P4", "P5"), ("P6", "P7")]


def adjacency(nodes, edges):
    adj = {n: set() for n in nodes}
    for u, v in edges:
        if u != v:
            adj[u].add(v); adj[v].add(u)
    return adj


def floyd(nodes, edges):
    """Independent oracle for distances: Floyd-Warshall."""
    inf = float("inf")
    d = {a: {b: (0 if a == b else inf) for b in nodes} for a in nodes}
    for u, v in edges:
        if u != v:
            d[u][v] = d[v][u] = 1
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def smallest_shortest_path(adj, s, t, seen=()):
    """Independent oracle for the tie rule: exhaustive simple paths, shortest, then min."""
    def walk(node, path):
        if node == t:
            yield path
            return
        for nxt in adj[node]:
            if nxt not in path:
                yield from walk(nxt, path + [nxt])
    paths = list(walk(s, [s]))
    if not paths:
        return None
    best = min(map(len, paths))
    return min(p for p in paths if len(p) == best)


def lcg_graph(seed, size, tries):
    state, nodes, edges = seed, [f"G{i:02d}" for i in range(size)], []
    for _ in range(tries):
        state = (1103515245 * state + 12345) % 2 ** 31
        u = nodes[(state >> 8) % size]
        state = (1103515245 * state + 12345) % 2 ** 31
        edges.append((u, nodes[(state >> 8) % size]))
    return nodes, edges


@pytest.mark.t0
def test_contract(impl):
    """Distances are a dict of ints, lengths are int or None, paths are lists of names or None."""
    nodes, edges = ["A", "B", "C"], [("A", "B")]
    d = impl.distances_from(nodes, edges, "A")
    assert isinstance(d, dict) and d == {"A": 0, "B": 1}
    assert type(impl.shortest_path_length(nodes, edges, "A", "B")) is int
    assert impl.shortest_path_length(nodes, edges, "A", "C") is None
    assert impl.shortest_path(nodes, edges, "A", "B") == ["A", "B"]
    assert impl.shortest_path(nodes, edges, "A", "C") is None


@pytest.mark.t1
def test_documented_toy_network(impl):
    """P1 to P5 has two shortest paths; the lexicographically smallest is returned."""
    assert impl.distances_from(TOY_NODES, TOY_EDGES, "P1") == {"P1": 0, "P2": 1, "P3": 1, "P4": 2, "P5": 3}
    assert impl.shortest_path_length(TOY_NODES, TOY_EDGES, "P1", "P5") == 3
    assert impl.shortest_path(TOY_NODES, TOY_EDGES, "P1", "P5") == ["P1", "P2", "P4", "P5"]
    assert impl.shortest_path_length(TOY_NODES, TOY_EDGES, "P1", "P6") is None
    assert impl.shortest_path(TOY_NODES, TOY_EDGES, "P1", "P7") is None


@pytest.mark.t1
def test_matches_floyd_warshall_and_exhaustive_path_oracles(impl):
    """Distances match Floyd-Warshall; paths match an exhaustive lexicographic-minimum search."""
    for seed, size, tries in [(1, 7, 8), (2, 8, 11), (3, 6, 4), (4, 8, 16), (5, 9, 10)]:
        nodes, edges = lcg_graph(seed, size, tries)
        dist, adj = floyd(nodes, edges), adjacency(nodes, edges)
        for s in nodes:
            got = impl.distances_from(nodes, edges, s)
            assert got == {t: d for t, d in dist[s].items() if d != float("inf")}, (seed, s)
            for t in nodes:
                expected = None if dist[s][t] == float("inf") else dist[s][t]
                assert impl.shortest_path_length(nodes, edges, s, t) == expected, (seed, s, t)
                assert impl.shortest_path(nodes, edges, s, t) == smallest_shortest_path(adj, s, t), (seed, s, t)


@pytest.mark.t2
def test_source_equals_target_and_unreachable_targets(impl):
    """source == target is distance 0 even for isolated nodes; unreachable targets are None."""
    nodes = ["A", "B", "C", "D"]
    edges = [("A", "B")]
    assert impl.shortest_path_length(nodes, edges, "C", "C") == 0
    assert impl.shortest_path(nodes, edges, "C", "C") == ["C"]
    assert impl.distances_from(nodes, edges, "C") == {"C": 0}
    assert impl.shortest_path_length(nodes, edges, "A", "D") is None
    assert impl.shortest_path(nodes, edges, "D", "A") is None


@pytest.mark.t2
def test_deterministic_tie_handling(impl):
    """Edge order, node order and neighbor discovery order never change the returned path."""
    nodes = ["S", "A", "B", "C", "T"]
    edges = [("S", "C"), ("S", "B"), ("S", "A"), ("C", "T"), ("B", "T"), ("A", "T")]
    for order in (edges, list(reversed(edges)), sorted(edges, key=lambda e: e[1])):
        assert impl.shortest_path(nodes, order, "S", "T") == ["S", "A", "T"]
        assert impl.shortest_path(list(reversed(nodes)), [(v, u) for u, v in order], "S", "T") == ["S", "A", "T"]
    # Lexicographic order compares paths element by element, longer shared prefixes first.
    nodes = ["S", "A", "B", "X", "Y", "T"]
    edges = [("S", "A"), ("S", "B"), ("A", "Y"), ("B", "X"), ("X", "T"), ("Y", "T")]
    assert impl.shortest_path(nodes, edges, "S", "T") == ["S", "A", "Y", "T"]


@pytest.mark.t2
def test_paths_are_valid_shortest_simple_paths(impl):
    """Every returned path starts and ends correctly, follows real edges, and has length d(s, t)."""
    nodes, edges = lcg_graph(21, 12, 20)
    adj = adjacency(nodes, edges)
    for s in nodes:
        dist = impl.distances_from(nodes, edges, s)
        for t in nodes:
            path = impl.shortest_path(nodes, edges, s, t)
            if t not in dist:
                assert path is None
                continue
            assert path[0] == s and path[-1] == t and len(set(path)) == len(path)
            assert all(b in adj[a] for a, b in zip(path, path[1:]))
            assert len(path) - 1 == dist[t] == impl.shortest_path_length(nodes, edges, s, t)


@pytest.mark.t2
def test_undirected_symmetry_and_triangle_inequality(impl):
    """Lengths are symmetric and obey d(s,u) <= d(s,v) + d(v,u)."""
    nodes, edges = lcg_graph(33, 10, 14)
    d = {s: impl.distances_from(nodes, edges, s) for s in nodes}
    for s in nodes:
        for t in nodes:
            assert d[s].get(t) == d[t].get(s)
            for v in nodes:
                if v in d[s] and t in d[v]:
                    assert d[s][t] <= d[s][v] + d[v][t]


@pytest.mark.t2
def test_noise_and_validation(impl):
    """Repeated edges and self-loops are ignored; unknown names raise ValueError."""
    nodes = ["A", "B", "C"]
    noisy = [("A", "B"), ("B", "A"), ("B", "B"), ("B", "C"), ("C", "C")]
    assert impl.distances_from(nodes, noisy, "A") == {"A": 0, "B": 1, "C": 2}
    for bad in (lambda: impl.distances_from(nodes, noisy, "Z"),
                lambda: impl.shortest_path_length(nodes, noisy, "A", "Z"),
                lambda: impl.shortest_path(nodes, noisy, "Z", "A"),
                lambda: impl.shortest_path(nodes, noisy, "A", "Z"),
                lambda: impl.distances_from(nodes, [("A", "Q")], "A"),
                lambda: impl.distances_from(["A", "A"], [], "A")):
        with pytest.raises(ValueError):
            bad()


@pytest.mark.t3
def test_proximity_experiment(impl, artifact):
    """Run the distance and shortcut-edge experiment with your functions."""
    from problems.network_biology.shortest_paths.experiment import run
    artifact("toy_experiment", run(impl.distances_from, impl.shortest_path_length, impl.shortest_path))
