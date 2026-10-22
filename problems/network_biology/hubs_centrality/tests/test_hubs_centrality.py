import math
import pytest

A = ["A1", "A2", "A3", "A4"]
B = ["B1", "B2", "B3", "B4"]
BRIDGE_NODES = A + B + ["X"]
BRIDGE_EDGES = ([(u, v) for i, u in enumerate(A) for v in A[i + 1:]]
                + [(u, v) for i, u in enumerate(B) for v in B[i + 1:]] + [("A1", "X"), ("X", "B1")])
C4_NODES, C4_EDGES = ["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
PATH_NODES, PATH_EDGES = ["P1", "P2", "P3", "P4"], [("P1", "P2"), ("P2", "P3"), ("P3", "P4")]


def simple_paths(adj, source, target, seen=()):
    """All simple paths by exhaustive depth-first search (independent of BFS counting)."""
    if source == target:
        yield seen + (target,)
        return
    for nxt in sorted(adj[source]):
        if nxt not in seen and nxt != source:
            yield from simple_paths(adj, nxt, target, seen + (source,))


def oracle(nodes, edges):
    adj = {n: set() for n in nodes}
    for u, v in edges:
        if u != v:
            adj[u].add(v); adj[v].add(u)
    n = len(nodes)
    degree = {x: (len(adj[x]) / (n - 1) if n > 1 else 0.0) for x in nodes}
    between = {x: 0.0 for x in nodes}
    order = sorted(nodes)
    for i, s in enumerate(order):
        for t in order[i + 1:]:
            paths = list(simple_paths(adj, s, t))
            if not paths:
                continue
            shortest = [p for p in paths if len(p) == min(map(len, paths))]
            for p in shortest:
                for v in p[1:-1]:
                    between[v] += 1 / len(shortest)
    if n > 2:
        between = {x: v * 2 / ((n - 1) * (n - 2)) for x, v in between.items()}
    return degree, between


def close(got, expected):
    assert set(got) == set(expected), (sorted(got), sorted(expected))
    for key in expected:
        assert math.isclose(got[key], expected[key], rel_tol=1e-9, abs_tol=1e-9), (key, got[key], expected[key])


def lcg_graph(seed, size, tries):
    state, nodes, edges = seed, [f"G{i}" for i in range(size)], []
    for _ in range(tries):
        state = (1103515245 * state + 12345) % 2 ** 31
        u = nodes[(state >> 8) % size]
        state = (1103515245 * state + 12345) % 2 ** 31
        edges.append((u, nodes[(state >> 8) % size]))
    return nodes, edges


@pytest.mark.t0
def test_contract(impl):
    """Both centralities return a float per node; rank_hubs returns a list of names."""
    degree = impl.degree_centrality(["A", "B", "C"], [("A", "B"), ("B", "C")])
    between = impl.betweenness_centrality(["A", "B", "C"], [("A", "B"), ("B", "C")])
    assert isinstance(degree, dict) and set(degree) == {"A", "B", "C"} and all(isinstance(v, float) for v in degree.values())
    assert isinstance(between, dict) and set(between) == {"A", "B", "C"} and all(isinstance(v, float) for v in between.values())
    assert math.isclose(degree["B"], 1.0) and math.isclose(between["B"], 1.0) and between["A"] == 0.0
    ranked = impl.rank_hubs(degree, 2)
    assert isinstance(ranked, list) and ranked == ["B", "A"]


@pytest.mark.t1
def test_two_module_bridge_network_has_published_values(impl):
    """Hand-counted values: X lies on all 16 A-B shortest paths, A1 and B1 on 15 each."""
    degree = impl.degree_centrality(BRIDGE_NODES, BRIDGE_EDGES)
    between = impl.betweenness_centrality(BRIDGE_NODES, BRIDGE_EDGES)
    assert math.isclose(degree["A1"], 4 / 8) and math.isclose(degree["B1"], 4 / 8) and math.isclose(degree["X"], 2 / 8)
    assert math.isclose(degree["A2"], 3 / 8)
    assert math.isclose(between["X"], 16 / 28) and math.isclose(between["A1"], 15 / 28) and math.isclose(between["B1"], 15 / 28)
    assert all(between[n] == 0.0 for n in ("A2", "A3", "A4", "B2", "B3", "B4"))


@pytest.mark.t1
def test_matches_exhaustive_path_oracle(impl):
    """Centralities agree with an exhaustive enumeration of simple paths."""
    cases = [(BRIDGE_NODES, BRIDGE_EDGES), (C4_NODES, C4_EDGES), (PATH_NODES, PATH_EDGES)]
    cases += [lcg_graph(seed, size, tries) for seed, size, tries in [(1, 6, 8), (2, 7, 9), (3, 7, 16), (4, 8, 6), (5, 5, 12)]]
    for nodes, edges in cases:
        degree, between = oracle(nodes, edges)
        close(impl.degree_centrality(nodes, edges), degree)
        close(impl.betweenness_centrality(nodes, edges), between)


@pytest.mark.t2
def test_degree_and_betweenness_rankings_can_differ(impl):
    """The least-connected-looking connector has the highest betweenness."""
    degree = impl.degree_centrality(BRIDGE_NODES, BRIDGE_EDGES)
    between = impl.betweenness_centrality(BRIDGE_NODES, BRIDGE_EDGES)
    assert impl.rank_hubs(degree, 3) == ["A1", "B1", "A2"]
    assert impl.rank_hubs(between, 3) == ["X", "A1", "B1"]
    assert "X" not in impl.rank_hubs(degree, 8)


@pytest.mark.t2
def test_multiple_shortest_paths_share_credit(impl):
    """In a 4-cycle the two routes between opposite nodes split the credit equally."""
    between = impl.betweenness_centrality(C4_NODES, C4_EDGES)
    close(between, {"A": 1 / 6, "B": 1 / 6, "C": 1 / 6, "D": 1 / 6})
    chord = impl.betweenness_centrality(C4_NODES, C4_EDGES + [("A", "C")])
    # A chord A-C leaves B and D as the only pair with two routes (via A or via C).
    close(chord, {"A": 1 / 6, "B": 0.0, "C": 1 / 6, "D": 0.0})


@pytest.mark.t2
def test_small_isolated_and_disconnected_graphs(impl):
    """Empty, one-node, two-node, isolated-node and disconnected graphs are well defined."""
    assert impl.degree_centrality([], []) == {} and impl.betweenness_centrality([], []) == {}
    assert impl.degree_centrality(["A"], []) == {"A": 0.0} and impl.betweenness_centrality(["A"], []) == {"A": 0.0}
    assert impl.degree_centrality(["A", "B"], [("A", "B")]) == {"A": 1.0, "B": 1.0}
    assert impl.betweenness_centrality(["A", "B"], [("A", "B")]) == {"A": 0.0, "B": 0.0}
    nodes, edges = ["A", "B", "C", "D", "E"], [("A", "B"), ("B", "C")]
    close(impl.degree_centrality(nodes, edges), {"A": 1 / 4, "B": 2 / 4, "C": 1 / 4, "D": 0.0, "E": 0.0})
    close(impl.betweenness_centrality(nodes, edges), {"A": 0.0, "B": 1 / 6, "C": 0.0, "D": 0.0, "E": 0.0})
    triangle = impl.betweenness_centrality(["A", "B", "C"], [("A", "B"), ("B", "C"), ("A", "C")])
    assert all(v == 0.0 for v in triangle.values())


@pytest.mark.t2
def test_duplicates_self_loops_and_validation(impl):
    """Duplicate edges and self-loops do not change scores; invalid graphs raise ValueError."""
    clean = impl.degree_centrality(PATH_NODES, PATH_EDGES)
    noisy = PATH_EDGES + [("P2", "P1"), ("P2", "P3"), ("P4", "P4")]
    close(impl.degree_centrality(PATH_NODES, noisy), clean)
    close(impl.betweenness_centrality(PATH_NODES, noisy), impl.betweenness_centrality(PATH_NODES, PATH_EDGES))
    for bad in (lambda: impl.degree_centrality(["A"], [("A", "Z")]), lambda: impl.betweenness_centrality(["A", "A"], []),
                lambda: impl.rank_hubs({"A": 1.0}, -1)):
        with pytest.raises(ValueError):
            bad()


@pytest.mark.t2
def test_rank_hubs_tie_and_boundary_rules(impl):
    """Ties are broken by ascending name after 12-decimal rounding; k is clamped."""
    assert impl.rank_hubs({"B": 0.5, "A": 0.5, "C": 0.9}, 3) == ["C", "A", "B"]
    assert impl.rank_hubs({"b": 0.1 + 0.2, "a": 0.3}, 2) == ["a", "b"]
    assert impl.rank_hubs({"A": 1.0, "B": 2.0}, 10) == ["B", "A"]
    assert impl.rank_hubs({"A": 1.0}, 0) == [] and impl.rank_hubs({}, 3) == []
    ties = impl.betweenness_centrality(PATH_NODES, PATH_EDGES)
    assert impl.rank_hubs(ties, 4) == ["P2", "P3", "P1", "P4"]
    star = impl.degree_centrality(["C", "L1", "L2", "L3"], [("C", "L1"), ("C", "L2"), ("C", "L3")])
    assert impl.rank_hubs(star, 2) == ["C", "L1"]


@pytest.mark.t2
def test_scores_are_input_order_independent_and_bounded(impl):
    """Relabeling the input order keeps scores and rankings; normalized scores stay in [0, 1]."""
    nodes, edges = lcg_graph(9, 8, 14)
    base_d, base_b = impl.degree_centrality(nodes, edges), impl.betweenness_centrality(nodes, edges)
    flipped = [(v, u) for u, v in reversed(edges)]
    close(impl.degree_centrality(list(reversed(nodes)), flipped), base_d)
    close(impl.betweenness_centrality(list(reversed(nodes)), flipped), base_b)
    assert impl.rank_hubs(base_b, 8) == impl.rank_hubs(impl.betweenness_centrality(list(reversed(nodes)), flipped), 8)
    assert all(0.0 <= v <= 1.0 for v in list(base_d.values()) + list(base_b.values()))


@pytest.mark.t3
def test_hub_experiment(impl, artifact):
    """Run the two-module degree-versus-betweenness experiment with your functions."""
    from problems.network_biology.hubs_centrality.experiment import run
    artifact("toy_experiment", run(impl.degree_centrality, impl.betweenness_centrality, impl.rank_hubs))
