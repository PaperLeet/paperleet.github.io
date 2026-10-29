import math
import pytest

TOY_NODES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
TOY_EDGES = [("P1", "P2"), ("P1", "P3"), ("P1", "P4"), ("P1", "P5"),
             ("P2", "P3"), ("P5", "P6"), ("P6", "P7")]
TOY_DEGREES = {"P1": 4, "P2": 2, "P3": 2, "P4": 1, "P5": 2, "P6": 2, "P7": 1, "P8": 0}
TOY_DISTRIBUTION = {0: 1 / 8, 1: 2 / 8, 2: 4 / 8, 3: 0.0, 4: 1 / 8}


def matrix_degrees(nodes, edges):
    """Independent oracle: row sums of a symmetric 0/1 adjacency matrix without loops."""
    index = {name: i for i, name in enumerate(nodes)}
    matrix = [[0] * len(nodes) for _ in nodes]
    for u, v in edges:
        if u != v:
            matrix[index[u]][index[v]] = matrix[index[v]][index[u]] = 1
    return {name: sum(matrix[index[name]]) for name in nodes}


def lcg_graph(seed, size, tries):
    """Deterministic pseudo-random graph with duplicates, reversed pairs and self-loops."""
    state, nodes = seed, [f"G{i:02d}" for i in range(size)]
    edges = []
    for _ in range(tries):
        state = (1103515245 * state + 12345) % 2 ** 31
        u = nodes[(state >> 8) % size]
        state = (1103515245 * state + 12345) % 2 ** 31
        v = nodes[(state >> 8) % size]
        edges.append((u, v))
    return nodes, edges


@pytest.mark.t0
def test_contract(impl):
    """Both functions return plain dicts with int degrees and float probabilities."""
    degrees = impl.node_degrees(["A", "B"], [("A", "B")])
    assert isinstance(degrees, dict) and degrees == {"A": 1, "B": 1}
    assert all(type(v) is int for v in degrees.values())
    distribution = impl.degree_distribution(["A", "B"], [("A", "B")])
    assert isinstance(distribution, dict) and set(distribution) == {0, 1}
    assert all(isinstance(k, int) and isinstance(v, float) for k, v in distribution.items())
    assert math.isclose(distribution[1], 1.0) and distribution[0] == 0.0


@pytest.mark.t1
def test_toy_network_expected_values(impl):
    """The documented 8-node toy network has the published degrees and P(k)."""
    assert impl.node_degrees(TOY_NODES, TOY_EDGES) == TOY_DEGREES
    distribution = impl.degree_distribution(TOY_NODES, TOY_EDGES)
    assert list(distribution) == [0, 1, 2, 3, 4]
    for k, expected in TOY_DISTRIBUTION.items():
        assert math.isclose(distribution[k], expected, abs_tol=1e-12), (k, distribution)


@pytest.mark.t1
def test_matches_adjacency_matrix_oracle(impl):
    """Pseudo-random graphs with noise edges agree with matrix row sums."""
    for seed, size, tries in [(1, 6, 10), (2, 12, 40), (3, 9, 5), (4, 20, 60)]:
        nodes, edges = lcg_graph(seed, size, tries)
        expected = matrix_degrees(nodes, edges)
        assert impl.node_degrees(nodes, edges) == expected, (seed, size, tries)
        counts = {}
        for d in expected.values():
            counts[d] = counts.get(d, 0) + 1
        distribution = impl.degree_distribution(nodes, edges)
        for k in range(max(expected.values()) + 1):
            assert math.isclose(distribution[k], counts.get(k, 0) / size, abs_tol=1e-12), (seed, k)


@pytest.mark.t2
def test_isolated_nodes_and_minimal_graphs(impl):
    """Isolated nodes keep degree 0; empty and one-node graphs are valid."""
    assert impl.node_degrees([], []) == {}
    assert impl.degree_distribution([], []) == {}
    assert impl.node_degrees(["A"], []) == {"A": 0}
    assert impl.degree_distribution(["A"], []) == {0: 1.0}
    degrees = impl.node_degrees(["A", "B", "C"], [("A", "B")])
    assert degrees == {"A": 1, "B": 1, "C": 0}
    distribution = impl.degree_distribution(["A", "B", "C"], [("A", "B")])
    assert math.isclose(distribution[0], 1 / 3) and math.isclose(distribution[1], 2 / 3)


@pytest.mark.t2
def test_duplicate_edges_and_self_loops_follow_simple_graph_rule(impl):
    """Parallel edges, reversed duplicates and self-loops never inflate a degree."""
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("A", "B"), ("B", "A"), ("C", "C"), ("B", "B")]
    assert impl.node_degrees(nodes, edges) == {"A": 1, "B": 1, "C": 0}
    distribution = impl.degree_distribution(nodes, edges)
    assert list(distribution) == [0, 1]
    assert math.isclose(distribution[0], 1 / 3) and math.isclose(distribution[1], 2 / 3)


@pytest.mark.t2
def test_invalid_graphs_raise_value_error(impl):
    """An endpoint outside `nodes`, or a repeated node name, is rejected."""
    with pytest.raises(ValueError):
        impl.node_degrees(["A", "B"], [("A", "Z")])
    with pytest.raises(ValueError):
        impl.node_degrees(["A", "B"], [("Z", "A")])
    with pytest.raises(ValueError):
        impl.node_degrees(["A", "A"], [])
    with pytest.raises(ValueError):
        impl.degree_distribution(["A", "B"], [("A", "Z")])


@pytest.mark.t2
def test_distribution_is_a_normalized_dense_histogram(impl):
    """P(k) sums to one, keys are 0..max degree, and the mean degree equals 2m/N."""
    nodes, edges = lcg_graph(5, 15, 30)
    degrees = impl.node_degrees(nodes, edges)
    distribution = impl.degree_distribution(nodes, edges)
    assert list(distribution) == list(range(max(degrees.values()) + 1))
    assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12)
    assert all(p >= 0 for p in distribution.values())
    unique_edges = {frozenset(e) for e in edges if e[0] != e[1]}
    assert sum(degrees.values()) == 2 * len(unique_edges)
    mean = sum(k * p for k, p in distribution.items())
    assert math.isclose(mean, 2 * len(unique_edges) / len(nodes), abs_tol=1e-12)


@pytest.mark.t2
def test_undirected_and_order_independent(impl):
    """Edge direction, edge order and node order do not change the answer."""
    nodes, edges = lcg_graph(6, 10, 25)
    base = impl.node_degrees(nodes, edges)
    flipped = [(v, u) for u, v in reversed(edges)]
    assert impl.node_degrees(list(reversed(nodes)), flipped) == base
    assert impl.degree_distribution(list(reversed(nodes)), flipped) == impl.degree_distribution(nodes, edges)
    assert impl.node_degrees(["A", "B"], [("A", "B")]) == impl.node_degrees(["A", "B"], [("B", "A")])


@pytest.mark.t3
def test_degree_distribution_experiment(impl, artifact):
    """Run the synthetic toy and growth-network degree experiments with your functions."""
    from problems.network_biology.degree_distribution.experiment import run
    artifact("toy_experiment", run(impl.node_degrees, impl.degree_distribution))
