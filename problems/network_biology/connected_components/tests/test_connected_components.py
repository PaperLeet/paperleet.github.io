import math
import pytest

TOY_NODES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
TOY_EDGES = [("P1", "P2"), ("P2", "P3"), ("P3", "P1"), ("P3", "P4"), ("P5", "P6"), ("P6", "P7")]
TOY_COMPONENTS = [["P1", "P2", "P3", "P4"], ["P5", "P6", "P7"], ["P8"], ["P9"]]


def union_find_components(nodes, edges):
    """Independent oracle: disjoint-set union instead of graph traversal."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges:
        parent[find(u)] = find(v)
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


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
    """Components are lists of sorted names; the largest component is one such list."""
    components = impl.connected_components(["B", "A", "C"], [("A", "B")])
    assert isinstance(components, list) and components == [["A", "B"], ["C"]]
    assert all(isinstance(c, list) for c in components)
    assert impl.largest_component(["B", "A", "C"], [("A", "B")]) == ["A", "B"]
    fraction = impl.largest_component_fraction(["B", "A", "C"], [("A", "B")])
    assert isinstance(fraction, float) and math.isclose(fraction, 2 / 3)


@pytest.mark.t1
def test_documented_toy_network(impl):
    """The 9-node toy network has components of sizes 4, 3, 1 and 1."""
    assert impl.connected_components(TOY_NODES, TOY_EDGES) == TOY_COMPONENTS
    assert impl.largest_component(TOY_NODES, TOY_EDGES) == ["P1", "P2", "P3", "P4"]
    assert math.isclose(impl.largest_component_fraction(TOY_NODES, TOY_EDGES), 4 / 9)


@pytest.mark.t1
def test_matches_union_find_oracle(impl):
    """Pseudo-random sparse graphs agree with a disjoint-set oracle."""
    for seed, size, tries in [(1, 10, 6), (2, 14, 9), (3, 8, 3), (4, 20, 18), (5, 12, 30), (6, 9, 0)]:
        nodes, edges = lcg_graph(seed, size, tries)
        expected = union_find_components(nodes, edges)
        assert impl.connected_components(nodes, edges) == expected, (seed, size, tries)
        assert impl.largest_component(nodes, edges) == expected[0]
        assert math.isclose(impl.largest_component_fraction(nodes, edges), len(expected[0]) / size)


@pytest.mark.t2
def test_isolated_nodes_and_empty_graphs(impl):
    """Isolated nodes are singleton components; empty graphs have no components."""
    assert impl.connected_components([], []) == []
    assert impl.largest_component([], []) == []
    assert impl.largest_component_fraction([], []) == 0.0
    assert impl.connected_components(["A"], []) == [["A"]]
    assert impl.connected_components(["C", "B", "A"], []) == [["A"], ["B"], ["C"]]
    assert impl.largest_component(["C", "B", "A"], []) == ["A"]
    assert math.isclose(impl.largest_component_fraction(["C", "B", "A"], []), 1 / 3)


@pytest.mark.t2
def test_largest_component_tie_rule(impl):
    """Equal-size components are ordered by their smallest node name."""
    nodes = ["Z", "B", "C", "A", "Y"]
    edges = [("A", "Z"), ("B", "C")]
    assert impl.connected_components(nodes, edges) == [["A", "Z"], ["B", "C"], ["Y"]]
    assert impl.largest_component(nodes, edges) == ["A", "Z"]
    assert impl.largest_component(list(reversed(nodes)), list(reversed(edges))) == ["A", "Z"]


@pytest.mark.t2
def test_edges_are_undirected_and_noise_is_ignored(impl):
    """Direction, repeated edges and self-loops do not change components."""
    nodes = ["A", "B", "C", "D"]
    assert impl.connected_components(nodes, [("B", "A"), ("C", "B")]) == [["A", "B", "C"], ["D"]]
    assert impl.connected_components(nodes, [("A", "B"), ("C", "B")]) == [["A", "B", "C"], ["D"]]
    noisy = [("A", "B"), ("B", "A"), ("A", "B"), ("D", "D"), ("C", "C")]
    assert impl.connected_components(nodes, noisy) == [["A", "B"], ["C"], ["D"]]


@pytest.mark.t2
def test_invalid_graphs_raise_value_error(impl):
    """Unknown endpoints and repeated node names are rejected."""
    for bad in (lambda: impl.connected_components(["A"], [("A", "Z")]),
                lambda: impl.largest_component(["A", "B"], [("Q", "A")]),
                lambda: impl.largest_component_fraction(["A", "A"], [])):
        with pytest.raises(ValueError):
            bad()


@pytest.mark.t2
def test_components_partition_the_graph(impl):
    """Components partition the nodes, are internally connected, and no edge crosses them."""
    nodes, edges = lcg_graph(11, 18, 14)
    components = impl.connected_components(nodes, edges)
    flat = [n for c in components for n in c]
    assert sorted(flat) == sorted(nodes) and len(flat) == len(set(flat))
    owner = {n: i for i, c in enumerate(components) for n in c}
    assert all(owner[u] == owner[v] for u, v in edges)
    sizes = [len(c) for c in components]
    assert sizes == sorted(sizes, reverse=True) and sum(sizes) == len(nodes)
    assert all(c == sorted(c) for c in components)
    assert impl.connected_components(list(reversed(nodes)), list(reversed(edges))) == components


@pytest.mark.t2
def test_one_bridge_edge_changes_the_largest_component(impl):
    """Adding or removing a single bridge edge merges or splits two modules."""
    nodes = ["A1", "A2", "A3", "B1", "B2", "B3"]
    modules = [("A1", "A2"), ("A2", "A3"), ("B1", "B2"), ("B2", "B3")]
    assert impl.largest_component(nodes, modules) == ["A1", "A2", "A3"]
    assert math.isclose(impl.largest_component_fraction(nodes, modules), 0.5)
    joined = impl.largest_component(nodes, modules + [("A3", "B1")])
    assert joined == nodes
    assert impl.largest_component_fraction(nodes, modules + [("A3", "B1")]) == 1.0


@pytest.mark.t3
def test_bridge_removal_experiment(impl, artifact):
    """Run the chain-of-modules bridge-removal experiment with your functions."""
    from problems.network_biology.connected_components.experiment import run
    artifact("toy_experiment", run(impl.connected_components, impl.largest_component_fraction))
