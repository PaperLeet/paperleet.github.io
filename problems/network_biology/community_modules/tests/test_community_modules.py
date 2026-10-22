import math
import pytest

A = ["A1", "A2", "A3"]
B = ["B1", "B2", "B3"]
TRI_NODES = A + B
TRI_EDGES = [("A1", "A2"), ("A1", "A3"), ("A2", "A3"), ("B1", "B2"), ("B1", "B3"), ("B2", "B3"), ("A3", "B1")]
K4A = ["A1", "A2", "A3", "A4"]
K4B = ["B1", "B2", "B3", "B4"]
K4_NODES = K4A + K4B
K4_EDGES = ([(u, v) for i, u in enumerate(K4A) for v in K4A[i + 1:]]
            + [(u, v) for i, u in enumerate(K4B) for v in K4B[i + 1:]] + [("A1", "B1")])


def matrix_modularity(nodes, edges, communities, gamma=1.0):
    """Independent oracle: Q = 1/(2m) * sum_ij (A_ij - gamma k_i k_j / 2m) [c_i = c_j]."""
    index = {n: i for i, n in enumerate(nodes)}
    a = [[0] * len(nodes) for _ in nodes]
    for u, v in edges:
        if u != v:
            a[index[u]][index[v]] = a[index[v]][index[u]] = 1
    k = [sum(row) for row in a]
    two_m = sum(k)
    if two_m == 0:
        return 0.0
    label = {n: i for i, c in enumerate(communities) for n in c}
    return sum((a[i][j] - gamma * k[i] * k[j] / two_m) for i, u in enumerate(nodes) for j, v in enumerate(nodes)
               if label[u] == label[v]) / two_m


def set_partitions(items):
    """All partitions of a list (Bell number many); used only for tiny graphs."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in set_partitions(rest):
        yield [[first]] + partition
        for i in range(len(partition)):
            yield partition[:i] + [[first] + partition[i]] + partition[i + 1:]


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
    """modularity returns a float; greedy returns sorted lists of names."""
    q = impl.modularity(TRI_NODES, TRI_EDGES, [A, B])
    assert isinstance(q, float)
    communities = impl.greedy_modularity_communities(TRI_NODES, TRI_EDGES)
    assert isinstance(communities, list) and all(isinstance(c, list) for c in communities)
    assert communities == [A, B]
    assert impl.greedy_modularity_communities([], []) == []


@pytest.mark.t1
def test_modularity_documented_values_and_matrix_oracle(impl):
    """Q = 5/14 for the two triangles, 0 for one community; random partitions match the matrix formula."""
    assert math.isclose(impl.modularity(TRI_NODES, TRI_EDGES, [A, B]), 5 / 14, abs_tol=1e-12)
    assert math.isclose(impl.modularity(TRI_NODES, TRI_EDGES, [TRI_NODES]), 0.0, abs_tol=1e-12)
    assert math.isclose(impl.modularity(TRI_NODES, TRI_EDGES, [[n] for n in TRI_NODES]), -17 / 98, abs_tol=1e-12)
    for seed, size, tries in [(1, 7, 10), (2, 9, 16), (3, 6, 9), (4, 10, 22)]:
        nodes, edges = lcg_graph(seed, size, tries)
        for cut in (1, 2, 3):
            partition = [nodes[i::cut] for i in range(cut)]
            for gamma in (0.5, 1.0, 2.0):
                assert math.isclose(impl.modularity(nodes, edges, partition, gamma),
                                    matrix_modularity(nodes, edges, partition, gamma), abs_tol=1e-9), (seed, cut, gamma)


@pytest.mark.t1
def test_greedy_recovers_planted_modules(impl):
    """Well-separated modules are recovered and agree with the exhaustive optimum on the 8-node graph."""
    assert impl.greedy_modularity_communities(K4_NODES, K4_EDGES) == [K4A, K4B]
    best = max(set_partitions(K4_NODES), key=lambda p: matrix_modularity(K4_NODES, K4_EDGES, p))
    assert sorted(sorted(c) for c in best) == [K4A, K4B]
    modules = [[f"M{k}_{i}" for i in range(4)] for k in range(3)]
    nodes = [n for m in modules for n in m] + ["Z"]
    edges = [(u, v) for m in modules for i, u in enumerate(m) for v in m[i + 1:]] + [("M0_0", "M1_0"), ("M1_1", "M2_0")]
    assert impl.greedy_modularity_communities(nodes, edges) == modules + [["Z"]]


@pytest.mark.t2
def test_partition_and_graph_validation(impl):
    """Communities must partition the nodes exactly; graphs are validated as in network_biology/degree_distribution."""
    bad_partitions = [[A], [A, B[:2]], [A + ["A1"], B], [A, B, []], [A, B + ["Q"]], [A + B, ["A1"]]]
    for partition in bad_partitions:
        with pytest.raises(ValueError):
            impl.modularity(TRI_NODES, TRI_EDGES, partition)
    for bad in (lambda: impl.modularity(["A", "B"], [("A", "Z")], [["A", "B"]]),
                lambda: impl.greedy_modularity_communities(["A", "A"], []),
                lambda: impl.greedy_modularity_communities(["A", "B"], [("A", "Z")])):
        with pytest.raises(ValueError):
            bad()


@pytest.mark.t2
def test_edgeless_isolated_and_noisy_graphs(impl):
    """No edges gives Q = 0 and singletons; isolated nodes stay alone; noise edges are ignored."""
    assert impl.modularity(["A", "B"], [], [["A"], ["B"]]) == 0.0
    assert impl.greedy_modularity_communities(["C", "A", "B"], []) == [["A"], ["B"], ["C"]]
    assert impl.greedy_modularity_communities(["A"], [("A", "A")]) == [["A"]]
    noisy = TRI_EDGES + TRI_EDGES + [("A1", "A1"), ("B2", "B2"), ("B1", "A3")]
    assert impl.greedy_modularity_communities(TRI_NODES + ["Z"], noisy) == [A, B, ["Z"]]
    assert math.isclose(impl.modularity(TRI_NODES, noisy, [A, B]), 5 / 14, abs_tol=1e-12)


@pytest.mark.t2
def test_tie_rule_and_zero_gain_stop(impl):
    """Equal gains go to the smallest name pair; a merge with exactly zero gain is not made."""
    path5 = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
    assert impl.greedy_modularity_communities(path5, edges) == [["a", "b", "c"], ["d", "e"]]
    assert impl.greedy_modularity_communities(list(reversed(path5)), list(reversed(edges))) == [["a", "b", "c"], ["d", "e"]]
    cycle4 = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
    assert impl.greedy_modularity_communities(["a", "b", "c", "d"], cycle4) == [["a", "b"], ["c", "d"]]


@pytest.mark.t2
def test_result_is_a_connected_partition_that_improves_on_singletons(impl):
    """The output partitions the nodes, each community is connected, and Q beats singletons."""
    nodes, edges = lcg_graph(7, 14, 24)
    communities = impl.greedy_modularity_communities(nodes, edges)
    flat = [n for c in communities for n in c]
    assert sorted(flat) == sorted(nodes) and len(flat) == len(set(flat))
    sizes = [len(c) for c in communities]
    assert sizes == sorted(sizes, reverse=True) and all(c == sorted(c) for c in communities)
    adj = {n: set() for n in nodes}
    for u, v in edges:
        if u != v:
            adj[u].add(v); adj[v].add(u)
    for c in communities:
        seen, stack = {c[0]}, [c[0]]
        while stack:
            for nb in adj[stack.pop()] & set(c):
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        assert seen == set(c)
    q = impl.modularity(nodes, edges, communities)
    assert q >= impl.modularity(nodes, edges, [[n] for n in nodes]) - 1e-12
    assert math.isclose(q, matrix_modularity(nodes, edges, communities), abs_tol=1e-9)


@pytest.mark.t2
def test_resolution_changes_the_communities(impl):
    """Small resolution merges everything, large resolution fragments; Q decreases with resolution."""
    low = impl.greedy_modularity_communities(K4_NODES, K4_EDGES, 0.1)
    mid = impl.greedy_modularity_communities(K4_NODES, K4_EDGES, 1.0)
    high = impl.greedy_modularity_communities(K4_NODES, K4_EDGES, 4.0)
    assert low == [sorted(K4_NODES)] and mid == [K4A, K4B] and high == [[n] for n in sorted(K4_NODES)]
    values = [impl.modularity(K4_NODES, K4_EDGES, [K4A, K4B], g) for g in (0.5, 1.0, 2.0)]
    assert values[0] > values[1] > values[2]
    assert math.isclose(impl.modularity(K4_NODES, K4_EDGES, [K4A, K4B], 2.0), matrix_modularity(K4_NODES, K4_EDGES, [K4A, K4B], 2.0), abs_tol=1e-9)


@pytest.mark.t2
def test_deterministic_and_input_order_independent(impl):
    """Repeated runs and reordered node/edge lists give identical communities."""
    nodes, edges = lcg_graph(12, 12, 20)
    first = impl.greedy_modularity_communities(nodes, edges)
    assert impl.greedy_modularity_communities(nodes, edges) == first
    assert impl.greedy_modularity_communities(list(reversed(nodes)), [(v, u) for u, v in reversed(edges)]) == first


@pytest.mark.t3
def test_community_resolution_experiment(impl, artifact):
    """Run the three-module synthetic network and resolution sweep with your functions."""
    from problems.network_biology.community_modules.experiment import run
    artifact("toy_experiment", run(impl.modularity, impl.greedy_modularity_communities))
