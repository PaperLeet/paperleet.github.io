import itertools
import pytest

FFL_NODES = ["TF1", "TF2", "G1"]
FFL_EDGES = [("TF1", "TF2"), ("TF2", "G1"), ("TF1", "G1")]


def brute_force(nodes, edges, induced=False):
    """Independent oracle: test every ordered triple of distinct nodes."""
    present = {(u, v) for u, v in edges if u != v}
    found = []
    for x, y, z in itertools.permutations(sorted(nodes), 3):
        if (x, y) in present and (y, z) in present and (x, z) in present:
            if induced and any(p in present for p in ((y, x), (z, x), (z, y))):
                continue
            found.append((x, y, z))
    return sorted(found)


def lcg_digraph(seed, size, tries):
    state, nodes, edges = seed, [f"G{i}" for i in range(size)], []
    for _ in range(tries):
        state = (1103515245 * state + 12345) % 2 ** 31
        u = nodes[(state >> 8) % size]
        state = (1103515245 * state + 12345) % 2 ** 31
        edges.append((u, nodes[(state >> 8) % size]))
    return nodes, edges


@pytest.mark.t0
def test_contract(impl):
    """The result is a list of (x, y, z) tuples of node names."""
    result = impl.find_feed_forward_loops(FFL_NODES, FFL_EDGES)
    assert isinstance(result, list) and result == [("TF1", "TF2", "G1")]
    assert all(isinstance(t, tuple) and len(t) == 3 for t in result)
    assert impl.find_feed_forward_loops(FFL_NODES, FFL_EDGES, induced=True) == [("TF1", "TF2", "G1")]
    assert impl.find_feed_forward_loops([], []) == []


@pytest.mark.t1
def test_documented_example_roles(impl):
    """x regulates y and z, y regulates z: the roles are read from edge directions, not names."""
    nodes = ["A", "B", "C", "D"]
    edges = [("C", "A"), ("A", "B"), ("C", "B"), ("B", "D")]
    assert impl.find_feed_forward_loops(nodes, edges) == [("C", "A", "B")]
    assert impl.find_feed_forward_loops(nodes, edges + [("A", "D"), ("C", "D")]) == [
        ("A", "B", "D"), ("C", "A", "B"), ("C", "A", "D"), ("C", "B", "D")]


@pytest.mark.t1
def test_matches_brute_force_triple_oracle(impl):
    """Random directed graphs agree with exhaustive ordered-triple checking, in both modes."""
    for seed, size, tries in [(1, 6, 12), (2, 7, 20), (3, 8, 28), (4, 5, 14), (5, 9, 30)]:
        nodes, edges = lcg_digraph(seed, size, tries)
        for induced in (False, True):
            assert impl.find_feed_forward_loops(nodes, edges, induced=induced) == brute_force(nodes, edges, induced), (seed, induced)


@pytest.mark.t2
def test_negative_examples_are_not_feed_forward_loops(impl):
    """Chains, fan-out, fan-in, cycles and reversed shortcuts are not feed-forward loops."""
    n = ["X", "Y", "Z"]
    negatives = {
        "chain": [("X", "Y"), ("Y", "Z")],
        "fan-out": [("X", "Y"), ("X", "Z")],
        "fan-in": [("X", "Z"), ("Y", "Z")],
        "cycle": [("X", "Y"), ("Y", "Z"), ("Z", "X")],
        "mutual edge only": [("X", "Y"), ("Y", "Z"), ("Z", "Y")],
        "single edge": [("X", "Y")],
        "none": [],
    }
    for name, edges in negatives.items():
        assert impl.find_feed_forward_loops(n, edges) == [], name


@pytest.mark.t2
def test_direction_distinguishes_motif_from_undirected_triangle(impl):
    """Of the 8 orientations of a triangle, 6 contain exactly one loop and the 2 cycles contain none."""
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    with_loop = without = 0
    for flips in itertools.product((False, True), repeat=3):
        edges = [(v, u) if flip else (u, v) for (u, v), flip in zip(pairs, flips)]
        found = impl.find_feed_forward_loops(["A", "B", "C"], edges)
        assert found == brute_force(["A", "B", "C"], edges)
        assert len(found) in (0, 1)
        with_loop += len(found) == 1
        without += len(found) == 0
    assert (with_loop, without) == (6, 2)


@pytest.mark.t2
def test_induced_versus_non_induced_matching(impl):
    """Extra edges among the three nodes are allowed unless induced=True."""
    base = FFL_EDGES
    for extra in (("G1", "TF2"), ("G1", "TF1"), ("TF2", "TF1")):
        edges = base + [extra]
        assert impl.find_feed_forward_loops(FFL_NODES, edges, induced=True) == [], extra
        assert ("TF1", "TF2", "G1") in impl.find_feed_forward_loops(FFL_NODES, edges)
    both = impl.find_feed_forward_loops(FFL_NODES, base + [("G1", "TF2")])
    assert both == [("TF1", "G1", "TF2"), ("TF1", "TF2", "G1")]


@pytest.mark.t2
def test_overlapping_loops_and_noise(impl):
    """Shared edges give separate loops; repeated edges and self-loops change nothing."""
    nodes = ["X", "Y1", "Y2", "Z"]
    edges = [("X", "Y1"), ("X", "Y2"), ("Y1", "Z"), ("Y2", "Z"), ("X", "Z")]
    assert impl.find_feed_forward_loops(nodes, edges) == [("X", "Y1", "Z"), ("X", "Y2", "Z")]
    noisy = edges + edges + [("X", "X"), ("Z", "Z"), ("Y1", "Y1")]
    assert impl.find_feed_forward_loops(nodes, noisy) == [("X", "Y1", "Z"), ("X", "Y2", "Z")]
    assert impl.find_feed_forward_loops(nodes, noisy, induced=True) == [("X", "Y1", "Z"), ("X", "Y2", "Z")]
    assert impl.find_feed_forward_loops(["A", "B"], [("A", "A"), ("A", "B")]) == []


@pytest.mark.t2
def test_validation_and_input_order_independence(impl):
    """Unknown or repeated node names raise ValueError; order of nodes and edges is irrelevant."""
    with pytest.raises(ValueError):
        impl.find_feed_forward_loops(["A", "B"], [("A", "Z")])
    with pytest.raises(ValueError):
        impl.find_feed_forward_loops(["A", "A"], [])
    nodes, edges = lcg_digraph(8, 8, 26)
    base = impl.find_feed_forward_loops(nodes, edges)
    assert impl.find_feed_forward_loops(list(reversed(nodes)), list(reversed(edges))) == base
    assert base == sorted(base)


@pytest.mark.t3
def test_motif_randomization_experiment(impl, artifact):
    """Run the synthetic regulatory network and degree-preserving randomization experiment."""
    from problems.network_biology.feed_forward_loops.experiment import run
    artifact("toy_experiment", run(impl.find_feed_forward_loops))
