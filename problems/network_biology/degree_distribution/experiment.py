"""Synthetic degree-distribution experiments. No real interaction data is used."""
import math

TOY_NODES = [f"P{i}" for i in range(1, 9)]
TOY_EDGES = [("P1", "P2"), ("P1", "P3"), ("P1", "P4"), ("P1", "P5"),
             ("P2", "P3"), ("P5", "P6"), ("P6", "P7")]


def growth_network(size=60, seed=7):
    """Deterministic preferential-attachment toy graph (a synthetic teaching graph).

    Each new node links to two distinct earlier nodes chosen with probability
    proportional to degree, using a fixed linear-congruential generator so the
    result does not depend on the Python version.
    """
    state = seed

    def uniform():
        nonlocal state
        state = (1103515245 * state + 12345) % 2 ** 31
        return state / 2 ** 31

    nodes = [f"N{i:02d}" for i in range(size)]
    edges = [(nodes[0], nodes[1])]
    endpoints = [nodes[0], nodes[1]]
    for i in range(2, size):
        targets = []
        while len(targets) < 2:
            pick = endpoints[int(uniform() * len(endpoints))]
            if pick not in targets:
                targets.append(pick)
        for target in targets:
            edges.append((nodes[i], target))
            endpoints += [nodes[i], target]
    return nodes, edges


def _check(degrees, distribution, nodes, edges):
    # Independent adjacency-matrix oracle for the checks inside the experiment.
    index = {name: i for i, name in enumerate(nodes)}
    matrix = [[0] * len(nodes) for _ in nodes]
    for u, v in edges:
        if u != v:
            matrix[index[u]][index[v]] = matrix[index[v]][index[u]] = 1
    expected = {name: sum(matrix[index[name]]) for name in nodes}
    assert degrees == expected, "node_degrees disagrees with the adjacency-matrix row sums"
    assert list(distribution) == list(range(max(expected.values()) + 1))
    assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12)
    return expected


def run(node_degrees, degree_distribution):
    toy = degree_distribution(TOY_NODES, TOY_EDGES)
    _check(node_degrees(TOY_NODES, TOY_EDGES), toy, TOY_NODES, TOY_EDGES)
    # One extra edge between two degree-1 proteins moves both into the degree-2 bin.
    changed_edges = TOY_EDGES + [("P4", "P7")]
    changed = degree_distribution(TOY_NODES, changed_edges)
    _check(node_degrees(TOY_NODES, changed_edges), changed, TOY_NODES, changed_edges)
    width = max(len(toy), len(changed))
    pad = lambda d: [d.get(k, 0.0) for k in range(width)]
    nodes, edges = growth_network()
    growth_degrees = node_degrees(nodes, edges)
    growth = degree_distribution(nodes, edges)
    _check(growth_degrees, growth, nodes, edges)
    ranked = sorted(growth_degrees.items(), key=lambda item: (-item[1], item[0]))
    chart_toy = {"kind": "bar", "title": "Toy network: P(k) before and after adding edge P4-P7",
                 "x": [str(k) for k in range(width)], "x_label": "degree k", "y_label": "P(k)",
                 "series": [{"name": "Original 8-node toy network", "values": pad(toy)},
                            {"name": "After adding P4-P7", "values": pad(changed)}]}
    chart_growth = {"kind": "bar", "title": "Synthetic 60-node growth network: P(k)",
                    "x": [str(k) for k in range(len(growth))], "x_label": "degree k", "y_label": "P(k)",
                    "series": [{"name": "Preferential-attachment toy graph", "values": [growth[k] for k in range(len(growth))]}]}
    return {"provenance": "Synthetic teaching graphs generated in this package; no real interactome, no fitted exponent and no claim that any biological network is scale-free.",
            "charts": [chart_toy, chart_growth],
            "metrics": {"toy_nodes": len(TOY_NODES), "toy_edges": len(TOY_EDGES), "toy_max_degree": len(toy) - 1,
                        "growth_nodes": len(nodes), "growth_edges": len(edges),
                        "growth_mean_degree": 2 * len(edges) / len(nodes),
                        "growth_max_degree": ranked[0][1], "growth_top_node": ranked[0][0]}}
