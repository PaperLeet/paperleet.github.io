"""Degree versus betweenness on a synthetic two-module network. No real data."""
import math

MODULE_A = ["A1", "A2", "A3", "A4"]
MODULE_B = ["B1", "B2", "B3", "B4"]
NODES = MODULE_A + MODULE_B + ["X"]
EDGES = ([(u, v) for i, u in enumerate(MODULE_A) for v in MODULE_A[i + 1:]]
         + [(u, v) for i, u in enumerate(MODULE_B) for v in MODULE_B[i + 1:]]
         + [("A1", "X"), ("X", "B1")])
SHORTCUT = ("A4", "B4")


def run(degree_centrality, betweenness_centrality, rank_hubs):
    degree = degree_centrality(NODES, EDGES)
    between = betweenness_centrality(NODES, EDGES)
    # Counting shortest paths by hand: all 16 A-B pairs pass X; A1 and B1 each lie on 15 pairs.
    assert math.isclose(between["X"], 16 / 28, abs_tol=1e-9), "betweenness of X should be 16/28"
    assert math.isclose(between["A1"], 15 / 28, abs_tol=1e-9) and math.isclose(between["B1"], 15 / 28, abs_tol=1e-9)
    assert math.isclose(degree["A1"], 4 / 8, abs_tol=1e-12) and math.isclose(degree["X"], 2 / 8, abs_tol=1e-12)
    top_degree, top_between = rank_hubs(degree, 3), rank_hubs(between, 3)
    assert top_degree == ["A1", "B1", "A2"] and top_between == ["X", "A1", "B1"]
    shortcut_edges = EDGES + [SHORTCUT]
    after = betweenness_centrality(NODES, shortcut_edges)
    assert after["X"] < between["X"], "a shortcut between the modules must reduce X's betweenness"
    chart_compare = {"kind": "bar", "title": "Same synthetic network, two centralities", "x": NODES,
                     "x_label": "node", "y_label": "normalized centrality",
                     "series": [{"name": "Degree centrality", "values": [degree[n] for n in NODES]},
                                {"name": "Betweenness centrality", "values": [between[n] for n in NODES]}]}
    chart_shortcut = {"kind": "bar", "title": "Betweenness before and after adding edge A4-B4", "x": NODES,
                      "x_label": "node", "y_label": "betweenness",
                      "series": [{"name": "Original", "values": [between[n] for n in NODES]},
                                 {"name": "With A4-B4", "values": [after[n] for n in NODES]}]}
    return {"provenance": "Synthetic two-module teaching graph; centrality rankings here say nothing about biological essentiality or any real network.",
            "charts": [chart_compare, chart_shortcut],
            "metrics": {"nodes": len(NODES), "edges": len(EDGES), "top3_by_degree": ", ".join(top_degree),
                        "top3_by_betweenness": ", ".join(top_between), "x_betweenness": between["X"],
                        "x_betweenness_with_shortcut": after["X"]}}
