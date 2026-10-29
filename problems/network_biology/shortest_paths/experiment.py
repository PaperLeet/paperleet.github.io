"""Shortest-path distances and one shortcut edge on a synthetic network. No real data."""
NODES = [f"P{i:02d}" for i in range(1, 11)]
EDGES = [("P01", "P02"), ("P01", "P03"), ("P02", "P04"), ("P03", "P04"), ("P04", "P05"),
         ("P05", "P06"), ("P06", "P07"), ("P08", "P09")]
SHORTCUT = ("P01", "P06")
REACHABLE = NODES[:7]


def run(distances_from, shortest_path_length, shortest_path):
    before = distances_from(NODES, EDGES, "P01")
    assert before == {"P01": 0, "P02": 1, "P03": 1, "P04": 2, "P05": 3, "P06": 4, "P07": 5}, before
    assert shortest_path_length(NODES, EDGES, "P01", "P08") is None and shortest_path(NODES, EDGES, "P01", "P10") is None
    path_before = shortest_path(NODES, EDGES, "P01", "P07")
    assert path_before == ["P01", "P02", "P04", "P05", "P06", "P07"], path_before
    edges_after = EDGES + [SHORTCUT]
    after = distances_from(NODES, edges_after, "P01")
    assert after == {"P01": 0, "P02": 1, "P03": 1, "P04": 2, "P05": 2, "P06": 1, "P07": 2}, after
    path_after = shortest_path(NODES, edges_after, "P01", "P07")
    assert path_after == ["P01", "P06", "P07"], path_after
    unreachable = [n for n in NODES if n not in before]
    chart = {"kind": "bar", "title": "Distance from P01 before and after adding edge P01-P06",
             "x": REACHABLE, "x_label": "reachable protein", "y_label": "edges on a shortest path",
             "series": [{"name": "Original", "values": [float(before[n]) for n in REACHABLE]},
                        {"name": "With P01-P06", "values": [float(after[n]) for n in REACHABLE]}]}
    return {"provenance": "Synthetic 10-node teaching graph. Path length here is graph distance only; it is a hypothesis about proximity, not evidence of functional or causal relationship.",
            "charts": [chart],
            "metrics": {"nodes": len(NODES), "reachable_from_P01": len(before), "unreachable_from_P01": len(unreachable),
                        "unreachable_nodes": ", ".join(unreachable), "path_P01_to_P07_before": " > ".join(path_before),
                        "path_P01_to_P07_after": " > ".join(path_after)}}
