"""Removing bridge edges from a synthetic chain of modules. No real data."""
import math

MODULES, SIZE = 6, 5
NODES = [f"M{m}_{i}" for m in range(MODULES) for i in range(SIZE)]
RING_EDGES = [(f"M{m}_{i}", f"M{m}_{(i + 1) % SIZE}") for m in range(MODULES) for i in range(SIZE)]
BRIDGES = [(f"M{m}_0", f"M{m + 1}_0") for m in range(MODULES - 1)]
REMOVAL_ORDER = [2, 0, 4, 1, 3]   # indices into BRIDGES, removed one by one


def expected_after(removed):
    """Independent arithmetic: runs of modules joined by surviving bridges."""
    cut = set(REMOVAL_ORDER[:removed])
    runs, current = [], 1
    for bridge in range(MODULES - 1):
        if bridge in cut:
            runs.append(current)
            current = 1
        else:
            current += 1
    runs.append(current)
    return sorted((r * SIZE for r in runs), reverse=True)


def run(connected_components, largest_component_fraction):
    toy_nodes = [f"P{i}" for i in range(1, 10)]
    toy_edges = [("P1", "P2"), ("P2", "P3"), ("P3", "P1"), ("P3", "P4"), ("P5", "P6"), ("P6", "P7")]
    toy = connected_components(toy_nodes, toy_edges)
    assert toy == [["P1", "P2", "P3", "P4"], ["P5", "P6", "P7"], ["P8"], ["P9"]], toy
    steps, fractions, counts = list(range(len(BRIDGES) + 1)), [], []
    for removed in steps:
        gone = {BRIDGES[i] for i in REMOVAL_ORDER[:removed]}
        edges = RING_EDGES + [b for b in BRIDGES if b not in gone]
        sizes = [len(c) for c in connected_components(NODES, edges)]
        assert sizes == expected_after(removed), (removed, sizes)
        fraction = largest_component_fraction(NODES, edges)
        assert math.isclose(fraction, sizes[0] / len(NODES), abs_tol=1e-12)
        fractions.append(fraction)
        counts.append(float(len(sizes)))
    return {"provenance": "Synthetic chain of six 5-node modules; the numbers show graph connectivity only and say nothing about whether separated components are biologically independent.",
            "charts": [{"kind": "line", "title": "Largest component after removing bridge edges", "x": steps,
                        "x_label": "bridge edges removed", "y_label": "largest component / all nodes",
                        "series": [{"name": "Largest component fraction", "values": fractions}]},
                       {"kind": "line", "title": "Number of components after removing bridge edges", "x": steps,
                        "x_label": "bridge edges removed", "y_label": "components",
                        "series": [{"name": "Connected components", "values": counts}]}],
            "metrics": {"nodes": len(NODES), "ring_edges": len(RING_EDGES), "bridges": len(BRIDGES),
                        "toy_component_sizes": ", ".join(str(len(c)) for c in toy),
                        "final_largest_fraction": fractions[-1]}}
