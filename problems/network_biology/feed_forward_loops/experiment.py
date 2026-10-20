"""Feed-forward loop counts versus degree-preserving randomization on a synthetic network."""
import math

TFS = ["TF1", "TF2", "TF3", "TF4"]
GENES = [f"G{i}" for i in range(1, 9)]
NODES = TFS + GENES
EDGES = [("TF1", "TF2"), ("TF1", "G1"), ("TF2", "G1"), ("TF1", "G2"), ("TF2", "G2"),
         ("TF3", "TF4"), ("TF3", "G3"), ("TF4", "G3"), ("TF2", "G4"), ("TF4", "G5"),
         ("TF3", "G6"), ("TF1", "G7"), ("TF2", "G8")]
EXPECTED_OBSERVED = [("TF1", "TF2", "G1"), ("TF1", "TF2", "G2"), ("TF3", "TF4", "G3")]
ENSEMBLE, SWAPS_PER_EDGE, SEED = 200, 10, 2024


def swapped(edges, seed):
    """Degree-preserving randomization: swap the targets of two edges, never creating loops or duplicates."""
    state = seed

    def draw(limit):
        nonlocal state
        state = (1103515245 * state + 12345) % 2 ** 31
        return (state >> 8) % limit

    current = list(edges)
    present = set(current)
    for _ in range(SWAPS_PER_EDGE * len(current)):
        i, j = draw(len(current)), draw(len(current))
        (a, b), (c, d) = current[i], current[j]
        if i == j or a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        present -= {(a, b), (c, d)}
        present |= {(a, d), (c, b)}
        current[i], current[j] = (a, d), (c, b)
    return current


def run(find_feed_forward_loops):
    observed = find_feed_forward_loops(NODES, EDGES)
    assert observed == EXPECTED_OBSERVED, observed
    assert find_feed_forward_loops(NODES, EDGES, induced=True) == EXPECTED_OBSERVED
    cyclic = find_feed_forward_loops(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")])
    assert cyclic == [], "a directed 3-cycle is not a feed-forward loop"
    counts = []
    for k in range(ENSEMBLE):
        randomized = swapped(EDGES, SEED + 7919 * k)
        assert sorted(a for a, _ in randomized) == sorted(a for a, _ in EDGES)
        assert sorted(b for _, b in randomized) == sorted(b for _, b in EDGES)
        counts.append(len(find_feed_forward_loops(NODES, randomized)))
    mean = sum(counts) / len(counts)
    std = math.sqrt(sum((c - mean) ** 2 for c in counts) / (len(counts) - 1))
    top = max(max(counts), len(observed))
    histogram = [counts.count(v) / len(counts) for v in range(top + 1)]
    marker = [1.0 if v == len(observed) else 0.0 for v in range(top + 1)]
    chart = {"kind": "bar", "title": "FFL count: degree-preserving randomizations versus the observed network",
             "x": [str(v) for v in range(top + 1)], "x_label": "number of feed-forward loops", "y_label": "fraction of networks",
             "series": [{"name": f"{ENSEMBLE} randomized networks", "values": histogram},
                        {"name": "Observed synthetic network", "values": marker}]}
    return {"provenance": "Synthetic 12-node regulatory teaching network with three planted loops; the randomization compares counts only and is not a significance claim, an experimentally validated circuit, or a statement about function.",
            "charts": [chart],
            "metrics": {"observed_ffl": len(observed), "randomized_mean": mean, "randomized_std": std,
                        "ensemble_size": ENSEMBLE, "randomized_max": max(counts),
                        "z_score_descriptive": (len(observed) - mean) / std if std else 0.0}}
