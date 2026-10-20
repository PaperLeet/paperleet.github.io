"""Resolution sweep of greedy modularity communities on a synthetic network. No real data."""
MODULES = [[f"M{k}_{i}" for i in range(4)] for k in range(3)]
NODES = [name for module in MODULES for name in module] + ["Z"]
EDGES = ([(a, b) for module in MODULES for i, a in enumerate(module) for b in module[i + 1:]]
         + [("M0_0", "M1_0"), ("M1_1", "M2_0")])
PLANTED = [sorted(module) for module in MODULES] + [["Z"]]
RESOLUTIONS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]


def run(modularity, greedy_modularity_communities):
    planted_q = modularity(NODES, EDGES, PLANTED)
    assert 0.5 < planted_q < 0.6, planted_q
    counts, scores, sizes = [], [], []
    for gamma in RESOLUTIONS:
        communities = greedy_modularity_communities(NODES, EDGES, gamma)
        assert sorted(n for c in communities for n in c) == sorted(NODES)
        counts.append(float(len(communities)))
        scores.append(modularity(NODES, EDGES, communities, gamma))
        sizes.append("/".join(str(len(c)) for c in communities))
    default = greedy_modularity_communities(NODES, EDGES)
    assert default == [sorted(m) for m in MODULES] + [["Z"]], default
    assert greedy_modularity_communities(NODES, EDGES, 1.0) == greedy_modularity_communities(list(reversed(NODES)), list(reversed(EDGES)), 1.0)
    index = [str(g) for g in RESOLUTIONS]
    return {"provenance": "Synthetic network of three 4-node modules, two bridge edges and one isolated node. Detected communities are candidate structural modules for this graph and resolution value; they are not experimentally validated pathways or complexes.",
            "charts": [{"kind": "line", "title": "Number of detected communities versus resolution", "x": RESOLUTIONS,
                        "x_label": "resolution (gamma)", "y_label": "communities",
                        "series": [{"name": "Greedy modularity communities", "values": counts}]},
                       {"kind": "line", "title": "Modularity of the detected partition at its own resolution", "x": RESOLUTIONS,
                        "x_label": "resolution (gamma)", "y_label": "Q",
                        "series": [{"name": "Q at the same gamma", "values": scores}]}],
            "metrics": {"nodes": len(NODES), "edges": len(EDGES), "planted_modularity": planted_q,
                        "recovers_planted_at_gamma_1": 1.0, "community_sizes_by_resolution": "; ".join(f"{g}: {s}" for g, s in zip(index, sizes))}}
