from problems.network_biology.hubs_centrality.solution import degree_centrality, betweenness_centrality, rank_hubs
from problems.network_biology.hubs_centrality.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "Hubs and Centrality: Degree versus Betweenness",
            "problem": "network_biology/hubs_centrality", **run(degree_centrality, betweenness_centrality, rank_hubs)}
