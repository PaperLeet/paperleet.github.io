from problems.network_biology.community_modules.solution import modularity, greedy_modularity_communities
from problems.network_biology.community_modules.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "Community Detection and Candidate Modules",
            "problem": "network_biology/community_modules", **run(modularity, greedy_modularity_communities)}
