from problems.network_biology.shortest_paths.solution import distances_from, shortest_path_length, shortest_path
from problems.network_biology.shortest_paths.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "Shortest Paths as Graph Proximity",
            "problem": "network_biology/shortest_paths", **run(distances_from, shortest_path_length, shortest_path)}
