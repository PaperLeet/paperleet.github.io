from problems.network_biology.degree_distribution.solution import node_degrees, degree_distribution
from problems.network_biology.degree_distribution.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "Degree Distribution of a Biological Interaction Network",
            "problem": "network_biology/degree_distribution", **run(node_degrees, degree_distribution)}
