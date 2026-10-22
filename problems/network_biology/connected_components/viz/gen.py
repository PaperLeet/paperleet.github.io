from problems.network_biology.connected_components.solution import connected_components, largest_component_fraction
from problems.network_biology.connected_components.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "Connected Components and the Largest Component",
            "problem": "network_biology/connected_components", **run(connected_components, largest_component_fraction)}
