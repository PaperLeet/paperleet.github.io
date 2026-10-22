from problems.network_biology.feed_forward_loops.solution import find_feed_forward_loops
from problems.network_biology.feed_forward_loops.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "Feed-forward Loops versus Randomized Networks",
            "problem": "network_biology/feed_forward_loops", **run(find_feed_forward_loops)}
