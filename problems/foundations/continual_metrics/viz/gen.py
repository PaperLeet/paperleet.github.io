from problems.foundations.continual_metrics.solution import continual_metrics
from problems.foundations.continual_metrics.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": "Continual learning evaluation",
            "problem": "foundations/continual_metrics", **run(continual_metrics)}
