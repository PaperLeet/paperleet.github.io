from problems.foundations.gradient_accumulation.solution import accumulate_gradients
from problems.foundations.gradient_accumulation.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": 'Gradient Accumulation: Valid-token Weighting', "problem": "foundations/gradient_accumulation", **run(accumulate_gradients)}
