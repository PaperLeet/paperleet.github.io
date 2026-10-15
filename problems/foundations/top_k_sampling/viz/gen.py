from problems.foundations.top_k_sampling.solution import top_k_probs
from problems.foundations.top_k_sampling.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": 'Top-k: Temperature and Candidate Probabilities', "problem": "foundations/top_k_sampling", **run(top_k_probs)}
