from problems.foundations.top_p_sampling.solution import top_p_probs
from problems.foundations.top_p_sampling.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": 'Top-p: Cumulative Probability and Nucleus Candidates', "problem": "foundations/top_p_sampling", **run(top_p_probs)}
