from problems.foundations.gradient_clipping.solution import clip_gradients
from problems.foundations.gradient_clipping.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": 'Global Gradient Clipping: One Norm for All Parameters', "problem": "foundations/gradient_clipping", **run(clip_gradients)}
