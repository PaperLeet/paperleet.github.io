from problems.foundations.top_p_draw.solution import sample_top_p
from problems.foundations.top_p_draw.experiment import run


def generate():
    return {'kind':'toy_experiment','title':'Top-p: Draw Tokens from the Nucleus',
            'problem':'foundations/top_p_draw',**run(sample_top_p)}
