from problems.foundations.next_token_loss.solution import next_token_loss
from problems.foundations.next_token_loss.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": 'Next-token Loss: Shifted Causal Targets', "problem": "foundations/next_token_loss", **run(next_token_loss)}
