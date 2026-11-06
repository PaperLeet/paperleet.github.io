from problems.representation.beit.experiment import run
from problems.representation.beit.solution import masked_token_cross_entropy


def generate():
    return run(masked_token_cross_entropy)
