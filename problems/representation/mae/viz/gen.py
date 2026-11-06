from problems.representation.mae.experiment import run
from problems.representation.mae.solution import masked_reconstruction_loss


def generate():
    return run(masked_reconstruction_loss)
