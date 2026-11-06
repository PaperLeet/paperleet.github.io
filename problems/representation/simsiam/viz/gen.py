from problems.representation.simsiam.experiment import run
from problems.representation.simsiam.solution import simsiam_loss


def generate():
    return run(simsiam_loss)
