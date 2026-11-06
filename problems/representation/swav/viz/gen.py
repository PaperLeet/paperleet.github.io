from problems.representation.swav.experiment import run
from problems.representation.swav.solution import sinkhorn_assignments, swapped_prediction_loss


def generate():
    return run(sinkhorn_assignments, swapped_prediction_loss)
