from problems.representation.barlow_twins.experiment import run
from problems.representation.barlow_twins.solution import cross_correlation, barlow_twins_loss


def generate():
    return run(cross_correlation, barlow_twins_loss)
