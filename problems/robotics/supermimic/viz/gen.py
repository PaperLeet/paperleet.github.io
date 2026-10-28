from problems.robotics.supermimic.experiment import run
from problems.robotics.supermimic.solution import imagined_return, lambda_return_targets


def generate():
    return run(imagined_return, lambda_return_targets)
