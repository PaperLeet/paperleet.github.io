from problems.representation.dino.experiment import run
from problems.representation.dino.solution import teacher_probabilities, dino_loss, update_center


def generate():
    return run(teacher_probabilities, dino_loss, update_center)
