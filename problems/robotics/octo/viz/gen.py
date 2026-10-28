from problems.robotics.octo.experiment import run
from problems.robotics.octo.solution import masked_action_loss

def generate():
    return run(masked_action_loss)
