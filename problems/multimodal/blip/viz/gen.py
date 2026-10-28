from problems.multimodal.blip.experiment import run
from problems.multimodal.blip.solution import momentum_target_loss


def generate():
    return run(momentum_target_loss)
