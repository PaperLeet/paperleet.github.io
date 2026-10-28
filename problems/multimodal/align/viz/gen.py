from problems.multimodal.align.experiment import run
from problems.multimodal.align.solution import align_loss


def generate():
    return run(align_loss)
