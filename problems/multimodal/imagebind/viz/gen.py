from problems.multimodal.imagebind.experiment import run
from problems.multimodal.imagebind.solution import image_anchor_loss


def generate():
    return run(image_anchor_loss)
