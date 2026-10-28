from problems.multimodal.siglip.experiment import run
from problems.multimodal.siglip.solution import pairwise_logits, siglip_loss


def generate():
    return run(pairwise_logits, siglip_loss)
