from problems.multimodal.flamingo.experiment import run
from problems.multimodal.flamingo.solution import gated_media_attention


def generate():
    return run(gated_media_attention)
