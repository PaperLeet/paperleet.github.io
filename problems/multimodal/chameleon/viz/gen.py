from problems.multimodal.chameleon.experiment import run
from problems.multimodal.chameleon.solution import qk_layer_norm


def generate():
    return run(qk_layer_norm)
