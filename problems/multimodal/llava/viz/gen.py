from problems.multimodal.llava.experiment import run
from problems.multimodal.llava.solution import pack_visual_tokens


def generate():
    return run(pack_visual_tokens)
