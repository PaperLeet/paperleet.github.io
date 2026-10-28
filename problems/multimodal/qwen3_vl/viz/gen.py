from problems.multimodal.qwen3_vl.experiment import run
from problems.multimodal.qwen3_vl.solution import interleaved_mrope


def generate():
    return run(interleaved_mrope)
