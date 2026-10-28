from problems.multimodal.qwen2_vl.experiment import run
from problems.multimodal.qwen2_vl.solution import chunked_mrope


def generate():
    return run(chunked_mrope)
