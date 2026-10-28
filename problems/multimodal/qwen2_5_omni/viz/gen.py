from problems.multimodal.qwen2_5_omni.experiment import run
from problems.multimodal.qwen2_5_omni.solution import time_aligned_interleave


def generate():
    return run(time_aligned_interleave)
