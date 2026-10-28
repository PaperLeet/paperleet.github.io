from problems.ai_infra.sglang.experiment import run
from problems.ai_infra.sglang.solution import radix_cache_step

def generate():
    return run(radix_cache_step)
