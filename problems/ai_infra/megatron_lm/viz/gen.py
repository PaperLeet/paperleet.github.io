from problems.ai_infra.megatron_lm.experiment import run
from problems.ai_infra.megatron_lm.solution import parallel_mlp

def generate():
    return run(parallel_mlp)
