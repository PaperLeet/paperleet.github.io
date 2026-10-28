from problems.ai_infra.zero.experiment import run
from problems.ai_infra.zero.solution import zero2_adam_step, state_bytes

def generate():
    return run(zero2_adam_step, state_bytes)
