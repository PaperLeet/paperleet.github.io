from problems.ai_infra.gptq.experiment import run
from problems.ai_infra.gptq.solution import gptq_quantize

def generate():
    return run(gptq_quantize)
