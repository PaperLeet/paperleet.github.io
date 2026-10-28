from problems.ai_infra.awq.experiment import run
from problems.ai_infra.awq.solution import group_quantize, awq_search

def generate():
    return run(group_quantize, awq_search)
