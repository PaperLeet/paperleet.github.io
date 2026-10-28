from problems.ai_infra.distserve.experiment import run
from problems.ai_infra.distserve.solution import disaggregated_trace, slo_goodput

def generate():
    return run(disaggregated_trace, slo_goodput)
