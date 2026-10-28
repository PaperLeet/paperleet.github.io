from problems.ai_infra.gpipe.experiment import run
from problems.ai_infra.gpipe.solution import pipeline_schedule, microbatch_mse

def generate():
    return run(pipeline_schedule, microbatch_mse)
