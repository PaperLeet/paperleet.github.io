from problems.ai_infra.orca.experiment import run
from problems.ai_infra.orca.solution import iteration_schedule

def generate():
    return run(iteration_schedule)
