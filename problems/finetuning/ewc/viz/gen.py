from problems.finetuning.ewc.experiment import run
from problems.finetuning.ewc.solution import ewc_penalty


def generate():
    return run(ewc_penalty)
