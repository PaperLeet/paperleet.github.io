from problems.multimodal.instructblip.experiment import run
from problems.multimodal.instructblip.solution import instruction_query_block


def generate():
    return run(instruction_query_block)
