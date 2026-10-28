from problems.multimodal.blip_2.experiment import run
from problems.multimodal.blip_2.solution import max_query_similarity


def generate():
    return run(max_query_similarity)
