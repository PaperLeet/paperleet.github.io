from problems.ai_infra.paged_attention.experiment import run
from problems.ai_infra.paged_attention.solution import gather_paged_kv, paged_attention


def generate():
    return run(gather_paged_kv, paged_attention)
