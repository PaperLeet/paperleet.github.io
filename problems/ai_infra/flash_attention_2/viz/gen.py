from problems.ai_infra.flash_attention_2.experiment import run
from problems.ai_infra.flash_attention_2.solution import flash2_forward

def generate():
    return run(flash2_forward)
