from problems.multimodal.siglip_2.experiment import run
from problems.multimodal.siglip_2.solution import masked_patch_distillation


def generate():
    return run(masked_patch_distillation)
