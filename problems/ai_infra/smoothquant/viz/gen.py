from problems.ai_infra.smoothquant.experiment import run
from problems.ai_infra.smoothquant.solution import smooth_scales, smooth_transform

def generate():
    return run(smooth_scales, smooth_transform)
