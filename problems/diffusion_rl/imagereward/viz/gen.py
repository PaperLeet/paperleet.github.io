from problems.diffusion_rl.imagereward.experiment import run
from problems.diffusion_rl.imagereward.solution import ranking_loss, refl_clean

def generate():
    return run(ranking_loss, refl_clean)
