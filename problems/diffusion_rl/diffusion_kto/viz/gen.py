from problems.diffusion_rl.diffusion_kto.experiment import run
from problems.diffusion_rl.diffusion_kto.solution import kto_loss

def generate():
    return run(kto_loss)
