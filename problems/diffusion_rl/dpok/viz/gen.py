from problems.diffusion_rl.dpok.experiment import run
from problems.diffusion_rl.dpok.solution import gaussian_kl, dpok_surrogate

def generate():
    return run(gaussian_kl, dpok_surrogate)
