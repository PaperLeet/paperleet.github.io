from problems.diffusion_rl.ddpo.experiment import run
from problems.diffusion_rl.ddpo.solution import gaussian_log_prob, ddpo_loss

def generate():
    return run(gaussian_log_prob, ddpo_loss)
