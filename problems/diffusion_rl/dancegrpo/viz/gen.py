from problems.diffusion_rl.dancegrpo.experiment import run
from problems.diffusion_rl.dancegrpo.solution import shared_noise, group_advantages, dance_loss

def generate():
    return run(shared_noise, group_advantages, dance_loss)
