from problems.diffusion_rl.mixgrpo.experiment import run
from problems.diffusion_rl.mixgrpo.solution import window_start, mixed_transition, window_loss

def generate():
    return run(window_start, mixed_transition, window_loss)
