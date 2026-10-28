from problems.robotics.diffusion_policy.experiment import run
from problems.robotics.diffusion_policy.solution import add_action_noise, action_denoising_loss, action_chunk


def generate():
    return run(add_action_noise, action_denoising_loss, action_chunk)
