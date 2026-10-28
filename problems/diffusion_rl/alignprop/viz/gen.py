from problems.diffusion_rl.alignprop.experiment import run
from problems.diffusion_rl.alignprop.solution import reward_rollout

def generate():
    return run(reward_rollout)
