from problems.diffusion_rl.diffusion_dpo.experiment import run
from problems.diffusion_rl.diffusion_dpo.solution import paired_noising, diffusion_dpo_loss


def generate():
    return run(paired_noising, diffusion_dpo_loss)
