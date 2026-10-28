from problems.diffusion_rl.flow_grpo.experiment import run
from problems.diffusion_rl.flow_grpo.solution import sde_transition, flow_grpo_loss


def generate():
    return run(sde_transition, flow_grpo_loss)
