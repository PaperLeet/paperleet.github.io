from problems.diffusion_rl.flow_dppo.experiment import run
from problems.diffusion_rl.flow_dppo.solution import policy_kl, flow_dppo_loss

def generate():
    return run(policy_kl, flow_dppo_loss)
