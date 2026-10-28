from problems.diffusion_rl.pc_flow.experiment import run
from problems.diffusion_rl.pc_flow.solution import pcflow_loss

def generate():
    return run(pcflow_loss)
