from problems.diffusion_rl.d3po.experiment import run
from problems.diffusion_rl.d3po.solution import d3po_loss

def generate():
    return run(d3po_loss)
