from problems.robotics.td_mpc2.experiment import run
from problems.robotics.td_mpc2.solution import simnorm,elite_moments

def generate():
    return run(simnorm,elite_moments)
