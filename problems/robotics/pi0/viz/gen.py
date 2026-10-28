from problems.robotics.pi0.experiment import run
from problems.robotics.pi0.solution import flow_interpolate,euler_step

def generate():
    return run(flow_interpolate,euler_step)
