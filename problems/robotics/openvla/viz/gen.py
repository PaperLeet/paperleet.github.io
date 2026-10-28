from problems.robotics.openvla.experiment import run
from problems.robotics.openvla.solution import encode_actions,decode_actions,unnormalize_actions

def generate():
    return run(encode_actions,decode_actions,unnormalize_actions)
