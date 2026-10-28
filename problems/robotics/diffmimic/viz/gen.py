from problems.robotics.diffmimic.experiment import run
from problems.robotics.diffmimic.solution import replay_state,pd_torque,state_matching_loss

def generate():
    return run(replay_state,pd_torque,state_matching_loss)
