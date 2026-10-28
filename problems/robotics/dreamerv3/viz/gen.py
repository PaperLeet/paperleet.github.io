from problems.robotics.dreamerv3.experiment import run
from problems.robotics.dreamerv3.solution import symlog,symexp,two_hot

def generate():
    return run(symlog,symexp,two_hot)
