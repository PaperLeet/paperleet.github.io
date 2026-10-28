from problems.robotics.dagger.experiment import run
from problems.robotics.dagger.solution import mixture_actions,aggregate_dataset

def generate():
    return run(mixture_actions,aggregate_dataset)
