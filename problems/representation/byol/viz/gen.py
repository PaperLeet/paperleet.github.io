from problems.representation.byol.experiment import run
from problems.representation.byol.solution import byol_loss, ema_update


def generate():
    return run(byol_loss, ema_update)
