from problems.representation.ibot.experiment import run
from problems.representation.ibot.solution import teacher_patch_targets, ibot_patch_loss


def generate():
    return run(teacher_patch_targets, ibot_patch_loss)
