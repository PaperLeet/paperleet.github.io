from problems.foundations.sft_loss_mask.solution import sft_loss
from problems.foundations.sft_loss_mask.experiment import run


def generate():
    return {"kind": "toy_experiment", "title": 'SFT Loss Mask: Train on Response Tokens', "problem": "foundations/sft_loss_mask", **run(sft_loss)}
