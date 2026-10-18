from problems.foundations.multiturn_sft_mask.solution import assistant_target_mask
from problems.foundations.multiturn_sft_mask.experiment import run


def generate():
    return {'kind': 'toy_experiment', 'title': 'Multi-turn SFT: Assistant Targets',
            'problem': 'foundations/multiturn_sft_mask', **run(assistant_target_mask)}
