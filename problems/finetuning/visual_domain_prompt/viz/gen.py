from problems.finetuning.visual_domain_prompt.solution import apply_prompts, homeostatic_factor, homeostatic_penalty
from problems.finetuning.visual_domain_prompt.experiment import run


def generate():
    result = run(apply_prompts, homeostatic_factor, homeostatic_penalty)
    return {'kind': 'toy_experiment', 'title': 'VDP: Visual Prompts and Homeostatic Regularization', 'problem': 'finetuning/visual_domain_prompt',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2212.04145',
                           'anchor': 'Method, Eq. (1): additive domain-specific and domain-agnostic visual prompts', 'adaptation': 'Two-pixel identity readout, supplied fixed teacher features and supplied history; 120 SGD updates at lr=0.1 for strengths 0 and 1. Squared feature consistency replaces classifier cross entropy for an independent analytic optimum. Not CTTA benchmark reproduction.'},
            'charts': result['charts'], 'metrics': result['metrics']}
