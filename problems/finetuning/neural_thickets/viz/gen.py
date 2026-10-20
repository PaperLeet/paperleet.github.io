from problems.finetuning.neural_thickets.solution import perturb_candidates, select_experts, majority_vote
from problems.finetuning.neural_thickets.experiment import run


def generate():
    result = run(perturb_candidates, select_experts, majority_vote)
    return {'kind': 'toy_experiment', 'title': 'Neural Thickets: Random Experts and Majority Voting', 'problem': 'finetuning/neural_thickets',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2603.12228',
                           'anchor': 'Section 4, Eqs. (3)-(5): weight perturbations, top-K expert selection and prediction ensembling', 'adaptation': '120 random two-parameter linear classifiers, scales 0.25/0.5/1, 41 selection samples, 9 selected experts and 80 distinct test samples; CPU float64, seed 42. A synthetic search demonstration, not proof of neural thicket density or an LLM benchmark.'},
            'charts': result['charts'], 'metrics': result['metrics']}
