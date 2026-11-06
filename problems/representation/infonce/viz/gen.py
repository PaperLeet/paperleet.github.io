from problems.representation.infonce.solution import info_nce_loss
from problems.representation.infonce.experiment import run


def generate():
    result = run(info_nce_loss)
    return {'kind': 'toy_experiment', 'title': 'InfoNCE positive-candidate CPU lab',
            'problem': 'representation/infonce',
            'provenance': {'kind': 'teaching_adaptation',
                           'source': 'https://arxiv.org/abs/1807.03748',
                           'anchor': 'Section 2.3, Equation (4)',
                           'adaptation': result['provenance']},
            'charts': result['charts'], 'metrics': result['metrics']}
