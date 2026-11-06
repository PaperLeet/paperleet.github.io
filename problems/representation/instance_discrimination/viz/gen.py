from problems.representation.instance_discrimination.solution import instance_nce_loss, update_memory
from problems.representation.instance_discrimination.experiment import run


def generate():
    result = run(instance_nce_loss, update_memory)
    return {'kind': 'toy_experiment', 'title': 'Instance Discrimination: Binary NCE and Memory Bank',
            'problem': 'representation/instance_discrimination',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1805.01978',
                           'anchor': 'Section 3.2: noise-contrastive estimation', 'adaptation': result['provenance']},
            'charts': result['charts'], 'metrics': result['metrics']}
