from problems.representation.dinov2_koleo.solution import koleo_loss
from problems.representation.dinov2_koleo.experiment import run


def generate():
    result = run(koleo_loss)
    return {'kind': 'toy_experiment', 'title': 'DINOv2: KoLeo Feature-Spreading Regularization',
            'problem': 'representation/dinov2_koleo',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2304.07193',
                           'anchor': 'KoLeo regularizer in the training recipe', 'adaptation': result['provenance']},
            'charts': result['charts'], 'metrics': result['metrics']}
