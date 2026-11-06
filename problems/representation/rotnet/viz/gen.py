from problems.representation.rotnet.solution import rotation_batch, rotation_loss
from problems.representation.rotnet.experiment import run


def generate():
    result = run(rotation_batch, rotation_loss)
    return {'kind': 'toy_experiment', 'title': 'RotNet: Predicting Four Image Rotations',
            'problem': 'representation/rotnet',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1803.07728',
                           'anchor': 'Section 3: image rotation prediction', 'adaptation': result['provenance']},
            'charts': result['charts'], 'metrics': result['metrics']}
