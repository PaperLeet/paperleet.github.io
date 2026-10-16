from problems.architecture.group_norm.solution import group_norm
from problems.architecture.group_norm.experiment import run


def generate():
    result = run(group_norm)
    return {'kind': 'toy_experiment', 'title': 'GroupNorm: group choice and batch independence',
            'problem': 'architecture/group_norm',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/1803.08494v3#S3',
                           'anchor': 'Equations (1), (2), (6), (7); Figure 3',
                           'adaptation': 'Controlled synthetic companion samples and group choices; no ImageNet/COCO training or performance benchmark.'},
            'charts': result['charts'], 'metrics': result['metrics']}
