from problems.architecture.residual_block.solution import residual_block
from problems.architecture.residual_block.experiment import run


def generate():
    result = run(residual_block)
    return {'kind': 'toy_experiment', 'title': 'ResNet: Residual Branches and Projection Shortcuts',
            'problem': 'architecture/residual_block',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/1512.03385v1#S3.SS2',
                           'anchor': 'Section 3.2, Equations (1) and (2), Figure 2: two-layer residual mapping and post-addition ReLU',
                           'adaptation': 'Four two-layer matrix residual blocks learn a positive-vector nonlinear regression task in 80 Adam updates. No CIFAR/ImageNet, convolutional ResNet, normalization or benchmark reproduction.'},
            'charts': result['charts'], 'metrics': result['metrics']}
