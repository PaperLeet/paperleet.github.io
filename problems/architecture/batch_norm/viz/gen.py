from problems.architecture.batch_norm.solution import batch_norm
from problems.architecture.batch_norm.experiment import run


def generate():
    result = run(batch_norm)
    return {'kind': 'toy_experiment', 'title': 'BatchNorm: Batch Moments, Moving Estimates and Inference',
            'problem': 'architecture/batch_norm',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/1502.03167v3#S3',
                           'anchor': 'Section 3, Algorithm 1: differentiable minibatch population moments; Section 3.1, Algorithm 2: unbiased population variance estimates and frozen inference; Section 3.2: per-channel reduction over batch and spatial positions',
                           'adaptation': 'Eighty minibatches from synthetic N(3,4) features fit scale/shift with Adam and update detached estimates. Frozen inference is tested with changed companion samples. No network acceleration, causal explanation of BN efficacy, or ImageNet result is claimed.'},
            'charts': result['charts'], 'metrics': result['metrics']}
