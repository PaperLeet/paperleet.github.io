from problems.architecture.rms_norm.solution import rms_norm, partial_rms_norm
from problems.architecture.rms_norm.experiment import run


def generate():
    result = run(rms_norm, partial_rms_norm)
    return {'kind': 'toy_experiment', 'title': 'RMSNorm and prefix-statistic bias',
            'problem': 'architecture/rms_norm',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1910.07467',
                           'anchor': 'Equation (4), Section 5',
                           'adaptation': 'Two synthetic eight-feature rows and an offset sweep; no training or runtime benchmark.'},
            'charts': result['charts'], 'metrics': result['metrics']}
