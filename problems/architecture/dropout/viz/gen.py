from problems.architecture.dropout.solution import inverted_dropout
from problems.architecture.dropout.experiment import run


def generate():
    result = run(inverted_dropout)
    return {'kind': 'toy_experiment', 'title': 'Inverted Dropout: Explicit Masks and Expected Activations', 'problem': 'architecture/dropout',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf',
                           'anchor': 'Section 4; Section 10, page 1951: inverted Bernoulli scaling and its mean/variance', 'adaptation': 'The harness supplies 64×4 Gaussian features, a fixed linear target and trainable linear weights. Your dropout transforms training inputs at p=0.25 for 100 Adam updates (lr=0.06, seed=193, float64); the chart evaluates MSE without dropout. Another 8192 realized masks measure mean/variance errors against the analytic moments. Acceptance requires MSE<20% of initial, maximum mean error<0.06 and variance error<0.08. This demonstrates behavior, without claiming dropout always improves generalization.'},
            'charts': result['charts'], 'metrics': result['metrics']}
