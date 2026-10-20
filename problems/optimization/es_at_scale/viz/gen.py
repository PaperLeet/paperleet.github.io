from problems.optimization.es_at_scale.solution import es_step
from problems.optimization.es_at_scale.experiment import run


def generate():
    result = run(es_step)
    return {'kind': 'toy_experiment', 'title': 'ES at Scale: Reward-Weighted Parameter Updates', 'problem': 'optimization/es_at_scale',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2509.24372',
                           'anchor': 'Section 3, Algorithm 1: z-score rewards and the mean reward-weighted parameter perturbation', 'adaptation': '100 reward-only updates on a two-dimensional squared-distance objective, population 30, sigma=0.1 and alpha=0.05; CPU float64, seed 42. Not an LLM benchmark.'},
            'charts': result['charts'], 'metrics': result['metrics']}
