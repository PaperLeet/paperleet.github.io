from problems.architecture.rope.solution import apply_rope
from problems.architecture.rope.experiment import run


def generate():
    result = run(apply_rope)
    return {'kind': 'toy_experiment', 'title': 'RoPE relative logits and cache offsets',
            'problem': 'architecture/rope',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2104.09864',
                           'anchor': 'Sections 3.2, 3.4.2; Equations (13)–(16), (34)',
                           'adaptation': 'Synthetic six-feature vectors and a nine-token KV cache; no pretrained model, training or speed claim.'},
            'charts': result['charts'], 'metrics': result['metrics']}
