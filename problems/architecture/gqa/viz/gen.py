from problems.architecture.gqa.solution import mean_pool_kv_heads, grouped_query_attention
from problems.architecture.gqa.experiment import run


def generate():
    result = run(mean_pool_kv_heads, grouped_query_attention)
    return {'kind': 'toy_experiment', 'title': 'GQA: KV sharing and checkpoint conversion',
            'problem': 'architecture/gqa',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://aclanthology.org/2023.emnlp-main.298/',
                           'anchor': 'Sections 2.1–2.2, Figures 1–2',
                           'adaptation': 'Eight synthetic projection heads; persistent KV counts and conversion error before uptraining, not hardware or model-quality measurements.'},
            'charts': result['charts'], 'metrics': result['metrics']}
