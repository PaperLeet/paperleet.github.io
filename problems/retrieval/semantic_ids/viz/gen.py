from problems.retrieval.semantic_ids.solution import residual_quantize, resolve_collisions
from problems.retrieval.semantic_ids.experiment import run


def generate():
    result = run(residual_quantize, resolve_collisions)
    return {'kind':'toy_experiment','title':'TIGER: Residual Quantization and Semantic IDs','problem':'retrieval/semantic_ids',
            'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2305.05065',
                           'anchor':'Sections 3.1, 3.2, 4 and 4.5','adaptation':'Frozen residual encoding from Section 3.1 and collision suffixes from Section 4; no encoder/decoder/codebook training.'},
            'charts':result['charts'],'metrics':result['metrics']}
