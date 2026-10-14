from problems.retrieval.generative_retrieval.solution import beam_search, lookup_candidates
from problems.retrieval.generative_retrieval.experiment import run


def generate():
    result = run(beam_search, lookup_candidates)
    return {'kind':'toy_experiment','title':'TIGER: Generate IDs and Retrieve Valid Items','problem':'retrieval/generative_retrieval',
            'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2305.05065',
                           'anchor':'Sections 3.1, 3.2, 4 and 4.5','adaptation':'Sections 3.2 and 4.5: unconstrained full-ID beam decoding followed by exact catalog filtering; fixed logits, no seq2seq training or prefix matching.'},
            'charts':result['charts'],'metrics':result['metrics']}
