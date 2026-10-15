from problems.retrieval.lightgcn.solution import normalized_bipartite_graph, lightgcn_embeddings
from problems.retrieval.lightgcn.experiment import run


def generate():
    result=run(normalized_bipartite_graph,lightgcn_embeddings)
    return {'kind':'toy_experiment','title':'LightGCN: Two-Hop Collaborative Propagation','problem':'retrieval/lightgcn',
            'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2002.02126',
                          'anchor':'Sections 3.1.1–3.1.3, Equations 3–8; Section 3.3, Equation 15','adaptation':result['provenance']},
            'charts':result['charts'],'metrics':result['metrics']}
