from problems.retrieval.late_interaction.solution import maxsim_scores, pairwise_loss
from problems.retrieval.late_interaction.experiment import run


def generate():
    result = run(maxsim_scores, pairwise_loss)
    return {'kind':'toy_experiment','title':'ColBERT: Token-Level MaxSim Retrieval','problem':'retrieval/late_interaction',
            'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/2004.12832',
                          'anchor':'Sections 3.2 and 3.3, equations 1–3',
                          'adaptation':'Supplied token vectors, cosine MaxSim and pairwise softmax; optimize four synthetic query vectors, not a BERT encoder or search index.'},
            'charts':result['charts'],'metrics':result['metrics']}
