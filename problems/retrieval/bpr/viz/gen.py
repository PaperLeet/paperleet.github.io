from problems.retrieval.bpr.solution import eligible_negatives, bpr_mf_loss
from problems.retrieval.bpr.experiment import run


def generate():
    result=run(eligible_negatives,bpr_mf_loss)
    return {'kind':'toy_experiment','title':'BPR-MF: Personalized Pairwise Training','problem':'retrieval/bpr',
            'provenance':{'kind':'teaching_adaptation','source':'https://proceedings.mlr.press/r7/rendle09a.html',
                          'anchor':'Sections 3.2, 4.1 and 4.3.1','adaptation':result['provenance']},
            'charts':result['charts'],'metrics':result['metrics']}
