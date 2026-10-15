from problems.retrieval.sasrec.solution import causal_item_mask, sampled_next_item_loss
from problems.retrieval.sasrec.experiment import run


def generate():
    result = run(causal_item_mask, sampled_next_item_loss)
    return {'kind':'toy_experiment','title':'SASRec: Causal Next-Item Training','problem':'retrieval/sasrec',
            'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/1808.09781v1','anchor':'Sections III-B and III-E',
                          'adaptation':result['provenance']},'charts':result['charts'],'metrics':result['metrics']}
