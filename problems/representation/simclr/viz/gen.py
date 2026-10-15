from problems.representation.simclr.solution import nt_xent_loss
from problems.representation.simclr.experiment import run


def generate():
    result=run(nt_xent_loss)
    return {'kind':'toy_experiment','title':'SimCLR paired-vector CPU lab','problem':'representation/simclr',
            'provenance':{'kind':'teaching_adaptation','source':'https://proceedings.mlr.press/v119/chen20j.html',
                          'anchor':'Section 2.1, Equation 1 and Algorithm 1','adaptation':result['provenance']},
            'charts':result['charts'],'metrics':result['metrics']}
