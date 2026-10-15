from problems.representation.moco.solution import moco_loss, momentum_update, enqueue
from problems.representation.moco.experiment import run


def generate():
    result=run(moco_loss,momentum_update,enqueue)
    return {'kind':'toy_experiment','title':'MoCo momentum and queue CPU lab','problem':'representation/moco',
        'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/1911.05722','anchor':'Equations 1–2 and Algorithm 1','adaptation':result['provenance']},
        'charts':result['charts'],'metrics':result['metrics']}
