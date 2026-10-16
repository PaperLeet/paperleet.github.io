from problems.representation.graphsage_mean.solution import graphsage_encode
from problems.representation.graphsage_mean.experiment import run


def generate():
    result=run(graphsage_encode)
    return {'kind':'toy_experiment','title':'GraphSAGE: feature aggregation on an unseen graph','problem':'representation/graphsage_mean',
            'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/abs/1706.02216','anchor':'Algorithm1 and Section3.3 mean aggregator','adaptation':result['provenance']},'charts':result['charts'],'metrics':result['metrics']}
