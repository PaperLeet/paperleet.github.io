"""Scalar log-sum-exp oracle and original deterministic paired-vector fixture."""
import math
import torch


def scalar_loss(a, b, temperature):
    batch = len(a)
    rows = a + b
    unit = [[v / max(math.sqrt(sum(x*x for x in row)), 1e-12) for v in row] for row in rows]
    terms = []
    for i, row in enumerate(unit):
        scores = {j: sum(x*y for x,y in zip(row, other))/temperature
                  for j,other in enumerate(unit) if j != i}
        positive = i+batch if i < batch else i-batch
        peak = max(scores.values())
        terms.append(peak + math.log(sum(math.exp(x-peak) for x in scores.values())) - scores[positive])
    return sum(terms) / (2*batch)


def fixture(dtype=torch.float64):
    a = [[math.sin((i+1)*(d+2)) + .3*math.cos(i+2*d) for d in range(8)] for i in range(6)]
    b = [[math.cos((i+3)*(d+1)) - .2*math.sin(2*i+d) for d in range(8)] for i in range(6)]
    return torch.tensor(a,dtype=dtype),torch.tensor(b,dtype=dtype)


def measure(a,b,temperature):
    rows = a.tolist()+b.tolist();batch=len(a)
    unit=[[v/max(math.sqrt(sum(x*x for x in row)),1e-12) for v in row] for row in rows]
    positives=[];negatives=[];correct=0
    for i,row in enumerate(unit):
        positive=i+batch if i<batch else i-batch
        scores=[sum(x*y for x,y in zip(row,other)) for other in unit]
        pos=scores[positive];neg=max(scores[j] for j in range(2*batch) if j not in (i,positive))
        positives.append(pos);negatives.append(neg);correct+=pos>neg
    return {'loss':scalar_loss(a.tolist(),b.tolist(),temperature), 'accuracy':correct/(2*batch),
            'positive':sum(positives)/(2*batch), 'negative':sum(negatives)/(2*batch),
            'minimum_margin':min(x-y for x,y in zip(positives,negatives)),
            'positive_per_anchor':positives,'hardest_negative_per_anchor':negatives}
