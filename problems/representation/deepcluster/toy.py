import math
import torch


def fixture(dtype=torch.float64):
    return torch.tensor([[2., .1], [1.8, -.2], [2.2, .25],
                         [-1., 1.8], [-.8, 2.], [-1.2, 1.7],
                         [-1., -1.8], [-.8, -2.], [-1.2, -1.7]], dtype=dtype)


def scalar_ce(logits, labels):
    total = 0.
    for row, label in zip(logits, labels):
        peak = max(row)
        total += peak + math.log(sum(math.exp(x - peak) for x in row)) - row[label]
    return total / len(labels)


def scalar_assign(features, centroids):
    return [min(range(len(centroids)), key=lambda k: sum((a-b)**2 for a,b in zip(row,centroids[k])))
            for row in features]
