import math
import torch


def fixture():
    queries = torch.tensor([[.3, -.2], [.7, .4]], dtype=torch.float64)
    memory = torch.tensor([[.8, .6], [-.6, .8], [-1., 0.], [.4, -.9]], dtype=torch.float64)
    candidates = torch.tensor([[2, 0, 1], [1, 3, 3]])
    return queries, memory, candidates


def softplus(x):
    return max(x, 0.) + math.log1p(math.exp(-abs(x)))


def scalar_nce(queries, memory, candidates, temperature, log_z):
    ratio = math.log((len(candidates[0]) - 1) / len(memory))
    total = 0.
    for q, ids in zip(queries, candidates):
        odds = [sum(a*b for a,b in zip(q,memory[j])) / temperature - log_z - ratio for j in ids]
        total += softplus(-odds[0]) + sum(softplus(v) for v in odds[1:])
    return total / len(queries)
