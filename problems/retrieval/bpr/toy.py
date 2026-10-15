"""Independent scalar objective and balanced conflicting recommendation preferences."""
import math
from collections import Counter
import torch


def scalar_loss(users, items, triples, l2=0.):
    total = 0.
    for u, i, j in triples:
        margin = sum(a * (b-c) for a, b, c in zip(users[u], items[i], items[j]))
        total += max(-margin, 0.) + math.log1p(math.exp(-abs(margin)))
    prior = l2 * sum(v*v for table in (users, items) for row in table for v in row) if l2 else 0.
    return total / max(1, len(triples)) + prior


def all_triples(observed):
    return [[u,i,j] for u,row in enumerate(observed) for i,seen in enumerate(row) if seen
            for j,other in enumerate(row) if not other]


def item_only_accuracy_ceiling(triples):
    """Every unordered item pair admits only one global score direction."""
    directions = Counter((i,j) for _,i,j in triples)
    pairs = {tuple(sorted((i,j))) for i,j in directions}
    return sum(max(directions[(i,j)],directions[(j,i)]) for i,j in pairs) / max(1,len(triples))


def data():
    observed = torch.tensor([[True,True,False,False,False,False],
                             [False,False,True,True,False,False],
                             [False,False,False,False,True,True]])
    users = torch.tensor([[.1*math.sin((u+1)*(d+2)) for d in range(3)] for u in range(3)],dtype=torch.float64)
    items = torch.tensor([[.1*math.cos((i+2)*(d+1)) for d in range(3)] for i in range(6)],dtype=torch.float64)
    return observed, users, items
