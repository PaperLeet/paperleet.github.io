"""Independent scalar oracles and a tiny overlapping user-item graph."""
import math
from collections import Counter
import torch


def scalar_graph(history, items=None):
    users = len(history)
    items = len(history[0]) if users else (0 if items is None else items)
    degrees = [sum(row) for row in history] + [sum(row[i] for row in history) for i in range(items)]
    graph = [[0. for _ in degrees] for _ in degrees]
    for u, row in enumerate(history):
        for i, seen in enumerate(row):
            if seen:
                graph[u][users+i] = graph[users+i][u] = 1 / math.sqrt(degrees[u] * degrees[users+i])
    return graph


def scalar_propagation(initial, graph, layers):
    state = [row[:] for row in initial]
    total = [row[:] for row in initial]
    for _ in range(layers):
        state = [[sum(graph[u][v] * state[v][d] for v in range(len(graph)))
                  for d in range(len(row))] for u, row in enumerate(state)]
        total = [[a+b for a,b in zip(old,new)] for old,new in zip(total,state)]
    return [[value/(layers+1) for value in row] for row in total]


def scalar_objective(initial, graph, layers, triples, users, l2):
    final = scalar_propagation(initial, graph, layers)
    total = 0.
    for u,i,j in triples:
        margin = sum(a*(b-c) for a,b,c in zip(final[u],final[users+i],final[users+j]))
        total += max(-margin,0.) + math.log1p(math.exp(-abs(margin)))
    prior = l2 * sum(v*v for row in initial for v in row) if l2 else 0.
    return total / max(1,len(triples)) + prior


def all_triples(history):
    return [[u,i,j] for u,row in enumerate(history) for i,seen in enumerate(row) if seen
            for j,other in enumerate(row) if not other]


def item_only_accuracy_ceiling(triples):
    directions = Counter((i,j) for _,i,j in triples)
    pairs = {tuple(sorted((i,j))) for i,j in directions}
    return sum(max(directions[(i,j)],directions[(j,i)]) for i,j in pairs) / max(1,len(triples))


def data():
    history = torch.tensor([[True,True,False,False,False,False],
                            [False,True,True,False,False,False],
                            [False,False,False,True,True,False],
                            [False,False,False,False,False,False]])
    initial = torch.tensor([[.1*math.sin((n+1)*(d+2)) + .07*math.cos((n+2)*(d+1))
                             for d in range(3)] for n in range(10)],dtype=torch.float64)
    return history, initial
