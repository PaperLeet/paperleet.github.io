import math
import torch


def fixture():
    return torch.tensor([[2., .1, .3], [1.1, .4, -.2], [-.3, 1.5, .4],
                         [-1.2, -.1, .7]],dtype=torch.float64)


def scalar_koleo(rows, eps=1e-8):
    unit = [[v/max(math.sqrt(sum(x*x for x in row)),eps) for v in row] for row in rows]
    indices = [min((j for j in range(len(unit)) if j != i),
                   key=lambda j: -sum(a*b for a,b in zip(row,unit[j]))) for i,row in enumerate(unit)]
    distances = [math.sqrt(sum((a-b+1e-8)**2 for a,b in zip(row,unit[j])))
                 for row,j in zip(unit,indices)]
    return -sum(math.log(x+eps) for x in distances)/len(rows)


def nearest_distance(rows):
    unit = [[v/math.sqrt(sum(x*x for x in row)) for v in row] for row in rows]
    return sum(min(math.sqrt(sum((a-b)**2 for a,b in zip(row,other)))
                   for j,other in enumerate(unit) if j != i) for i,row in enumerate(unit))/len(unit)
