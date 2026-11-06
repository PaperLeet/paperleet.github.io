import math
import torch


def fixture():
    base = [[0., 1., .2, 0.], [0., 1., 0., 0.], [0., 1., 1., .4], [0., 0., 0., 0.]]
    return torch.tensor([[[[value*(.8+.05*b)+.01*math.sin(b+3*r+c)
                            for c,value in enumerate(row)] for r,row in enumerate(base)]]
                          for b in range(8)], dtype=torch.float64)


def scalar_ce(rows, labels):
    total=0.
    for row,label in zip(rows,labels):
        peak=max(row)
        total += peak + math.log(sum(math.exp(x-peak) for x in row))-row[label]
    return total/len(rows)
