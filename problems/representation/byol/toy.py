"""Scalar reference calculations for the independently authored teaching lab."""
import math
import torch


def normalized_rows(rows, eps=1e-12):
    return [[x / math.sqrt(max(sum(t*t for t in row), eps)) for x in row] for row in rows]


def scalar_loss(p1, p2, z1, z2, eps=1e-12):
    p1, p2, z1, z2 = [normalized_rows(rows, eps) for rows in (p1, p2, z1, z2)]
    return sum(sum((a-b)**2 for a,b in zip(p,z)) for pp,zz in ((p1,z2),(p2,z1)) for p,z in zip(pp,zz)) / len(p1)


def fixture(dtype=torch.float64):
    rows = torch.arange(18, dtype=dtype).reshape(6, 3)
    return ((rows*.7).sin()+.2, (rows*.53+.4).cos()-.1,
            (rows*.31+.5).cos()+.3, (rows*.43-.3).sin()-.2)
