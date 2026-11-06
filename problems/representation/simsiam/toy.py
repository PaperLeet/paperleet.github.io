"""Independent scalar cosine oracle and deterministic asymmetric vectors."""
import math
import torch


def scalar_loss(prediction, target):
    total = 0.
    for p, z in zip(prediction, target):
        np = max(math.sqrt(sum(v * v for v in p)), 1e-12)
        nz = max(math.sqrt(sum(v * v for v in z)), 1e-12)
        total -= sum((a / np) * (b / nz) for a, b in zip(p, z))
    return total / len(prediction)


def fixture(dtype=torch.float64):
    return tuple(torch.tensor(rows, dtype=dtype) for rows in (
        [[1., -2., .5], [.3, 1.2, -1.], [-1., .8, 2.], [1.3, .4, -.7]],
        [[-.6, .7, 1.1], [2., -.5, .4], [.3, -1.2, .9], [-.8, .3, 1.6]],
        [[.7, 1., -.3], [-.4, 1.8, .2], [1.2, -.6, .3], [.6, -1.1, -.5]],
        [[-.5, .3, 1.4], [.8, .2, -1.3], [1.1, .6, -.7], [-.4, 1.2, .9]],
    ))
