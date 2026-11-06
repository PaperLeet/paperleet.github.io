"""Spread six learnable vectors on a sphere using the submitted regularizer."""
import math
import torch
from .toy import nearest_distance


def run(koleo_loss):
    features = torch.nn.Parameter(torch.tensor([[1., .12*math.sin(i), .12*math.cos(1.7*i)]
                                                for i in range(6)],dtype=torch.float64))
    optimizer = torch.optim.SGD([features],lr=.06)
    steps, losses, distances = [], [], []
    for step in range(101):
        loss = koleo_loss(features)
        if step % 10 == 0:
            steps.append(step); losses.append(float(loss.detach()))
            distances.append(nearest_distance(features.detach().tolist()))
        if step < 100:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {'provenance': 'Original six-vector, three-dimensional sphere-spreading lab: 18 trainable entries and 100 SGD updates (lr .06), deterministic trigonometric initialization. KoLeo alone is optimized; no image teacher, DINO/iBOT objective or downstream score is reproduced.',
            'charts': [{'kind': 'line','title': 'Spreading nearest neighbors','x': steps,
                        'x_label': 'SGD update','y_label': 'Training metric',
                        'series': [{'name': 'KoLeo loss','values': losses},
                                   {'name': 'Mean nearest distance','values': distances}]}],
            'metrics': {'updates': 100,'trainable_parameters': 18,
                        'initial_loss': losses[0],'final_loss': losses[-1],
                        'initial_nearest_distance': distances[0],
                        'final_nearest_distance': distances[-1]}}
