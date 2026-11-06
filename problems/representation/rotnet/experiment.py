"""Learn rotation labels of eight asymmetric synthetic glyphs with a linear head."""
import torch
from .toy import fixture


def run(rotation_batch, rotation_loss):
    images, labels = rotation_batch(fixture())
    flat = images.reshape(32,16)
    weight = torch.nn.Parameter(torch.zeros((16,4),dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=.6)
    steps, losses, accuracies = [], [], []
    for step in range(81):
        logits = flat @ weight
        loss = rotation_loss(logits,labels)
        if step % 10 == 0:
            steps.append(step); losses.append(float(loss.detach()))
            accuracies.append(float((logits.argmax(1)==labels).double().mean()))
        if step < 80:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {'provenance': 'Original eight asymmetric 4x4 glyphs and their 32 exact rotations; 64 linear-head parameters, 80 full-batch SGD updates at lr .6, deterministic data and zero initialization. These are pretext training metrics, not transfer performance.',
            'charts': [{'kind': 'line','title': 'Learning self-generated rotation labels','x': steps,
                        'x_label': 'SGD update','y_label': 'Training metric',
                        'series': [{'name': 'Rotation CE','values': losses},
                                   {'name': 'Rotation accuracy','values': accuracies}]}],
            'metrics': {'updates': 80,'trainable_parameters': 64,'augmented_images': 32,
                        'initial_loss': losses[0],'final_loss': losses[-1],
                        'initial_accuracy': accuracies[0],'final_accuracy': accuracies[-1]}}
