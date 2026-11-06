"""Train a six-by-four score table using the supplied InfoNCE implementation."""
import torch
from .toy import fixture, measure


def run(info_nce_loss):
    values, targets = fixture()
    logits = torch.nn.Parameter(values)
    optimizer = torch.optim.SGD([logits], lr=1.5)
    steps, losses, probabilities, accuracies = [], [], [], []
    maximum_error = 0.

    def checkpoint(step):
        nonlocal maximum_error
        with torch.no_grad():
            metrics = measure(logits, targets)
            actual = float(info_nce_loss(logits, targets))
        maximum_error = max(maximum_error, abs(actual - metrics['loss']))
        steps.append(step)
        losses.append(actual)
        probabilities.append(metrics['positive_probability'])
        accuracies.append(metrics['accuracy'])
        return metrics

    initial = checkpoint(0)
    for step in range(1, 81):
        optimizer.zero_grad()
        loss = info_nce_loss(logits, targets)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            final = checkpoint(step)
    provenance = ('Original synthetic score-table optimization: six contexts, four candidates each, '
                  '24 trainable logits, 80 full-batch SGD updates at learning rate 1.5. '
                  'The positive column varies by context. This teaches the objective; '
                  'it does not train a CPC encoder or measure downstream representation quality.')
    return {'provenance': provenance,
            'charts': [
                {'kind': 'line', 'title': 'InfoNCE during actual optimization',
                 'x': steps, 'x_label': 'SGD update', 'y_label': 'Mean loss',
                 'series': [{'name': 'Training loss', 'values': losses}]},
                {'kind': 'line', 'title': 'Finding the positive candidate',
                 'x': steps, 'x_label': 'SGD update', 'y_label': 'Fraction',
                 'series': [{'name': 'Mean positive probability', 'values': probabilities},
                            {'name': 'Strict top-1 accuracy', 'values': accuracies}]},
                {'kind': 'bar', 'title': 'Final positive probability by context',
                 'x': [f'Context {i + 1}' for i in range(6)],
                 'series': [{'name': 'Positive probability', 'values': final['probabilities']}]}],
            'metrics': {'updates': 80, 'trainable_parameters': logits.numel(), 'contexts': 6,
                        'candidates': 4, 'initial_loss': losses[0], 'final_loss': losses[-1],
                        'initial_accuracy': initial['accuracy'], 'final_accuracy': final['accuracy'],
                        'final_positive_probability': final['positive_probability'],
                        'final_minimum_margin': final['minimum_margin'],
                        'maximum_scalar_error': maximum_error}}
