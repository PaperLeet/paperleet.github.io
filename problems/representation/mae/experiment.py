"""Train a shared pixel decoder on deterministic synthetic context, without an image encoder."""
import math
import torch


def diagnostic(prediction, target, mask):
    a, b, m = prediction.detach().tolist(), target.detach().tolist(), mask.tolist()
    values = [sum((x-y)**2 for x, y in zip(a[i][j], b[i][j])) / len(a[i][j])
              for i in range(len(a)) for j in range(len(a[i])) if m[i][j]]
    return sum(values) / max(len(values), 1)


def run(masked_reconstruction_loss):
    t = torch.arange(24, dtype=torch.float64).reshape(6, 4)
    context = torch.stack(((t * .31).sin(), (t * .47).cos(), (t * .13).sin()), -1)
    truth = context @ context.new_tensor([[1., -.7, .3, .9], [.2, .6, -.5, .4], [-.4, .8, .7, -.2]]) + .25
    target = truth.clone().requires_grad_()
    mask = torch.tensor([[True, True, False, True], [False, True, False, False],
                         [True, False, True, False], [True, True, True, False],
                         [False, False, True, True], [True, False, True, True]])
    weight = torch.zeros(3, 4, dtype=context.dtype, requires_grad=True)
    bias = torch.zeros(4, dtype=context.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias, target], lr=.06)
    steps, losses, errors = [], [], []
    initial = diagnostic(context @ weight + bias, truth, mask)
    for step in range(121):
        prediction = context @ weight + bias
        loss = masked_reconstruction_loss(prediction, target, mask)
        if step % 10 == 0:
            steps.append(step); losses.append(float(loss.detach()))
            errors.append(diagnostic(prediction, truth, mask))
        if step < 120:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {'kind': 'toy_experiment', 'title': 'MAE: learn a shared masked-pixel decoder',
        'problem': 'representation/mae',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2111.06377',
            'anchor': 'Section 3 reconstruction target; official models_mae.py forward_loss',
            'adaptation': '120 Adam updates at lr .06 fit a shared 3-to-4 linear decoder on 24 deterministic synthetic context vectors. No ViT, image patchification, random masking pipeline, or ImageNet benchmark.'},
        'metrics': {'initial_mse': initial, 'final_mse': errors[-1], 'final_loss': losses[-1],
                    'target_drift': float((target.detach()-truth).abs().max()), 'updates': 120},
        'charts': [
            {'kind': 'line', 'title': 'Actual decoder optimization', 'x_label': 'Step', 'x': steps,
             'series': [{'name': 'Submitted loss', 'values': losses},
                        {'name': 'Independent masked-pixel MSE', 'values': errors}]},
            {'kind': 'bar', 'title': 'Pixels of the first reconstructed masked patch', 'x_label': 'Pixel coordinate',
             'x': ['1', '2', '3', '4'], 'series': [
                 {'name': 'Target', 'values': truth[0, 0].tolist()},
                 {'name': 'Decoded', 'values': prediction[0, 0].detach().tolist()}]},
        ]}
