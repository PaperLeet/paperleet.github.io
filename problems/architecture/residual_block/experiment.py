"""Synthetic positive-vector regression; not a CIFAR/ImageNet reproduction."""
import torch


def run(residual_block):
    generator = torch.Generator().manual_seed(73)
    x = torch.rand(32, 3, generator=generator, dtype=torch.float64) + .5
    target = x + .2 * x.square()
    parameters, blocks = [], []
    for _ in range(4):
        w1 = (.3 * torch.randn(3, 8, generator=generator, dtype=x.dtype)).requires_grad_()
        w2 = torch.zeros(8, 3, dtype=x.dtype, requires_grad=True)
        blocks.append((w1, w2))
        parameters.extend((w1, w2))
    optimizer = torch.optim.Adam(parameters, lr=.02)
    losses = []
    for step in range(81):
        output = x
        for w1, w2 in blocks:
            output = residual_block(output, w1, w2)
        loss = (output - target).square().mean()
        losses.append(loss.item())
        if step != 80:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {
        'charts': [{'kind': 'line', 'title': 'Four residual blocks learn a small nonlinear correction',
                    'x': list(range(81)), 'x_label': 'Adam update', 'y_label': 'mean squared error',
                    'series': [{'name': 'Your residual blocks', 'values': losses}]}],
        'metrics': {'initial_mse': losses[0], 'final_mse': losses[-1],
                    'maximum_prediction_error': (output - target).abs().max().item(),
                    'learned_residual_norm': sum(w2.detach().norm().item() for _, w2 in blocks)},
    }
