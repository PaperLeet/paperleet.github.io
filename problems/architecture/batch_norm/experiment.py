"""Fit affine parameters on synthetic batches and track detached population estimates."""
import math
import torch


def scalar_normalized(x, eps):
    """Independent scalar oracle for a matrix, reducing each column separately."""
    rows = x.detach().tolist()
    result = [[0.] * x.shape[1] for _ in rows]
    for c in range(x.shape[1]):
        mean = math.fsum(row[c] for row in rows) / len(rows)
        variance = math.fsum((row[c] - mean)**2 for row in rows) / len(rows)
        for n, row in enumerate(rows):
            result[n][c] = (row[c] - mean) / math.sqrt(variance + eps)
    return torch.tensor(result, dtype=x.dtype)


def run(batch_norm):
    generator = torch.Generator().manual_seed(91)
    weight = torch.full((2,), .2, dtype=torch.float64, requires_grad=True)
    bias = torch.ones(2, dtype=torch.float64, requires_grad=True)
    mean, variance = torch.zeros(2, dtype=weight.dtype), torch.ones(2, dtype=weight.dtype)
    optimizer = torch.optim.Adam([weight, bias], lr=.08)
    losses, means, variances = [], [], []
    for _ in range(80):
        x = 2 * torch.randn(32, 2, generator=generator, dtype=weight.dtype) + 3
        target = 1.5 * scalar_normalized(x, 1e-5) - .3
        output, mean, variance = batch_norm(x, weight, bias, mean, variance, True, .1)
        loss = (output - target).square().mean()
        losses.append(loss.item())
        means.append(mean[0].item())
        variances.append(variance[0].item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    anchor = torch.tensor([[2., 4.]], dtype=weight.dtype)
    alone = batch_norm(anchor, weight, bias, mean, variance, False)[0]
    together = batch_norm(torch.cat([anchor, anchor + 100]), weight, bias, mean, variance, False)[0][:1]
    return {
        'charts': [
            {'kind': 'line', 'title': 'Learn scale and shift from sampled minibatches',
             'x': list(range(80)), 'x_label': 'Adam update', 'y_label': 'mean squared error',
             'series': [{'name': 'Your training BatchNorm', 'values': losses}]},
            {'kind': 'line', 'title': 'Moving estimates for synthetic N(3, 4) inputs',
             'x': list(range(80)), 'x_label': 'minibatch', 'y_label': 'channel 0 statistic',
             'series': [{'name': 'Running mean', 'values': means}, {'name': 'Running variance', 'values': variances}]},
        ],
        'metrics': {'initial_mse': losses[0], 'final_mse': losses[-1],
                    'scale_error': (weight.detach() - 1.5).abs().max().item(),
                    'bias_error': (bias.detach() + .3).abs().max().item(),
                    'inference_companion_change': (alone - together).abs().max().item()},
    }
