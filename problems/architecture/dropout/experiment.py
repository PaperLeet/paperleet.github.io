"""Explicit Bernoulli masks train linear regression; no generalization ranking."""
import torch


def run(inverted_dropout):
    rng = torch.Generator().manual_seed(193)
    x = torch.randn(64, 4, generator=rng, dtype=torch.float64)
    target = x @ torch.tensor([[1.], [-.5], [.75], [-1.]], dtype=x.dtype)
    w = torch.zeros(4, 1, dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=.06)
    losses = []
    for step in range(101):
        prediction = inverted_dropout(x, .25, training=False) @ w
        losses.append((prediction - target).square().mean().item())
        if step == 100:
            break
        mask = torch.rand(64, 4, generator=rng) >= .25
        output = inverted_dropout(x, .25, mask) @ w
        loss = (output - target).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    values = torch.tensor([1., -2.], dtype=x.dtype).expand(8192, 2)
    sampled = inverted_dropout(values, .25, torch.rand(8192, 2, generator=rng) >= .25)
    mean_error = (sampled.mean(dim=0) - values[0]).abs().max().item()
    variance = (sampled - sampled.mean(dim=0)).square().mean(dim=0)
    variance_error = (variance - values[0].square() / 3.).abs().max().item()
    return {'charts': [{'kind': 'line', 'title': 'Regression trained through sampled keep masks',
                       'x': list(range(101)), 'x_label': 'Adam update', 'y_label': 'evaluation MSE',
                       'series': [{'name': 'Your dropout-trained weights', 'values': losses}]}],
            'metrics': {'initial_mse': losses[0], 'final_mse': losses[-1],
                        'mc_mean_max_error': mean_error, 'mc_variance_max_error': variance_error}}
