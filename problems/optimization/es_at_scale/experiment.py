"""A small deterministic-seed black-box search; no language model or backprop."""
import torch


def run(step_fn, seed=42, steps=100):
    generator = torch.Generator().manual_seed(seed)
    theta = torch.tensor([-2., 3.], dtype=torch.float64)
    target = torch.tensor([1., -1.], dtype=theta.dtype)
    losses, path = [], []
    for step in range(steps + 1):
        losses.append(float((theta - target).square().sum()))
        path.append(theta.tolist())
        if step == steps:
            break
        noise = torch.randn(30, 2, generator=generator, dtype=theta.dtype)
        candidates = theta.unsqueeze(0) + .1 * noise
        rewards = -(candidates - target).square().sum(dim=1)
        theta = step_fn(theta, noise, rewards, alpha=.05)
    return {'metrics': {'initial_loss': losses[0], 'final_loss': losses[-1]},
            'trajectory': path, 'loss_curve': [[i, v] for i, v in enumerate(losses)],
            'charts': [{'kind': 'line', 'title': 'Black-box distance to a known optimum',
                        'x': list(range(steps + 1)),
                        'series': [{'name': 'Squared distance', 'values': losses}]}]}
