"""Frozen two-pixel readout and supplied teacher targets; prompt-only adaptation."""
import torch


def run(apply_fn, factor_fn, penalty_fn, steps=120):
    dtype = torch.float64
    # A fixed identity readout exposes two prompt coordinates. These are features,
    # not classifier probabilities; squared consistency makes an exact oracle possible.
    images = torch.zeros(4, 1, 1, 2, dtype=dtype)
    specific = torch.zeros(1, 1, 2, dtype=dtype)
    anchor = torch.zeros_like(specific)
    teacher = torch.tensor([1., -1.], dtype=dtype)
    importance = torch.tensor([[[[.04, 0.]]]], dtype=dtype)
    displacement = torch.zeros_like(importance)
    factor = factor_fn(importance, displacement)
    results, charts = {}, []
    for name, strength in [('unregularized', 0.), ('regularized', 1.)]:
        prompt = torch.zeros_like(specific).requires_grad_()
        optimizer = torch.optim.SGD([prompt], lr=.1)
        losses = []
        for step in range(steps + 1):
            prediction = apply_fn(images, specific, prompt).reshape(4, 2)
            data_loss = .5 * (prediction - teacher).square().sum(dim=1).mean()
            loss = data_loss + penalty_fn(prompt, anchor, factor, strength)
            losses.append(float(data_loss.detach()))
            if step == steps:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results[name + '_prompt'] = prompt.detach().reshape(-1).tolist()
        results[name + '_data_loss'] = losses[-1]
        charts.append({'name': name, 'values': losses})
    return {'metrics': results, 'charts': [{'kind': 'line', 'title': 'Prompt retention / adaptation tradeoff',
            'x': list(range(steps + 1)), 'series': charts}]}
