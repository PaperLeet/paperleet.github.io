"""A fixed preference minibatch trains a scalar affine noise predictor."""
import torch


def run(paired_noising, diffusion_dpo_loss):
    dtype = torch.float64
    winner = torch.tensor([[1.], [1.5], [.8], [1.2]], dtype=dtype)
    loser = -winner
    noise = torch.tensor([[.2], [-.3], [.1], [-.1]], dtype=dtype)
    alpha = torch.tensor([.5, .6, .4, .7], dtype=dtype)
    xw, xl = paired_noising(winner, loser, noise, alpha)
    ref_w, ref_l = .5 * xw, .5 * xl
    parameters = torch.tensor([.5, 0.], dtype=dtype, requires_grad=True)
    losses, margins = [], []
    for step in range(81):
        w, b = parameters[0], parameters[1]
        pred_w, pred_l = w * xw + b, w * xl + b
        loss = diffusion_dpo_loss(pred_w, pred_l, ref_w, ref_l, noise, 2.)
        margin = -((pred_w - noise).square() - (pred_l - noise).square()
                   - (ref_w - noise).square() + (ref_l - noise).square()).mean(1)
        losses.append(loss.item()); margins.append(margin.detach().tolist())
        if step < 80:
            gradient, = torch.autograd.grad(loss, parameters)
            parameters = (parameters - .05 * gradient).detach().requires_grad_()
    return {
        'kind': 'toy_experiment', 'title': 'Diffusion-DPO: a tiny denoiser learns preferences',
        'problem': 'diffusion_rl/diffusion_dpo',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2311.12908',
                       'anchor': 'Section 4; official train.py paired noising and DPO loss',
                       'adaptation': 'Four fixed scalar latent pairs and a trainable two-parameter affine noise predictor; no image model or preference dataset.'},
        'metrics': {'initial_loss': losses[0], 'final_loss': losses[-1],
                    'minimum_margin': min(margins[-1]),
                    'shared_noise_error': ((xw - xl) - alpha.sqrt()[:, None] * (winner - loser)).abs().max().item()},
        'charts': [
            {'kind': 'line', 'title': 'Pairwise preference loss', 'x_label': 'Gradient step',
             'x': list(range(len(losses))), 'series': [{'name': 'Diffusion-DPO', 'values': losses}]},
            {'kind': 'line', 'title': 'Reference-relative preference margin', 'x_label': 'Gradient step',
             'x': list(range(len(margins))), 'series': [{'name': f'Pair {i + 1}', 'values': [m[i] for m in margins]} for i in range(4)]},
        ],
    }
