"""Online Gaussian transitions train a conditional scalar velocity model."""
import torch


def run(sde_transition, flow_grpo_loss):
    generator = torch.Generator().manual_seed(42)
    dtype = torch.float64
    prompts = torch.tensor([-1., .5, 1.5], dtype=dtype)
    groups = 24
    state = torch.zeros(3 * groups, 1, dtype=dtype)
    conditioning = prompts[:, None].expand(-1, groups).reshape(-1, 1)
    weight = torch.tensor(0., dtype=dtype, requires_grad=True)
    errors, losses = [], []
    time, next_time, noise_level = .7, .4, .6
    frozen_reference, std = sde_transition(state, torch.zeros_like(state), time, next_time, noise_level)
    for step in range(81):
        mean, std = sde_transition(state, -weight * conditioning, time, next_time, noise_level)
        errors.append((mean.reshape(3, groups).mean(1) - prompts).square().mean().item())
        if step == 80:
            break
        old = mean.detach()
        actions = old + std * torch.randn(old.shape, generator=generator, dtype=dtype)
        rewards = -(actions.reshape(3, groups) - prompts[:, None]).square()
        shape = (3, groups, 1, 1)
        loss = flow_grpo_loss(mean.reshape(shape), old.reshape(shape), frozen_reference.reshape(shape),
                              actions.reshape(shape), std.reshape(1), rewards, .2, .01)
        losses.append(loss.item())
        gradient, = torch.autograd.grad(loss, weight)
        weight = (weight - .15 * gradient).detach().requires_grad_()
    return {
        'kind': 'toy_experiment', 'title': 'Flow-GRPO: online stochastic velocity learning',
        'problem': 'diffusion_rl/flow_grpo',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2505.05470',
                       'anchor': 'Section 4.1 Eqs. (4)-(6); Section 4.2 Eqs. (11)-(13)',
                       'adaptation': 'Three scalar prompt targets, 24 fresh one-transition trajectories per prompt and 80 actual policy updates; no SD3 model or image rewards.'},
        'metrics': {'initial_expected_error': errors[0], 'final_expected_error': errors[-1],
                    'rollout_std': std.item(), 'updates': len(losses), 'learned_weight': weight.item()},
        'charts': [
            {'kind': 'line', 'title': 'Error of the expected transition', 'x_label': 'Online update',
             'x': list(range(len(errors))), 'series': [{'name': 'Mean squared target error', 'values': errors}]},
        ],
    }
