"""Controlled batch-composition probe, not a trained-model benchmark."""
import math
import torch


def scalar_expected(x, groups, weight, bias, eps):
    """Independent Python-loop oracle: collect channels/positions without grouped tensor moments."""
    n, channels = x.shape[:2]
    spatial = math.prod(x.shape[2:])
    values = x.detach().reshape(n, channels, spatial).tolist()
    w, b = weight.detach().tolist(), bias.detach().tolist()
    output = []
    for sample in values:
        sample_output = []
        width = channels // groups
        for group in range(groups):
            members = [v for c in range(group * width, (group + 1) * width) for v in sample[c]]
            mean = math.fsum(members) / len(members)
            variance = math.fsum((v - mean)**2 for v in members) / len(members)
            inverse = 1 / math.sqrt(variance + eps)
            for c in range(group * width, (group + 1) * width):
                sample_output.append([(v - mean) * inverse * w[c] + b[c] for v in sample[c]])
        output.append(sample_output)
    return torch.tensor(output, dtype=x.dtype, device=x.device).reshape(x.shape)


def run(group_norm):
    anchor = torch.tensor([[-3., -1., 1., 3.], [2., 4., 6., 8.],
                           [-9., -3., 3., 9.], [10., 12., 14., 16.]], dtype=torch.float64).reshape(1, 4, 2, 2)
    weight, bias, eps = torch.ones(4, dtype=anchor.dtype), torch.zeros(4, dtype=anchor.dtype), 1e-5
    sizes, gn_changes, bn_changes = [1, 2, 4, 8], [], []
    def batch_norm(x):
        centered = x - x.mean(dim=(0, 2, 3), keepdim=True)
        return centered * torch.rsqrt(centered.square().mean(dim=(0, 2, 3), keepdim=True) + eps)
    baseline, bn_baseline = group_norm(anchor, 2, weight, bias, eps), batch_norm(anchor)
    scalar_error = 0.
    for size in sizes:
        batch = torch.cat([anchor] + [anchor * (i + 2) + 3 * (i + 1) for i in range(size - 1)], dim=0)
        output = group_norm(batch, 2, weight, bias, eps)
        scalar_error = max(scalar_error, (output - scalar_expected(batch, 2, weight, bias, eps)).abs().max().item())
        gn_changes.append((output[:1] - baseline).abs().max().item())
        bn_changes.append((batch_norm(batch)[:1] - bn_baseline).abs().max().item())
    group_series = []
    for groups, name in [(1, 'G=1: layer statistics'), (2, 'G=2: two channel groups'), (4, 'G=C: instance statistics')]:
        output = group_norm(anchor, groups, weight, bias, eps)
        scalar_error = max(scalar_error, (output - scalar_expected(anchor, groups, weight, bias, eps)).abs().max().item())
        group_series.append({'name': name, 'values': output.reshape(-1).tolist()})
    return {
        'provenance': 'PaperLeet synthetic teaching adaptation of GN Section 3; no trained-model accuracy or speed claim.',
        'charts': [
            {'kind': 'line', 'title': 'Change the companions, keep the same anchor sample',
             'x': sizes, 'x_label': 'batch size (including anchor)', 'y_label': 'maximum change in anchor output',
             'series': [{'name': 'Your GroupNorm, G=2', 'values': gn_changes},
                        {'name': 'Training batch statistics (supplied)', 'values': bn_changes}]},
            {'kind': 'line', 'title': 'Which activations share statistics?',
             'x': list(range(1, 17)), 'x_label': 'flattened activation in the same sample',
             'y_label': 'normalized value (unit scale, zero bias)', 'series': group_series},
        ],
        'metrics': {'maximum_scalar_error': scalar_error, 'maximum_group_context_change': max(gn_changes),
                    'minimum_batch_context_change_with_companions': min(bn_changes[1:])},
    }
