"""Synthetic prefix-bias experiment; no training or performance benchmark."""
import math
import torch


def run(rms_norm, partial_rms_norm):
    rows = torch.tensor([[1., -1., 1., -1., 1., -1., 1., -1.],
                         [.05, -.05, .1, -.1, 3., -3., 5., -5.]], dtype=torch.float64)
    weight, eps = torch.ones(8, dtype=rows.dtype), 1e-5
    full = rms_norm(rows, weight, eps)
    counts, errors = [1, 2, 4, 8], [[], []]
    maximum_error = scale_error = 0.
    for k in counts:
        partial = partial_rms_norm(rows, weight, k, eps)
        scaled = partial_rms_norm(7 * rows, weight, k, 49 * eps)
        scale_error = max(scale_error, (scaled - partial).abs().max().item())
        for i, row in enumerate(rows.tolist()):
            denominator = math.sqrt(sum(value * value for value in row[:k]) / k + eps)
            expected = torch.tensor([value / denominator for value in row], dtype=rows.dtype)
            maximum_error = max(maximum_error, (partial[i] - expected).abs().max().item())
            errors[i].append((partial[i] - full[i]).abs().max().item())
    # A separate offset sweep makes the absence of mean-centering visible.
    offsets = [-4., -2., 0., 2., 4.]
    means, rms_values = [], []
    for shift in offsets:
        output = rms_norm(rows[0] + shift, weight, eps)
        means.append(output.mean().item())
        rms_values.append(output.square().mean().sqrt().item())
    return {
        'provenance': 'PaperLeet synthetic teaching adaptation of RMSNorm Eq. (4) and Section 5; not a paper benchmark.',
        'charts': [
            {'title': 'Partial RMS: the prefix distribution matters', 'kind': 'line', 'x': counts,
             'x_label': 'prefix feature count k', 'y_label': 'maximum deviation from full RMSNorm',
             'series': [{'name': 'balanced prefix', 'values': errors[0]},
                        {'name': 'low-energy prefix', 'values': errors[1]}]},
            {'title': 'RMSNorm does not remove the mean', 'kind': 'line', 'x': offsets,
             'x_label': 'input offset', 'y_label': 'output statistic',
             'series': [{'name': 'output mean', 'values': means}, {'name': 'output RMS', 'values': rms_values}]},
        ],
        'metrics': {'maximum_scalar_error': maximum_error, 'scale_identity_error': scale_error,
                    'full_prefix_error': max(error[-1] for error in errors),
                    'biased_prefix_error_k1': errors[1][0], 'balanced_prefix_error': max(errors[0]),
                    'positive_offset_mean': means[-1]},
    }
