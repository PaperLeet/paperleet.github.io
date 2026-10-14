"""Relative logits and chunked KV-cache offsets; synthetic, no training."""
import math
import torch


def run(apply_rope):
    q_values, k_values = [.2, -.4, .7, 1.1, -.3, .8], [1., -.6, .4, .9, .5, -.2]
    positions = torch.arange(-8, 9, dtype=torch.float64)
    q = torch.tensor(q_values, dtype=positions.dtype)[None, None, None].expand(1, 1, 17, 6)
    k = torch.tensor(k_values, dtype=positions.dtype)[None, None, None].expand_as(q)
    rotated_q = apply_rope(q, positions, 31.)
    logits = (rotated_q * apply_rope(k, torch.zeros_like(positions), 31.)).sum(-1).reshape(-1)
    shifted = (apply_rope(q, positions + 25, 31.) * apply_rope(k, torch.zeros_like(positions) + 25, 31.)).sum(-1).reshape(-1)
    expected = []
    for position in positions.tolist():
        value = 0.
        for pair in range(3):
            angle = position * math.pow(31., -2 * pair / 6)
            a, b = q_values[2 * pair:2 * pair + 2]
            c, d = k_values[2 * pair:2 * pair + 2]
            value += (a * math.cos(angle) - b * math.sin(angle)) * c + (a * math.sin(angle) + b * math.cos(angle)) * d
        expected.append(value)
    keys = torch.arange(54, dtype=positions.dtype).reshape(1, 1, 9, 6) / 17 - 1
    queries = q[:, :, :3]
    absolute = torch.arange(9, dtype=positions.dtype)
    rotated_query = apply_rope(queries, absolute[6:], 31.)
    full = apply_rope(keys, absolute, 31.)
    cached = torch.cat([apply_rope(keys[:, :, :6], absolute[:6], 31.), apply_rope(keys[:, :, 6:], absolute[6:], 31.)], dim=2)
    reset = torch.cat([apply_rope(keys[:, :, :6], absolute[:6], 31.), apply_rope(keys[:, :, 6:], absolute[:3], 31.)], dim=2)
    scores = (rotated_query @ full.transpose(-1, -2)) / math.sqrt(6)
    cache_scores = (rotated_query @ cached.transpose(-1, -2)) / math.sqrt(6)
    reset_scores = (rotated_query @ reset.transpose(-1, -2)) / math.sqrt(6)
    return {
        'provenance': 'PaperLeet synthetic teaching adaptation of RoFormer Sections 3.2 and 3.4.2; not a paper benchmark.',
        'charts': [
            {'title': 'Joint position shifts preserve attention logits', 'kind': 'line', 'x': positions.tolist(),
             'x_label': 'query position minus key position', 'y_label': 'unscaled dot product',
             'series': [{'name': 'original positions', 'values': logits.tolist()},
                        {'name': 'both shifted by 25', 'values': shifted.tolist()},
                        {'name': 'independent scalar formula', 'values': expected}]},
            {'title': 'Cached keys retain absolute offsets', 'kind': 'line', 'x': absolute.tolist(),
             'x_label': 'key position', 'y_label': 'scaled logit for query at position 6',
             'series': [{'name': 'full sequence', 'values': scores[0, 0, 0].tolist()},
                        {'name': 'correct chunk offsets', 'values': cache_scores[0, 0, 0].tolist()},
                        {'name': 'wrong: restarted new keys at zero', 'values': reset_scores[0, 0, 0].tolist()}]},
        ],
        'metrics': {'maximum_scalar_error': (logits - torch.tensor(expected, dtype=positions.dtype)).abs().max().item(),
                    'joint_shift_error': (logits - shifted).abs().max().item(),
                    'correct_cache_logits_error': (scores - cache_scores).abs().max().item(),
                    'reset_cache_logits_gap': (scores - reset_scores).abs().max().item(),
                    'maximum_pair_norm_error': (rotated_q.reshape(1, 1, 17, 3, 2).square().sum(-1) - q.reshape(1, 1, 17, 3, 2).square().sum(-1)).abs().max().item()},
    }
