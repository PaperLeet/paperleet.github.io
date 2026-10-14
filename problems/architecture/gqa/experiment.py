"""Synthetic checkpoint-conversion exercise; persistent cache counts, not timing."""
import torch
import torch.nn.functional as F


def run(mean_pool_kv_heads, grouped_query_attention):
    heads, length, model_width, key_width, value_width = 8, 6, 4, 2, 3
    x = torch.sin(torch.arange(length * model_width, dtype=torch.float64).reshape(1, length, model_width) * .31)
    wq = torch.cos(torch.arange(heads * model_width * key_width, dtype=x.dtype).reshape(heads, model_width, key_width) * .17)
    wk = torch.sin(torch.arange(wq.numel(), dtype=x.dtype).reshape(wq.shape) * .23)
    wv = torch.cos(torch.arange(heads * model_width * value_width, dtype=x.dtype).reshape(heads, model_width, value_width) * .19)
    q = torch.einsum('btd,hdf->bhtf', x, wq)
    full_k = torch.einsum('btd,hdf->bhtf', x, wk)
    full_v = torch.einsum('btd,hdf->bhtf', x, wv)
    visible = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    baseline = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=visible)
    groups, ratios, errors, counts, maximum_error = [1, 2, 4, 8], [], [], [], 0.
    for count in groups:
        k = torch.einsum('btd,hdf->bhtf', x, mean_pool_kv_heads(wk, count))
        v = torch.einsum('btd,hdf->bhtf', x, mean_pool_kv_heads(wv, count))
        actual, _ = grouped_query_attention(q, k, v, visible)
        # Independent routing uses one native/library attention call per query
        # head, with no repeat_interleave or call to the learner's attention.
        expected = torch.cat([F.scaled_dot_product_attention(q[:, h:h+1],
            k[:, h // (heads // count):h // (heads // count)+1],
            v[:, h // (heads // count):h // (heads // count)+1], attn_mask=visible)
            for h in range(heads)], dim=1)
        maximum_error = max(maximum_error, (actual - expected).abs().max().item())
        counts.append(k.numel() + v.numel())
        ratios.append(counts[-1] / (full_k.numel() + full_v.numel()))
        errors.append((actual - baseline).square().mean().sqrt().item())
    return {
        'provenance': 'PaperLeet synthetic teaching adaptation of GQA Sections 2.1–2.2; no uptraining, quality or timing benchmark.',
        'charts': [
            {'title': 'Persistent KV cache: fewer shared heads', 'kind': 'line', 'x': groups,
             'x_label': 'KV groups G (1 = MQA, 8 = MHA)', 'y_label': 'fraction of original KV elements',
             'series': [{'name': 'stored KV / original MHA KV', 'values': ratios}]},
            {'title': 'Mean pooling changes outputs before uptraining', 'kind': 'line', 'x': groups,
             'x_label': 'KV groups G', 'y_label': 'synthetic output RMSE vs original MHA',
             'series': [{'name': 'converted output RMSE', 'values': errors}]},
        ],
        'metrics': {'maximum_attention_oracle_error': maximum_error, 'mha_endpoint_rmse': errors[-1],
                    'mqa_conversion_rmse': errors[0], 'mqa_cache_ratio': ratios[0],
                    'gqa2_cache_ratio': ratios[1], 'mha_cache_elements': counts[-1]},
    }
