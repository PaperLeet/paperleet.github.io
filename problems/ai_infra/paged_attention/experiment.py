"""Teaching adaptation: replay growing decode contexts in scattered/shared physical pages."""
import math
import torch


def _dense(query, keys, values):
    # Independently compute attention with Python scalar arithmetic.
    output = []
    for head, q in enumerate(query.tolist()):
        scores = [sum(a*b for a, b in zip(q, token[head])) / math.sqrt(len(q)) for token in keys.tolist()]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        denominator = sum(weights)
        output.append([sum(w * token[head][dim] for w, token in zip(weights, values.tolist())) / denominator
                       for dim in range(len(q))])
    return query.new_tensor(output)


def run(gather_paged_kv, paged_attention):
    page_size, heads, width = 3, 2, 4
    base = torch.arange(8 * page_size * heads * width, dtype=torch.float64).reshape(8, page_size, heads, width)
    key_cache, value_cache = torch.sin(base / 7), torch.cos(base / 11)
    query = torch.sin(torch.arange(2 * heads * width, dtype=torch.float64).reshape(2, heads, width) / 3)
    # Both requests share page 5; their following logical blocks are different.
    tables = torch.tensor([[5, 1, 7], [5, 6, 2]], dtype=torch.long)
    errors, gather_errors, allocated, contiguous = [], [], [], []
    for length in range(1, 10):
        lengths = torch.tensor([length, max(1, length - 2)], dtype=torch.long)
        actual = paged_attention(query, key_cache, value_cache, tables, lengths)
        expected = []
        used = set()
        for request, context in enumerate(lengths.tolist()):
            # Token-level indexing differs from the learner's block-based gather.
            k = torch.stack([key_cache[int(tables[request, pos // page_size]), pos % page_size] for pos in range(context)])
            v = torch.stack([value_cache[int(tables[request, pos // page_size]), pos % page_size] for pos in range(context)])
            actual_k, actual_v = gather_paged_kv(key_cache, value_cache, tables[request], context)
            gather_errors.append(max((actual_k-k).abs().max().item(), (actual_v-v).abs().max().item()))
            expected.append(_dense(query[request], k, v))
            used.update(tables[request, :(context + page_size - 1) // page_size].tolist())
        errors.append((actual - torch.stack(expected)).abs().max().item())
        allocated.append(len(used) * page_size)
        contiguous.append(2 * 9)
    return {
        "kind": "toy_experiment", "title": "PagedAttention: scattered pages and shared prefixes",
        "problem": "ai_infra/paged_attention",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2309.06180",
                       "anchor": "Sections 4.1 and 4.2: block-table indirection and physical KV pages",
                       "adaptation": "Replay two CPU requests across nine context lengths, including one shared prefix page; compare with scalar dense attention and count token slots. No CUDA kernel or throughput benchmark."},
        "metrics": {"maximum_attention_error": max(errors), "maximum_gather_error": max(gather_errors),
                    "initial_paged_slots": allocated[0], "final_paged_slots": allocated[-1],
                    "reserved_contiguous_slots": contiguous[-1], "shared_prefix_slots_saved": page_size},
        "charts": [{"kind": "line", "title": "Attention agreement", "x_label": "Longest context length",
                    "x": list(range(1, 10)), "series": [{"name": "Max absolute error", "values": errors}]},
                   {"kind": "line", "title": "Allocated token slots", "x_label": "Longest context length",
                    "x": list(range(1, 10)), "series": [{"name": "Paged with shared prefix", "values": allocated},
                                                                        {"name": "Reserved capacity", "values": contiguous}]}],
    }
