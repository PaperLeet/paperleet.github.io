"""Independent scalar checks and a deliberately small causal recommendation model."""
import math
import torch
from torch.nn import functional as F


def scalar_mask(ids):
    return [[[bool(query and key and j <= i) for j, key in enumerate(row)]
             for i, query in enumerate(row)] for row in ids]


def scalar_loss(positive, negative, targets):
    terms = []
    for ps, ns, ids in zip(positive, negative, targets):
        for p, n, target in zip(ps, ns, ids):
            if target:
                terms.append(max(-p, 0.) + math.log1p(math.exp(-abs(p)))
                             + max(n, 0.) + math.log1p(math.exp(-abs(n))))
    return sum(terms) / len(terms) if terms else 0.


def data():
    histories = torch.tensor([[0, 1, 3, 4], [0, 2, 3, 4], [0, 0, 0, 0]])
    targets = torch.tensor([[0, 3, 4, 5], [0, 3, 4, 6], [0, 0, 0, 0]])
    embeddings = torch.tensor([[0.]*6] + [[float(i == j) + .03*math.cos((i+1)*(j+2)) for j in range(6)] for i in range(6)], dtype=torch.float64)
    return histories, targets, embeddings


def encode(item_ids, embeddings, causal_item_mask, query_projection=None):
    length, dim = item_ids.shape[1], embeddings.shape[1]
    positions = torch.tensor([[.03*math.sin((i+1)*(j+1)) for j in range(dim)] for i in range(length)], dtype=embeddings.dtype)
    x = (embeddings[item_ids] * math.sqrt(dim) + positions) * (item_ids != 0)[..., None]
    allowed = causal_item_mask(item_ids)
    query = x if query_projection is None else x @ query_projection
    scores = (query @ x.transpose(-1, -2)) / math.sqrt(dim)
    scores = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
    attention = safe.softmax(-1).masked_fill(~allowed, 0.)
    states = F.layer_norm(x + attention @ x, (dim,)) * (item_ids != 0)[..., None]
    return states, attention


def last_item_accuracy_ceiling(histories, targets):
    """Exact majority bound for any predictor seeing only item ID and position."""
    from collections import Counter
    groups = {}
    for history, target in zip(histories.tolist(), targets.tolist()):
        for position, (item, wanted) in enumerate(zip(history, target)):
            if wanted:
                groups.setdefault((item, position), Counter())[wanted] += 1
    return sum(max(counts.values()) for counts in groups.values()) / sum(sum(counts.values()) for counts in groups.values())
