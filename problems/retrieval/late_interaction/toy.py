"""Independent scalar cosine oracle and deliberately ambiguous pooled documents."""
import math
import torch


def scalar_scores(query, documents, query_mask, document_mask):
    def unit(vector):
        scale = max(math.sqrt(sum(value * value for value in vector)), 1e-12)
        return [value / scale for value in vector]
    rows = []
    for q, keep_q in zip(query, query_mask):
        scores = []
        for d, keep_d in zip(documents, document_mask):
            candidates = [unit(token) for token, keep in zip(d, keep_d) if keep]
            if not candidates:
                scores.append(-math.inf)
                continue
            score = 0.
            for token, keep in zip(q, keep_q):
                if keep:
                    encoded = unit(token)
                    score += max(sum(a * b for a, b in zip(encoded, candidate)) for candidate in candidates)
            scores.append(score)
        rows.append(scores)
    return rows


def data():
    basis = [[float(i == j) for i in range(4)] for j in range(4)]
    queries, documents, masks = [], [], []
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
        first, second = basis[a], basis[b]
        queries.append([[1.02 * x + y for x, y in zip(first, second)],
                        [x + 1.02 * y for x, y in zip(first, second)]])
        documents.extend([[[x + y for x, y in zip(first, second)], [100.] * 4, [100.] * 4],
                          [first, second, [100.] * 4]])
        masks.extend([[True, False, False], [True, True, False]])
    return (torch.tensor(queries, dtype=torch.float64), torch.tensor(documents, dtype=torch.float64),
            torch.ones(4, 2, dtype=torch.bool), torch.tensor(masks, dtype=torch.bool),
            torch.tensor([1, 3, 5, 7], dtype=torch.long), torch.tensor([0, 2, 4, 6], dtype=torch.long))


def pooled_pair_gap(documents, mask):
    """Each chosen positive/negative pair collapses to the same mean direction."""
    pooled = []
    for row, keep in zip(documents.tolist(), mask.tolist()):
        tokens = [token for token, valid in zip(row, keep) if valid]
        mean = [sum(token[i] for token in tokens) / len(tokens) for i in range(4)]
        size = math.sqrt(sum(value * value for value in mean))
        pooled.append([value / size for value in mean])
    return max(abs(a - b) for i in range(4) for a, b in zip(pooled[2*i], pooled[2*i+1]))
