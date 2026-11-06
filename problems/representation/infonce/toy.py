"""Independent scalar measurements for the original six-context teaching fixture."""
import math
import torch


def scalar_loss(rows, targets):
    terms = []
    for row, target in zip(rows, targets):
        peak = max(row)
        terms.append(peak + math.log(sum(math.exp(value - peak) for value in row)) - row[target])
    return sum(terms) / len(terms)


def fixture(dtype=torch.float64):
    rows = [[math.sin((i + 1) * (j + 2)) + .3 * math.cos(i + 2 * j)
             for j in range(4)] for i in range(6)]
    return torch.tensor(rows, dtype=dtype), torch.tensor([1, 3, 0, 2, 1, 0], dtype=torch.int64)


def measure(logits, positive_indices):
    rows, targets = logits.tolist(), positive_indices.tolist()
    probabilities, margins = [], []
    for row, target in zip(rows, targets):
        peak = max(row)
        denominator = sum(math.exp(value - peak) for value in row)
        probabilities.append(math.exp(row[target] - peak) / denominator)
        margins.append(row[target] - max(value for j, value in enumerate(row) if j != target))
    return {'loss': scalar_loss(rows, targets),
            'positive_probability': sum(probabilities) / len(rows),
            'accuracy': sum(margin > 0 for margin in margins) / len(rows),
            'minimum_margin': min(margins), 'probabilities': probabilities}
