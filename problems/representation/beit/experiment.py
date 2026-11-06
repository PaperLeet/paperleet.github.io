"""Learn a shared discrete-token predictor from synthetic context features."""
import math
import torch


def diagnostic(logits, labels, mask):
    rows, y, m = logits.detach().tolist(), labels.tolist(), mask.tolist()
    losses, correct = [], []
    for b in range(len(rows)):
        for n, row in enumerate(rows[b]):
            if m[b][n]:
                peak = max(row)
                losses.append(peak + math.log(sum(math.exp(v-peak) for v in row)) - row[y[b][n]])
                correct.append(int(row.index(max(row)) == y[b][n]))
    return sum(losses)/len(losses), sum(correct)/len(correct)


def run(masked_token_cross_entropy):
    labels = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1], [3, 2, 1, 0]], dtype=torch.int64)
    features = torch.eye(4, dtype=torch.float64)[labels]
    mask = torch.tensor([[True, False, False, False], [True, True, False, True], [True, False, True, False]])
    weight = torch.zeros(4, 4, dtype=features.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([weight], lr=.08)
    steps, losses, errors, accuracies = [], [], [], []
    for step in range(101):
        logits = features @ weight
        loss = masked_token_cross_entropy(logits, labels, mask)
        if step % 10 == 0:
            error, accuracy = diagnostic(logits, labels, mask)
            steps.append(step); losses.append(float(loss.detach())); errors.append(error); accuracies.append(accuracy)
        if step < 100:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {'kind': 'toy_experiment', 'title': 'BEiT: learn masked visual-token identities',
        'problem': 'representation/beit',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2106.08254',
            'anchor': 'Section 2 masked image modeling; official engine_for_pretraining.py',
            'adaptation': 'A 4-by-4 shared linear predictor learns six masked token labels from deterministic one-hot synthetic context in 100 Adam updates, lr .08. No image tokenizer, ViT, or benchmark pretraining.'},
        'metrics': {'initial_ce': errors[0], 'final_ce': errors[-1], 'final_loss': losses[-1],
                    'masked_accuracy': accuracies[-1], 'updates': 100},
        'charts': [
            {'kind': 'line', 'title': 'Actual masked-token optimization', 'x_label': 'Step', 'x': steps,
             'series': [{'name': 'Submitted loss', 'values': losses}, {'name': 'Independent CE', 'values': errors}]},
            {'kind': 'line', 'title': 'Synthetic masked-token accuracy', 'x_label': 'Step', 'x': steps,
             'series': [{'name': 'Accuracy', 'values': accuracies}]},
        ]}
