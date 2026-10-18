"""A bidirectional context table, not a Transformer or language benchmark."""
import torch


def run(make_mlm_inputs, masked_lm_loss):
    rng = torch.Generator().manual_seed(163)
    left = torch.arange(16).repeat(4) // 4
    right = torch.arange(16).repeat(4) % 4
    center = (2 * left + right) % 4
    tokens = torch.stack((left, center, right), dim=1)
    table = torch.zeros(16, 4, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([table], lr=.08)
    full_labels = torch.full_like(tokens, -100)
    full_labels[:, 1] = center
    losses = []
    for step in range(101):
        prediction = table[left * 4 + right].unsqueeze(1).expand(-1, 3, -1)
        losses.append(masked_lm_loss(prediction, full_labels).item())
        if step == 100:
            break
        selected = torch.zeros_like(tokens, dtype=torch.bool)
        selected[:, 1] = torch.rand(64, generator=rng) < .15
        draws = torch.rand(64, 3, generator=rng, dtype=torch.float64)
        replacements = torch.randint(0, 4, (64, 3), generator=rng)
        corrupted, labels = make_mlm_inputs(tokens, selected, draws, replacements, 4)
        context = corrupted[:, 0] * 4 + corrupted[:, 2]
        logits = table[context].unsqueeze(1).expand(-1, 3, -1)
        loss = masked_lm_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    expected = (2 * (torch.arange(16) // 4) + torch.arange(16) % 4) % 4
    accuracy = (table.argmax(dim=-1) == expected).to(torch.float64).mean().item()
    return {'charts': [{'kind': 'line', 'title': 'Selected-token learning from left and right context',
                       'x': list(range(101)), 'x_label': 'Adam update', 'y_label': 'masked-token NLL',
                       'series': [{'name': 'Your MLM loss', 'values': losses}]}],
            'metrics': {'initial_nll': losses[0], 'final_nll': losses[-1], 'context_accuracy': accuracy}}
