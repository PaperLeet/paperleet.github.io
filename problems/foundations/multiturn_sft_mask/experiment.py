"""Original position-logit training probe; no tokenizer or language-model benchmark."""
import math
import torch


def fixture():
    roles = torch.tensor([[0, 1, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2],
                          [2, 2, 3, 2, 2, 2, 1, 2, 2, 2, 0, 2]], dtype=torch.long)
    conversations = torch.tensor([[10]*7 + [11]*3 + [-9, -9],
                                  [20]*4 + [21]*8], dtype=torch.long)
    attention = torch.tensor([[True]*10 + [False, False],
                              [True]*8 + [False, True, True, True]])
    tokens = torch.tensor([[0, 1, 2, 3, 1, 0, 3, 0, 2, 3, 3, 3],
                           [1, 2, 0, 1, 3, 0, 1, 2, 3, 2, 0, 3]], dtype=torch.long)
    # Independent hand-labelled targets: six in row 0, five in row 1.
    expected = torch.tensor([[False, False, True, True, False, True, True, False, True, True, False, False],
                             [False, True, False, True, False, True, False, True, False, False, False, True]])
    return roles, conversations, attention, tokens, expected


def run(solve):
    roles, conversations, attention, tokens, expected = fixture()
    selected = solve(roles, conversations, attention)
    assert selected.shape == roles.shape and selected.dtype == torch.bool
    logits = torch.zeros(2, 12, 4, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=4.)
    x, training, evaluation = [], [], []
    wanted = expected[:, 1:]
    for step in range(81):
        logp = logits[:, :-1].log_softmax(-1)
        nll = -logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        train_loss = torch.where(selected[:, 1:], nll, torch.zeros_like(nll)).sum() / selected[:, 1:].sum().clamp(min=1)
        eval_loss = torch.where(wanted, nll, torch.zeros_like(nll)).sum() / wanted.sum()
        if step % 4 == 0:
            x.append(step)
            training.append(float(train_loss.detach()))
            evaluation.append(float(eval_loss.detach()))
        if step == 80:
            break
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    predictor_mask = torch.zeros_like(expected)
    predictor_mask[:, :-1] = wanted
    excluded = torch.where(predictor_mask.unsqueeze(-1), torch.zeros_like(logits), logits.detach())
    correct = (logits[:, :-1].argmax(-1) == tokens[:, 1:]) & wanted
    return {
        'provenance': 'Original deterministic position-logit training using the submitted mask; not a language-model benchmark.',
        'charts': [
            {'kind': 'line', 'title': 'Assistant-target learning with packed conversations',
             'x': x, 'x_label': 'SGD step', 'y_label': 'Mean token NLL',
             'series': [{'name': 'Submitted-mask training loss', 'values': training},
                        {'name': 'Independent assistant targets', 'values': evaluation}]},
            {'kind': 'line', 'title': 'Target mask on the truncated second row',
             'x': list(range(12)), 'x_label': 'Target position', 'y_label': 'Selected (0 or 1)',
             'series': [{'name': 'Submitted mask', 'values': selected[1].to(torch.long).tolist()},
                        {'name': 'Hand-labelled target mask', 'values': expected[1].to(torch.long).tolist()}]},
        ],
        'metrics': {'selected_targets': int(selected.sum()), 'expected_targets': int(expected.sum()),
                    'initial_assistant_nll': evaluation[0], 'uniform_baseline_nll': math.log(4),
                    'final_assistant_nll': evaluation[-1],
                    'assistant_accuracy': float(correct.sum()) / int(wanted.sum()),
                    'excluded_logit_change': float(excluded.abs().max()),
                    'mask_disagreements': int((selected != expected).sum()), 'steps': 80},
    }
