"""Random linear experts, disjoint selection/test grids and discrete voting."""
import torch


def run(perturb_fn, select_fn, vote_fn, seed=42):
    generator = torch.Generator().manual_seed(seed)
    theta = torch.tensor([.2, .8], dtype=torch.float64)
    noise = torch.randn(120, 2, generator=generator, dtype=theta.dtype)
    scales = torch.tensor([.25, .5, 1.] * 40, dtype=theta.dtype)
    candidates = perturb_fn(theta, noise, scales)
    # Selection and test samples have distinct coordinates; scores use only selection labels.
    select_x = torch.linspace(-2., 2., 41, dtype=theta.dtype)
    test_x = torch.linspace(-1.975, 1.975, 80, dtype=theta.dtype)
    train_labels = (select_x > 0).long()
    test_labels = (test_x > 0).long()
    train_pred = (candidates[:, :1] * select_x + candidates[:, 1:] > 0).long()
    scores = (train_pred == train_labels).to(theta.dtype).mean(dim=1)
    selected = select_fn(scores, 9)
    test_predictions = (candidates[:, :1] * test_x + candidates[:, 1:] > 0).long()
    voted = vote_fn(test_predictions, selected, 2)
    base = (theta[0] * test_x + theta[1] > 0).long()
    return {'metrics': {'base_accuracy': float((base == test_labels).to(theta.dtype).mean()),
                        'ensemble_accuracy': float((voted == test_labels).to(theta.dtype).mean()),
                        'selected_count': int(selected.numel())},
            'selected': selected.tolist(),
            'charts': [{'kind': 'line', 'title': 'Held-out decisions of selected random experts',
                        'x': test_x.tolist(), 'series': [{'name': 'Target', 'values': test_labels.tolist()},
                        {'name': 'Base', 'values': base.tolist()}, {'name': 'Ensemble', 'values': voted.tolist()}]}]}
