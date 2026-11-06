"""Original teaching adaptation: align two prediction tables to fixed crossed targets."""
import torch
from problems.representation.simsiam.toy import fixture, scalar_loss


def run(simsiam_loss):
    p1, p2, z1, z2 = fixture()
    p1.requires_grad_(); p2.requires_grad_()
    z1.requires_grad_(); z2.requires_grad_()
    original_z1, original_z2 = z1.detach().clone(), z2.detach().clone()
    # Include targets to make accidental target gradients observable in the trajectory.
    optimizer = torch.optim.SGD([p1, p2, z1, z2], lr=2.)
    steps, losses, agreement = [], [], []
    maximum_scalar_error = 0.
    for step in range(81):
        loss = simsiam_loss(p1, p2, z1, z2)
        independent = (scalar_loss(p1.detach().tolist(), z2.detach().tolist())
                       + scalar_loss(p2.detach().tolist(), z1.detach().tolist())) / 2
        maximum_scalar_error = max(maximum_scalar_error, abs(loss.item() - independent))
        if step % 4 == 0:
            steps.append(step); losses.append(loss.item()); agreement.append(-independent)
        if step < 80:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    movement = max((z1 - original_z1).abs().max().item(), (z2 - original_z2).abs().max().item())
    return {
        'kind': 'toy_experiment', 'title': 'SimSiam: follow the prediction gradients',
        'problem': 'representation/simsiam',
        'provenance': {
            'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2011.10566',
            'anchor': 'Section 3, Equations 1–4 and Algorithm 1',
            'adaptation': '80 SGD updates on two independent 4x3 prediction tables and fixed crossed target tables. Targets are offered to the optimizer, so only the stopped-target contract keeps them fixed. This isolates gradient routing; it is not shared-encoder image pretraining or a collapse-prevention benchmark.',
        },
        'metrics': {'updates': 80, 'trainable_prediction_parameters': 24, 'pairs': 4,
                    'initial_loss': losses[0], 'final_loss': losses[-1],
                    'initial_cosine': agreement[0], 'final_cosine': agreement[-1],
                    'target_movement': movement, 'maximum_scalar_error': maximum_scalar_error},
        'charts': [
            {'kind': 'line', 'title': 'Submitted symmetric negative cosine', 'x_label': 'SGD step',
             'x': steps, 'series': [{'name': 'Loss', 'values': losses}]},
            {'kind': 'line', 'title': 'Independent crossed-pair cosine', 'x_label': 'SGD step',
             'x': steps, 'series': [{'name': 'Mean cosine', 'values': agreement}]},
        ],
    }
