"""Original CPU teaching experiment: decorrelate two paired embedding tables."""
import math
import torch


def diagnostics(z1, z2, lambd=.2, eps=1e-5):
    """Independent Python arithmetic, without calling either submitted function."""
    a, b = z1.detach().tolist(), z2.detach().tolist()
    n, d = len(a), len(a[0])
    normalized = []
    for rows in (a, b):
        means = [sum(row[j] for row in rows) / n for j in range(d)]
        scales = [math.sqrt(sum((row[j] - means[j])**2 for row in rows)/n + eps) for j in range(d)]
        normalized.append([[(row[j]-means[j])/scales[j] for j in range(d)] for row in rows])
    x, y = normalized
    c = [[sum(x[k][i]*y[k][j] for k in range(n))/n for j in range(d)] for i in range(d)]
    diagonal = sum((c[i][i]-1)**2 for i in range(d))
    off = sum(c[i][j]**2 for i in range(d) for j in range(d) if i != j)
    return c, diagonal + lambd * off, diagonal, off


def run(cross_correlation, barlow_twins_loss):
    angle = torch.arange(16, dtype=torch.float64) * (2 * torch.pi / 16)
    basis = torch.stack((angle.cos(), angle.sin(), (2 * angle).cos()), 1)
    x = (basis @ basis.new_tensor([[1., .9, .7], [.1, .3, .1], [.0, .1, .4]])).requires_grad_()
    y = (basis @ basis.new_tensor([[.9, .7, .8], [.2, .4, -.1], [.1, .1, .3]])).requires_grad_()
    optimizer = torch.optim.Adam([x, y], lr=.04)
    before, _, _, _ = diagnostics(x, y)
    losses, offdiagonals, steps = [], [], []
    initial = diagnostics(x, y)[1]
    for step in range(121):
        loss = barlow_twins_loss(x, y, .2)
        if step % 10 == 0:
            steps.append(step)
            losses.append(float(loss.detach()))
            offdiagonals.append(diagnostics(x, y)[3])
        if step < 120:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    after, oracle_loss, diagonal, offdiagonal = diagnostics(x, y)
    observed = cross_correlation(x, y).detach().tolist()
    return {
        'kind': 'toy_experiment', 'title': 'Barlow Twins: reduce feature redundancy',
        'problem': 'representation/barlow_twins',
        'provenance': {'kind': 'teaching_adaptation',
            'source': 'https://proceedings.mlr.press/v139/zbontar21a.html',
            'anchor': 'Equations 1–2 and official main.py BarlowTwins.forward',
            'adaptation': 'Optimize two 16-by-3 embedding tables for 120 Adam updates at learning rate .04 and lambda .2; no image encoder, data augmentation or benchmark pretraining.'},
        'metrics': {'initial_loss': initial, 'final_loss': losses[-1],
            'oracle_final_loss': oracle_loss, 'diagonal_error': diagonal,
            'off_diagonal_energy': offdiagonal, 'updates': 120,
            'correlation_error': max(abs(observed[i][j]-after[i][j]) for i in range(3) for j in range(3))},
        'charts': [
            {'kind': 'line', 'title': 'Submitted objective during real optimization', 'x_label': 'Step', 'x': steps,
             'series': [{'name': 'Loss', 'values': losses}]},
            {'kind': 'line', 'title': 'Independent redundancy diagnostic', 'x_label': 'Step', 'x': steps,
             'series': [{'name': 'Sum of squared off-diagonal correlations', 'values': offdiagonals}]},
            {'kind': 'bar', 'title': 'Cross-correlation before and after optimization', 'x_label': 'Feature pair',
             'x': [f'{i+1},{j+1}' for i in range(3) for j in range(3)],
             'series': [{'name': 'Before', 'values': [v for row in before for v in row]},
                        {'name': 'After', 'values': [v for row in after for v in row]}]},
        ],
    }
