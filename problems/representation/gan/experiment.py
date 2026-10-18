"""Alternating scalar GAN updates on two shifted Gaussians; no image benchmark."""
import torch


def run(discriminator_loss, generator_loss):
    rng = torch.Generator().manual_seed(179)
    shift = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    slope = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    d_optimizer = torch.optim.Adam([slope, bias], lr=.03)
    g_optimizer = torch.optim.Adam([shift], lr=.03)
    positions, d_losses, g_losses = [0.], [], []
    for _ in range(240):
        real = torch.randn(64, generator=rng, dtype=torch.float64) + 2.
        fake = torch.randn(64, generator=rng, dtype=torch.float64) + shift
        d_loss = discriminator_loss(real * slope + bias, fake.detach() * slope + bias)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        generated = torch.randn(64, generator=rng, dtype=torch.float64) + shift
        g_loss = generator_loss(generated * slope.detach() + bias.detach())
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        positions.append(shift.item());d_losses.append(d_loss.item());g_losses.append(g_loss.item())
    return {'charts': [{'kind': 'line', 'title': 'Alternating adversarial updates learn a Gaussian location',
                       'x': list(range(241)), 'x_label': 'D/G update pair', 'y_label': 'generator location',
                       'series': [{'name': 'Your generator', 'values': positions}, {'name': 'Target location', 'values': [2.] * 241}]},
                      {'kind': 'line', 'title': 'Discriminator and non-saturating generator objectives',
                       'x': list(range(1, 241)), 'x_label': 'D/G update pair', 'y_label': 'loss',
                       'series': [{'name': 'Discriminator', 'values': d_losses}, {'name': 'Generator', 'values': g_losses}]}],
            'metrics': {'initial_location_error': 2., 'final_location_error': abs(shift.item() - 2.),
                        'final_generator_location': shift.item(), 'final_d_loss': d_losses[-1]}}
