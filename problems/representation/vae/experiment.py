"""Tiny Bernoulli VAE; synthetic patterns, not MNIST or an image benchmark."""
import torch
import torch.nn.functional as F


def run(reparameterize, gaussian_kl, negative_elbo):
    generator = torch.Generator().manual_seed(107)
    patterns = torch.tensor([[1., 1., 1., 0., 0., 0.], [0., 0., 0., 1., 1., 1.]], dtype=torch.float64)
    x = patterns.repeat(16, 1)
    def parameter(shape, scale=.1):
        return (scale * torch.randn(*shape, generator=generator, dtype=x.dtype)).requires_grad_()
    encoder = parameter((6, 4))
    encoder_bias = torch.zeros(4, dtype=x.dtype, requires_grad=True)
    decoder = parameter((2, 6))
    decoder_bias = torch.zeros(6, dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([encoder, encoder_bias, decoder, decoder_bias], lr=.04)
    losses, reconstructions, divergences = [], [], []
    for step in range(120):
        statistics = x @ encoder + encoder_bias
        mu, logvar = statistics[:, :2], statistics[:, 2:]
        noise = torch.randn(32, 2, generator=generator, dtype=x.dtype)
        z = reparameterize(mu, logvar, noise)
        logits = z @ decoder + decoder_bias
        loss = negative_elbo(logits, x, mu, logvar)
        reconstruction = (F.softplus(logits) - x * logits).sum(dim=1).mean()
        divergence = gaussian_kl(mu, logvar).mean()
        losses.append(loss.item())
        reconstructions.append(reconstruction.item())
        divergences.append(divergence.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tail = sum(losses[-20:]) / 20
    return {
        'charts': [{'kind': 'line', 'title': 'A tiny VAE learns two complementary Bernoulli patterns',
                    'x': list(range(120)), 'x_label': 'Adam update', 'y_label': 'nats per observation',
                    'series': [{'name': 'Your negative ELBO', 'values': losses},
                               {'name': 'Reconstruction NLL', 'values': reconstructions},
                               {'name': 'Positive KL', 'values': divergences}]}],
        'metrics': {'initial_negative_elbo': losses[0], 'final_20_mean_negative_elbo': tail,
                    'maximum_decomposition_error': max(abs(a - b - c) for a, b, c in zip(losses, reconstructions, divergences)),
                    'final_20_mean_kl': sum(divergences[-20:]) / 20},
    }
